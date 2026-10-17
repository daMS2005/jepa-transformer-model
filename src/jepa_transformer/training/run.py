"""Model training stages as plain functions: JEPA pretraining, Transformer training, evaluation.

Architecture and optimization come from a preset (configs/model/*.toml); the cache sets the
feature width, groups and masking units. The Transformer reads raw feature windows and runs
them through the frozen JEPA on the fly. Each training stage validates on the validation era's
holdout companies every epoch, writes ``<name>.pt`` (last epoch) and ``<name>.best.pt`` (best
holdout epoch) under ``out``, and stops after ``patience`` epochs without improvement.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import random
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_transformer.config import ModelConfig, load_model_config
from jepa_transformer.data.dataset import JEPADataset, SequenceDataset
from jepa_transformer.models.transformer import Transformer
from jepa_transformer.training.jepa import build_jepa, evaluate_jepa, train_jepa_epoch
from jepa_transformer.training.runtime import autocast_context, resolve_device, seed_everything
from jepa_transformer.training.state import (
    cache_contract,
    load_jepa_for_inference,
    load_transformer_for_inference,
    restore_training_state,
    save_training_state,
)
from jepa_transformer.training.transformer import evaluate_transformer, forecast_ic, train_transformer_epoch

LOGGER = logging.getLogger(__name__)
VALIDATION = ("holdout",)  # model selection never sees the companies it trains on


def _preset(preset) -> ModelConfig:
    return preset if isinstance(preset, ModelConfig) else load_model_config(preset)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _loader(dataset, batch_size, shuffle, workers, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def _limit(batches, limit):
    return batches if limit is None else itertools.islice(batches, limit)


def _interleave(loaders, seed: int):
    """Yield batches from every loader in a random order, so each step trains one sequence length."""
    iterators = [iter(loader) for loader in loaders]
    remaining = [len(loader) for loader in loaders]
    rng = random.Random(seed)
    while sum(remaining):
        i = rng.choices(range(len(iterators)), weights=remaining)[0]
        remaining[i] -= 1
        yield next(iterators[i])


def _encoded(batches, jepa, device, precision):
    """Run raw [B, T, F] feature windows through the frozen JEPA, giving [B, T, d] representations."""
    for x, y, valid in batches:
        x = x.to(device, non_blocking=True)
        with torch.no_grad(), autocast_context(device, precision):
            z = jepa.encode(x.flatten(0, 1)).view(*x.shape[:2], -1)
        yield z.float(), y, valid


def _epochs(name, out, model_type, model, optimizer, scheduler, config, contract, start, patience, step) -> Path:
    """Run epochs from ``start``; ``step(epoch)`` trains one and returns (holdout score, metrics)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    last, best_path = out / f"{name}.pt", out / f"{name}.best.pt"
    best = (float("inf"), start - 1)
    for epoch in range(start, config.num_epochs):
        score, metrics = step(epoch)
        scheduler.step()
        save_training_state(last, model_type=model_type, model=model, optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch, config=config, contract=contract, metrics=metrics)
        if score < best[0]:
            shutil.copyfile(last, best_path)
            best = (score, epoch)
        LOGGER.info("%s epoch=%d %s best_epoch=%d", name, epoch + 1, metrics, best[1] + 1)
        if epoch - best[1] >= patience:
            LOGGER.info("%s: no holdout improvement for %d epochs; stopping", name, patience)
            break
    return best_path


def train_jepa(preset, cache, out, *, epochs: int | None = None, patience: int = 5, batch_size: int = 2048,
               accumulation_steps: int = 1, limit_batches: int | None = None, device: str | None = None,
               precision: str | None = None, workers: int | None = None, resume: bool = False) -> Path:
    """Pretrain the daily JEPA; return the best checkpoint (``out/jepa.best.pt``)."""
    preset = _preset(preset)
    device = resolve_device(device or preset.runtime.device)
    precision = precision or preset.runtime.precision
    workers = preset.runtime.num_workers if workers is None else workers
    seed_everything(preset.runtime.seed)
    contract = cache_contract(cache)
    config = replace(
        preset.jepa,
        input_dim=len(contract["features"]),
        feature_groups=tuple((name, int(width)) for name, width in contract["feature_groups"]),
        mask_units=tuple(contract["mask_units"]),
        num_epochs=epochs or preset.jepa.num_epochs,
    )
    model = build_jepa(config).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs,
                                                           eta_min=config.min_learning_rate)
    start = 0
    if resume and (Path(out) / "jepa.pt").exists():
        start = restore_training_state(Path(out) / "jepa.pt", model_type="jepa", model=model, optimizer=optimizer,
                                       scheduler=scheduler, config=config, contract=contract, device=device)
    train = _loader(JEPADataset(cache, "train"), batch_size, True, workers, device)
    validation = _loader(JEPADataset(cache, "val", groups=VALIDATION), batch_size, False, workers, device)

    def step(epoch):
        train_loss = train_jepa_epoch(model, optimizer, _limit(train, limit_batches), device=device,
                                      precision=precision, accumulation_steps=accumulation_steps,
                                      max_grad_norm=config.max_grad_norm)
        holdout = evaluate_jepa(model, _limit(validation, limit_batches), device=device, precision=precision)
        return holdout, {"train_loss": round(train_loss, 6), "holdout_validation_loss": round(holdout, 6)}

    return _epochs("jepa", out, "jepa", model, optimizer, scheduler, config, contract, start, patience, step)


def train_transformer(preset, cache, out, *, jepa_checkpoint=None, epochs: int | None = None, patience: int = 5,
                      batch_size: int = 8, accumulation_steps: int = 4, stride: int = 64, eval_length: int = 252,
                      kernel_backend: str | None = None, limit_batches: int | None = None, device: str | None = None,
                      precision: str | None = None, workers: int | None = None, resume: bool = False) -> Path:
    """Train the temporal forecaster on the frozen JEPA (default ``out/jepa.best.pt``); return its best checkpoint."""
    preset = _preset(preset)
    device = resolve_device(device or preset.runtime.device)
    precision = precision or preset.runtime.precision
    workers = preset.runtime.num_workers if workers is None else workers
    seed_everything(preset.runtime.seed)
    contract = cache_contract(cache)
    jepa_checkpoint = Path(jepa_checkpoint or Path(out) / "jepa.best.pt")
    jepa, jepa_config = load_jepa_for_inference(jepa_checkpoint, device=device, contract=contract)
    config = replace(
        preset.transformer,
        input_dim=jepa_config.output_dim,
        target_dim=len(contract["horizons"]),
        jepa_checkpoint=f"sha256:{_sha256(jepa_checkpoint)}",
        num_epochs=epochs or preset.transformer.num_epochs,
        kernel_backend=kernel_backend or preset.transformer.kernel_backend,
    )
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs,
                                                           eta_min=config.min_learning_rate)
    start = 0
    if resume and (Path(out) / "transformer.pt").exists():
        start = restore_training_state(Path(out) / "transformer.pt", model_type="transformer", model=model,
                                       optimizer=optimizer, scheduler=scheduler, config=config, contract=contract,
                                       device=device)
    train = [_loader(SequenceDataset(cache, "train", length, stride=min(stride, length)), batch_size, True, workers,
                     device) for length in config.sequence_lengths]
    eval_length = eval_length if eval_length in config.sequence_lengths else max(config.sequence_lengths)
    validation_data = SequenceDataset(cache, "val", eval_length, stride=min(stride, eval_length), groups=VALIDATION)
    validation = _loader(validation_data, batch_size, False, workers, device)
    LOGGER.info("windows train=%s holdout_val=%d (length %d) skipped_val_targets=%d",
                {length: len(loader.dataset) for length, loader in zip(config.sequence_lengths, train)},
                len(validation_data), eval_length, validation_data.skipped_targets)

    def step(epoch):
        batches = _encoded(_limit(_interleave(train, preset.runtime.seed + epoch), limit_batches), jepa, device,
                           precision)
        train_loss = train_transformer_epoch(model, optimizer, batches, device=device, precision=precision,
                                             accumulation_steps=accumulation_steps, max_grad_norm=config.max_grad_norm)
        holdout = evaluate_transformer(model, _encoded(_limit(validation, limit_batches), jepa, device, precision),
                                       device=device, precision=precision)
        ic = forecast_ic(model, validation_data, jepa=jepa, device=device, precision=precision,
                         batch_size=batch_size, limit_batches=limit_batches)
        return holdout["loss"], {"train_loss": round(train_loss, 6), "holdout_validation_loss": round(holdout["loss"], 6),
                                 "rank_ic": [round(v, 4) for v in ic["ic"]], "ic_t": [round(v, 1) for v in ic["ic_t"]]}

    return _epochs("transformer", out, "transformer", model, optimizer, scheduler, config, contract, start, patience,
                   step)


def evaluate(cache, out, *, split: str = "val", groups=VALIDATION, length: int = 252, stride: int = 64,
             batch_size: int = 8, device: str | None = None, precision: str | None = None, workers: int = 0) -> dict:
    """Loss, MAE and rank IC of ``out/transformer.best.pt`` (on ``out/jepa.best.pt``) for one split and company groups."""
    device = resolve_device(device or "auto")
    precision = precision or "float32"
    contract = cache_contract(cache)
    jepa_checkpoint = Path(out) / "jepa.best.pt"
    jepa, _ = load_jepa_for_inference(jepa_checkpoint, device=device, contract=contract)
    model, config = load_transformer_for_inference(Path(out) / "transformer.best.pt", device=device, contract=contract)
    if config.jepa_checkpoint != f"sha256:{_sha256(jepa_checkpoint)}":
        raise ValueError("the Transformer was trained on a different JEPA checkpoint")
    dataset = SequenceDataset(cache, split, length, stride=min(stride, length), groups=groups)
    batches = _encoded(_loader(dataset, batch_size, False, workers, device), jepa, device, precision)
    metrics = evaluate_transformer(model, batches, device=device, precision=precision)
    metrics["rank_ic"] = forecast_ic(model, dataset, jepa=jepa, device=device, precision=precision,
                                     batch_size=batch_size)
    return metrics
