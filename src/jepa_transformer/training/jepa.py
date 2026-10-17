"""Independent-day JEPA optimization."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from jepa_transformer.config import JEPAConfig
from jepa_transformer.models.jepa import JEPA
from jepa_transformer.training.runtime import autocast_context, make_grad_scaler


def build_jepa(config: JEPAConfig) -> JEPA:
    return JEPA(
        config.input_dim,
        config.output_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        dropout=config.dropout,
        feature_groups=config.feature_groups,
        input_encoder_dim=config.input_encoder_dim,
        input_encoder_depth=config.input_encoder_depth,
        residual=config.residual,
        mask_units=config.mask_units,
        mask_ratio=config.mask_ratio,
        target_momentum=config.target_momentum,
    )


def latent_loss(pred: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if not valid.any():
        raise ValueError("a JEPA batch must contain at least one maskable row")
    return F.l1_loss(pred[valid], target.detach()[valid])


def train_jepa_epoch(
    model: JEPA,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[Tensor],
    *,
    device: torch.device,
    precision: str = "float32",
    accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
) -> float:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    scaler = make_grad_scaler(device, precision)
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    sample_count = 0
    pending = 0
    for batch_index, batch in enumerate(batches):
        batch = batch.to(device, non_blocking=True)
        with autocast_context(device, precision):
            result = model(batch)
            loss = latent_loss(result.pred, result.target, result.valid_samples)
        if not torch.isfinite(loss):
            raise ValueError("non-finite JEPA loss")
        scaler.scale(loss / accumulation_steps).backward()
        valid = int(result.valid_samples.sum().item())
        weighted_loss += float(loss.detach()) * valid
        sample_count += valid
        pending += 1
        if pending == accumulation_steps:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            model.update_target_encoder()
            pending = 0
    if pending:
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        model.update_target_encoder()
    if sample_count == 0:
        raise ValueError("JEPA epoch contained no valid samples")
    return weighted_loss / sample_count


@torch.no_grad()
def evaluate_jepa(
    model: JEPA,
    batches: Iterable[Tensor],
    *,
    device: torch.device,
    precision: str = "float32",
) -> float:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    try:
        for batch in batches:
            batch = batch.to(device, non_blocking=True)
            with autocast_context(device, precision):
                result = model(batch)
                loss = latent_loss(result.pred, result.target, result.valid_samples)
            valid = int(result.valid_samples.sum().item())
            weighted_loss += float(loss) * valid
            sample_count += valid
    finally:
        model.train(was_training)
    if sample_count == 0:
        raise ValueError("JEPA evaluation contained no valid samples")
    return weighted_loss / sample_count
