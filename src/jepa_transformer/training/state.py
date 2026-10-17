"""Construction and validation of reproducible training state."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from jepa_transformer.config import JEPAConfig, TransformerConfig
from jepa_transformer.models.transformer import Transformer
from jepa_transformer.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from jepa_transformer.training.jepa import build_jepa


def cache_contract(cache: str | Path) -> dict:
    meta = json.loads((Path(cache) / "meta.json").read_text())
    return {
        "model": meta["model"],
        "spec_sha256": meta["spec_sha256"],
        "features": meta["features"],
        "feature_groups": meta["feature_groups"],
        "mask_units": meta["mask_units"],
        "horizons": meta["horizons"],
    }


def _jepa_config(raw: dict) -> JEPAConfig:
    raw = dict(raw)
    raw["feature_groups"] = tuple(tuple(group) for group in raw.get("feature_groups", ()))
    raw["mask_units"] = tuple(raw.get("mask_units", ()))
    return JEPAConfig(**raw)


def _transformer_config(raw: dict) -> TransformerConfig:
    raw = dict(raw)
    raw["moe_layers"] = tuple(raw.get("moe_layers", ()))
    raw["sequence_lengths"] = tuple(raw["sequence_lengths"])
    return TransformerConfig(**raw)


def save_training_state(
    path: str | Path,
    *,
    model_type: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    config,
    contract: dict,
    metrics: dict,
) -> None:
    save_checkpoint(path, {
        "format_version": 1,
        "model_type": model_type,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": asdict(config),
        "data_contract": contract,
        "metrics": metrics,
        "rng_state": capture_rng_state(),
    })


def restore_training_state(path, *, model_type, model, optimizer, scheduler, config, contract, device) -> int:
    payload = load_checkpoint(path, map_location=device)
    if payload.get("model_type") != model_type:
        raise ValueError(f"checkpoint is not a {model_type} checkpoint")
    if payload.get("config") != asdict(config) or payload.get("data_contract") != contract:
        raise ValueError("checkpoint configuration or data contract differs from this run")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload["rng_state"])
    return int(payload["epoch"]) + 1


def load_jepa_for_inference(path: str | Path, *, device: torch.device, contract: dict):
    payload = load_checkpoint(path, map_location=device)
    if payload.get("model_type") != "jepa" or payload.get("data_contract") != contract:
        raise ValueError("JEPA checkpoint does not match this cache")
    config = _jepa_config(payload["config"])
    model = build_jepa(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def load_transformer_for_inference(path: str | Path, *, device: torch.device, contract: dict):
    payload = load_checkpoint(path, map_location=device)
    if payload.get("model_type") != "transformer" or payload.get("data_contract") != contract:
        raise ValueError("Transformer checkpoint does not match this cache")
    config = _transformer_config(payload["config"])
    model = Transformer(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config
