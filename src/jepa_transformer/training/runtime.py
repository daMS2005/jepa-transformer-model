"""Runtime helpers shared by the two independent training loops."""

from __future__ import annotations

import random
from contextlib import nullcontext

import torch


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if precision == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 autocast is not supported on CPU")
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    enabled = precision == "float16" and device.type == "cuda"
    return torch.amp.GradScaler("cuda", enabled=enabled)
