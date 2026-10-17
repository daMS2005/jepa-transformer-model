"""Rotary positions for chronological daily representations."""

import torch as t
from torch import nn
from torch.nn import functional as F


def rotary(input, cos, sin):
    """Rotate adjacent pairs of [B,H,T,D] by fixed [T,D/2] angles."""
    even, odd = input[..., 0::2], input[..., 1::2]
    return t.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def rms_rotary(input, weight, cos, sin, eps):
    """QK RMSNorm then RoPE, rounding the normalized values to the Q/K dtype first."""
    return rotary(F.rms_norm(input, input.shape[-1:], weight, eps).to(input.dtype), cos, sin)


class RoPE(nn.Module):
    """Rotate adjacent head-coordinate pairs at absolute sequence positions."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.base = base
        self.head_dim = head_dim
        self._angles = None

    def forward(self, input, past_len=0, *, norm=None):
        """Optionally apply QK RMSNorm first; cache only the most recent angle table."""
        T, D = input.shape[-2:]
        dtype = t.float64 if input.dtype == t.float64 else t.float32
        key = (self.base, D, T, past_len, input.device, input.dtype)
        if self._angles is None or self._angles[0] != key:
            # Inference-created tables must also be safe to save for training backward.
            with t.inference_mode(False), t.no_grad(), t.autocast(device_type=input.device.type, enabled=False):
                freq = self.base ** (-t.arange(0, D, 2, device=input.device, dtype=dtype) / D)
                pos = t.arange(past_len, past_len + T, device=input.device, dtype=dtype)
                angle = pos[:, None] * freq[None, :]
                cos, sin = angle.cos().to(input.dtype), angle.sin().to(input.dtype)
            self._angles = (key, cos, sin)
        _, cos, sin = self._angles
        if norm is not None:
            return rms_rotary(input, norm.weight, cos, sin, norm.eps)
        return rotary(input, cos, sin)
