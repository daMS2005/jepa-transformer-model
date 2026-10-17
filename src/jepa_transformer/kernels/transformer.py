"""Transformer kernel dispatch: MoE experts and strict-FP32 linear layers run on Triton.

RMSNorm, QK-norm + RoPE and SwiGLU stay in PyTorch: their Triton versions did not
beat torch.compile, which fuses these ops itself.
"""

import torch as t
from torch import nn


@t.compiler.assume_constant_result  # Dynamo cannot trace the flag; a compiled graph keeps its compile-time value
def _tf32():
    return t.backends.cuda.matmul.fp32_precision == "tf32"


class Linear(nn.Linear):
    """``nn.Linear`` whose strict-FP32 GEMMs run as 3xTF32 on tensor cores; same parameters.

    Under autocast or with TF32 enabled it is plain ``nn.Linear``: those modes already
    trade FP32 accuracy for speed, and cuBLAS serves them best.
    """

    def forward(self, input):
        if input.dtype != t.float32 or t.is_autocast_enabled("cuda") or _tf32():
            return super().forward(input)
        from jepa_transformer.kernels.triton.linear import linear

        return linear(input, self.weight, self.bias)


@t.compiler.disable  # Dynamo cannot trace the autotuned autograd.Function; compiled models graph-break here
def moe_swiglu(input, up, gate, down, indices, weights):
    """Router-weighted top-k SwiGLU expert mixture [N,C] on the Triton expert kernels.

    ``up``/``gate`` are stacked [E,H,C] weights, ``down`` is [E,C,H], ``weights`` the
    router's FP32 [N,k] gates.
    """
    from jepa_transformer.kernels.triton.moe import moe_swiglu as kernel

    if t.is_autocast_enabled("cuda"):  # cast here so gradients reach the FP32 master weights
        dtype = t.get_autocast_dtype("cuda")
        up, gate, down = up.to(dtype), gate.to(dtype), down.to(dtype)
    return kernel(input, up, gate, down, indices, weights)
