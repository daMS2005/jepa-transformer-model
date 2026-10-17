"""Training kernels: Triton where it beats torch.compile, PyTorch elsewhere."""

from jepa_transformer.kernels.transformer import Linear, moe_swiglu

__all__ = ["Linear", "moe_swiglu"]
