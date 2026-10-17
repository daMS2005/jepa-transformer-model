# Custom training kernels

A custom Triton kernel is kept only where it measured faster than the best
off-the-shelf option (`torch.compile`, cuBLAS, SDPA/FlashAttention) on an L4.
Everything else is plain PyTorch, which `torch.compile` fuses on its own.

| Operation | Implementation | vs `torch.compile` (L4, A100) |
|---|---|---|
| Attention | `F.scaled_dot_product_attention` (FlashAttention on sm80+) | baseline |
| RMSNorm, QK-norm + RoPE, SwiGLU | PyTorch | Triton versions tied or lost; removed |
| MoE experts | Triton | up to 3.6x per layer; full model up to 2.4x at width <= 384, about 1x at 768+ |
| Strict-FP32 linear layers | Triton 3xTF32 | 1.3-1.5x per GEMM on A100, lower error than cuBLAS FP32; full model 1.4-1.9x |
| JEPA encoder | PyTorch | a fused first-block kernel gained at most 1.12x in FP32; removed |

## MoE experts

Routed slots are sorted by expert on the GPU and padded to 64-row tiles, without
host synchronization. For FP16/BF16 models up to 256 wide, one fused kernel per
tile loads the tokens once, streams the hidden dimension through up/gate, SwiGLU
and the down projection in registers, and applies the router weight in its
epilogue, so no up/gate/activation tensor is written; backward recomputes
up/gate. FP32 lacks the half-precision tensor cores that make that recomputation
cheap, so it runs grouped GEMMs that store up/gate for backward; wider FP16/BF16
models do too, since their token rows do not fit on-chip. Expert-weight
gradients use split, non-atomic reductions. Tile shapes are autotuned per GPU
for each model shape and power-of-two token-count bucket; strict FP32 (TF32 off)
autotunes IEEE FP32 against 3xTF32 emulation, whose error stays at FP32 level.

```python
from dataclasses import replace
from jepa_transformer import Transformer

model = torch.compile(Transformer(replace(config, kernel_backend="triton")).cuda())
```

The kernel is `torch.compiler.disable`d, so a compiled model graph-breaks around
it and compiles everything else. Parameter names and shapes do not depend on the
backend, so state dictionaries load across backends with `strict=True`. Training
targets NVIDIA GPUs (sm80+) with Triton installed.

## Strict-FP32 linear layers

With `kernel_backend="triton"`, the Transformer's linear layers (embedding,
attention projections, dense SwiGLU) run FP32 inputs outside autocast and with
TF32 off as 3xTF32 matmuls: each operand is split into a TF32 high part and a TF32
remainder, and hi*hi + hi*lo + lo*hi accumulate in FP32 on tensor cores. Measured
against FP64, its error (2e-7) is below cuBLAS FP32's (5e-7); on an A100, whose
FP32 CUDA-core rate is an eighth of its TF32 rate, it is also 1.3-1.5x faster, and
strict-FP32 training of the 15M/39M/100M-parameter presets runs 1.9x/1.5x/1.4x
faster than compiled PyTorch FP32. On an L4 it matches cuBLAS speed. The matmul is
a `torch.library.triton_op`, so `torch.compile` traces through it without a graph
break; weight gradients, whose reduction runs over every token, split that
reduction across programs. Under autocast or with TF32 enabled the layers are
plain `nn.Linear`.

## Measurement

Every kernel was checked for forward and gradient parity against PyTorch on an
NVIDIA L4 and timed against `torch.compile`, with warmups that include
compilation. Beating eager PyTorch was not enough to keep a kernel: an earlier
sparse-gather JEPA kernel was 30-100x slower than PyTorch, and the Transformer
normalization/activation kernels never beat compile.
