"""FP32-accurate matrix multiply on TF32 tensor cores (3xTF32).

Each FP32 operand is split into a TF32 high part and a TF32 remainder, and the
product is hi*hi + hi*lo + lo*hi accumulated in FP32: three tensor-core MMAs per
tile instead of FP32 CUDA-core FMAs. Its error stays at or below cuBLAS FP32's,
and on GPUs whose FP32 throughput is far below TF32 (A100, H100) it is faster.
The op is a ``torch.library.triton_op``, so torch.compile traces through it
without a graph break, and its backward is two more of the same multiply. The
weight gradient's output is small while its reduction runs over every token, so
the reduction is split across programs and the partial sums are added in PyTorch.
"""

from functools import lru_cache

import torch as t
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.autotune(configs=[
    triton.Config(dict(BM=bm, BN=bn, BK=bk), num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [(128, 128, 32, 8, 3), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4),
                             (128, 128, 16, 8, 4), (64, 64, 32, 4, 4), (128, 256, 16, 8, 3)]
], key=["M", "N", "K"])
@triton.jit
def _mm(A, B, C, M, N, K, chunk, sam, sak, sbk, sbn, scs, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Grouped launch order: 8 row tiles share each column tile while it is in L2.
    pid = tl.program_id(0)
    group = 8 * tl.cdiv(N, BN)
    first = pid // group * 8
    rows = tl.minimum(tl.cdiv(M, BM) - first, 8)
    m = (first + pid % group % rows) * BM + tl.arange(0, BM)
    n = (pid % group // rows) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(split * chunk, tl.minimum(split * chunk + chunk, K), BK):
        k = start + tl.arange(0, BK)
        a = tl.load(A + m[:, None] * sam + k[None, :] * sak, (m[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * sbk + n[None, :] * sbn, (k[:, None] < K) & (n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc=acc, input_precision="tf32x3")
    tl.store(C + split * scs + m[:, None] * scm + n[None, :] * scn, acc, (m[:, None] < M) & (n[None, :] < N))


@lru_cache(maxsize=None)
def _sm_count(device):
    return t.cuda.get_device_properties(device).multi_processor_count


@triton_op("jepa_transformer::mm_3xtf32", mutates_args=())
def mm(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """[M,K] @ [K,N] in FP32 at FP32 accuracy; operands may be strided views."""
    M, K = a.shape
    N = b.shape[1]
    # Split long reductions until about two waves of 128x128 tiles fill the GPU.
    splits = max(1, min(2 * _sm_count(a.device) // (triton.cdiv(M, 128) * triton.cdiv(N, 128)), K // 1024))
    chunk = triton.cdiv(triton.cdiv(K, splits), 256) * 256
    splits = triton.cdiv(K, chunk)
    c = a.new_empty(splits, M, N)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]), splits)
    wrap_triton(_mm)[grid](a, b, c, M, N, K, chunk, *a.stride(), *b.stride(), *c.stride())
    return c.view(M, N) if splits == 1 else c.sum(0)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    return (mm(grad, b.t()) if ctx.needs_input_grad[0] else None,
            mm(a.t(), grad) if ctx.needs_input_grad[1] else None)


mm.register_autograd(_backward, setup_context=_setup)


def linear(input, weight, bias=None):
    """``F.linear`` with its forward and backward GEMMs in 3xTF32."""
    out = mm(input.reshape(-1, input.shape[-1]), weight.t()).view(*input.shape[:-1], weight.shape[0])
    return out if bias is None else out + bias
