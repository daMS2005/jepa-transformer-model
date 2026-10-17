"""SwiGLU expert MLP for top-k MoE routing.

Routed slots are sorted by expert on the GPU and padded per expert to row tiles,
so each program owns one expert's tile and the host never waits for token counts.
Narrow FP16/BF16 models (width <= 256) run a fused expert MLP: one kernel per tile
loads the tokens once, streams the hidden dimension through up, gate, SwiGLU and the
down projection, and applies the router weight in its epilogue, so up/gate/activations
are never written; backward recomputes up/gate. FP32 has no half-precision tensor
cores to make that recomputation and the fused kernels' narrow tiles cheap, so FP32
runs grouped GEMMs that keep up/gate for backward instead; so do wider FP16/BF16
models, whose token rows do not fit on-chip.
Weight gradients use split per-expert reductions summed in PyTorch, without atomics.
Tile shapes are autotuned per GPU, keyed by model shape and a power-of-two
token-count bucket; buckets are runtime arguments and never trigger recompilation.
Strict FP32 (TF32 disabled) autotunes between IEEE FP32 and 3xTF32 tensor-core
emulation, whose error stays at FP32 level; TF32-enabled FP32 uses TF32.
"""

from functools import lru_cache

import torch as t
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


_TILE = 64  # rows per expert tile; the host pads each expert's slots to whole tiles


def _bucket(n_slots):
    return triton.next_power_of_2(max(n_slots, 1024))


def _bc(C):
    return max(16, triton.next_power_of_2(C))


def _configs(names, rows):
    return [triton.Config(dict(zip(names, tiles)), num_warps=w, num_stages=s) for *tiles, w, s in rows]


# The FP32 grouped kernels autotune their dot precision too: strict FP32 picks IEEE or
# 3xTF32 (FP32-level error on tensor cores); with TF32 enabled it uses TF32.
_PRECISIONS = {False: ("ieee", "tf32x3"), True: ("tf32",)}


def _fp32_configs(names, rows):
    return [triton.Config({**c.kwargs, "PREC": prec}, num_warps=c.num_warps, num_stages=c.num_stages)
            for prec in ("tf32", "ieee", "tf32x3") for c in _configs(names, rows)]


def _keep_precision(configs, named_args, **kwargs):
    return [c for c in configs if c.kwargs["PREC"] in _PRECISIONS[named_args["tf32"]]]


_ROW_CONFIGS = _configs(("BH",), [(16, 4, 2), (16, 8, 2), (32, 4, 2), (32, 8, 2), (32, 8, 3), (64, 8, 2), (64, 8, 3)])
_TUNING = dict(key=["C", "H", "bucket"])


@triton.jit
def _silu_parts(gate, dtype):
    """Return gate, sigmoid(gate) and silu(gate) in FP32, rounding silu like PyTorch's op."""
    g = gate.to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-g))
    return g, sigmoid, (g * sigmoid).to(dtype).to(tl.float32)


@triton.autotune(configs=_ROW_CONFIGS, **_TUNING)
@triton.jit
def _mlp_fwd(X, Wu, Wg, Wd, RW, Out, Slots, BlockExpert, n_slots, n_experts, C, H, div,
             sx0, sx1, swe, swh, swc, sde, sdc, sdh, bucket,
             BM: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr):
    """Out[s] = RW[s] * (silu(x Wg^T) * x Wu^T) Wd^T for one expert tile, x = X[s // div]."""
    block = tl.program_id(0)
    expert = tl.load(BlockExpert + block)
    if expert < n_experts:
        slot = tl.load(Slots + block * BM + tl.arange(0, BM))
        valid = slot < n_slots
        c = tl.arange(0, BC)
        rows = valid[:, None] & (c[None, :] < C)
        dtype = Wu.dtype.element_ty
        x = tl.load(X + (slot // div)[:, None] * sx0 + c[None, :] * sx1, rows, other=0).to(dtype)
        acc = tl.zeros((BM, BC), tl.float32)
        for start in range(0, H, BH):
            h = start + tl.arange(0, BH)
            w_mask = (c[:, None] < C) & (h[None, :] < H)
            w_off = expert * swe + h[None, :] * swh + c[:, None] * swc
            up = tl.dot(x, tl.load(Wu + w_off, w_mask, other=0)).to(dtype)
            _, _, silu = _silu_parts(tl.dot(x, tl.load(Wg + w_off, w_mask, other=0)).to(dtype), dtype)
            act = (silu * up.to(tl.float32)).to(dtype)
            wd = tl.load(Wd + expert * sde + h[:, None] * sdh + c[None, :] * sdc,
                         (h[:, None] < H) & (c[None, :] < C), other=0)
            acc = tl.dot(act, wd, acc=acc)
        weight = tl.load(RW + slot, valid, other=0)
        tl.store(Out + slot[:, None] * C + c[None, :], acc.to(dtype).to(tl.float32) * weight[:, None], rows)


@triton.autotune(configs=_ROW_CONFIGS, **_TUNING)
@triton.jit
def _mlp_bwd_dx(X, Wu, Wg, Wd, RW, G, DX, DRW, Slots, BlockExpert, n_slots, n_experts, C, H, div,
                sx0, sx1, swe, swh, swc, sde, sdc, sdh, sg0, sg1, bucket,
                BM: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr):
    """Input and router-weight gradients for one expert tile, recomputing up/gate."""
    block = tl.program_id(0)
    expert = tl.load(BlockExpert + block)
    if expert < n_experts:
        slot = tl.load(Slots + block * BM + tl.arange(0, BM))
        valid = slot < n_slots
        row = slot // div
        c = tl.arange(0, BC)
        rows = valid[:, None] & (c[None, :] < C)
        dtype = Wu.dtype.element_ty
        x = tl.load(X + row[:, None] * sx0 + c[None, :] * sx1, rows, other=0).to(dtype)
        grad = tl.load(G + row[:, None] * sg0 + c[None, :] * sg1, rows, other=0).to(tl.float32)
        # Weight before rounding, as PyTorch's expert path does, so large upstream gradients stay finite.
        dy = (grad * tl.load(RW + slot, valid, other=0)[:, None]).to(dtype)
        dx = tl.zeros((BM, BC), tl.float32)
        y = tl.zeros((BM, BC), tl.float32)
        for start in range(0, H, BH):
            h = start + tl.arange(0, BH)
            w_mask = (c[:, None] < C) & (h[None, :] < H)
            w_off = expert * swe + h[None, :] * swh + c[:, None] * swc
            wu = tl.load(Wu + w_off, w_mask, other=0)
            wg = tl.load(Wg + w_off, w_mask, other=0)
            wd = tl.load(Wd + expert * sde + c[:, None] * sdc + h[None, :] * sdh, w_mask, other=0)
            up = tl.dot(x, wu).to(dtype).to(tl.float32)
            g, sigmoid, silu = _silu_parts(tl.dot(x, wg).to(dtype), dtype)
            act = (silu * up).to(dtype)
            y = tl.dot(act, tl.trans(wd), acc=y)  # expert output, for the router gradient
            d_act = tl.dot(dy, wd).to(dtype).to(tl.float32)
            d_up = (d_act * silu).to(dtype)
            d_gate = ((d_act * up).to(dtype).to(tl.float32) * sigmoid * (1.0 + g * (1.0 - sigmoid))).to(dtype)
            dx = tl.dot(d_up, tl.trans(wu), acc=dx)
            dx = tl.dot(d_gate, tl.trans(wg), acc=dx)
        tl.store(DX + slot[:, None] * C + c[None, :], dx, rows)
        tl.store(DRW + slot, tl.sum(grad * y.to(dtype).to(tl.float32), axis=1), valid)


@triton.autotune(configs=_configs(("BH", "BS"), [
    (16, 16, 4, 2), (16, 32, 4, 2), (16, 16, 8, 2), (16, 32, 8, 2), (32, 16, 4, 2), (32, 32, 8, 2),
]), **_TUNING)
@triton.jit
def _mlp_bwd_dw(X, Wu, Wg, Wd, RW, G, DWu, DWg, DWd, Order, Start, Count, C, H, div, splits,
                sx0, sx1, swe, swh, swc, sde, sdc, sdh, sg0, sg1, bucket,
                BC: tl.constexpr, BH: tl.constexpr, BS: tl.constexpr):
    """Partial expert-weight gradients for one hidden chunk over one split of an expert's slots."""
    expert, split = tl.program_id(0) // splits, tl.program_id(0) % splits
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    c = tl.arange(0, BC)
    w_mask = (c[:, None] < C) & (h[None, :] < H)
    w_off = expert * swe + h[None, :] * swh + c[:, None] * swc
    wd_off = expert * sde + c[:, None] * sdc + h[None, :] * sdh
    dtype = Wu.dtype.element_ty
    # All three accumulators are [BC, BH]: the up/gate gradients are accumulated transposed so
    # one x^T serves both and every product shares a layout.
    d_wu = tl.zeros((BC, BH), tl.float32)
    d_wg = tl.zeros((BC, BH), tl.float32)
    d_wd = tl.zeros((BC, BH), tl.float32)
    start = tl.load(Start + expert)
    count = tl.load(Count + expert)
    chunk = tl.cdiv(tl.cdiv(count, splits), BS) * BS
    lo = split * chunk
    hi = tl.minimum(lo + chunk, count)
    for i in range(lo, hi, BS):
        r = i + tl.arange(0, BS)
        active = r < hi
        slot = tl.load(Order + start + r, active, other=0)
        row = slot // div
        rows = active[:, None] & (c[None, :] < C)
        x = tl.load(X + row[:, None] * sx0 + c[None, :] * sx1, rows, other=0).to(dtype)
        grad = tl.load(G + row[:, None] * sg0 + c[None, :] * sg1, rows, other=0).to(tl.float32)
        dy = (grad * tl.load(RW + slot, active, other=0)[:, None]).to(dtype)
        # Weight chunks are re-read from L2 each step rather than held in registers.
        up = tl.dot(x, tl.load(Wu + w_off, w_mask, other=0)).to(dtype).to(tl.float32)
        g, sigmoid, silu = _silu_parts(
            tl.dot(x, tl.load(Wg + w_off, w_mask, other=0)).to(dtype), dtype)
        act = (silu * up).to(dtype)
        d_act = tl.dot(dy, tl.load(Wd + wd_off, w_mask, other=0)).to(dtype).to(tl.float32)
        d_up = (d_act * silu).to(dtype)
        d_gate = ((d_act * up).to(dtype).to(tl.float32) * sigmoid * (1.0 + g * (1.0 - sigmoid))).to(dtype)
        x_t = tl.trans(x)
        d_wu = tl.dot(x_t, d_up, acc=d_wu)
        d_wg = tl.dot(x_t, d_gate, acc=d_wg)
        d_wd = tl.dot(tl.trans(dy), act, acc=d_wd)
    base = (split * (tl.num_programs(0) // splits) + expert) * H * C  # partials are [splits, E, H, C]
    tl.store(DWu + base + h[None, :] * C + c[:, None], d_wu, w_mask)
    tl.store(DWg + base + h[None, :] * C + c[:, None], d_wg, w_mask)
    tl.store(DWd + base + c[:, None] * H + h[None, :], d_wd, w_mask)


@triton.autotune(configs=_fp32_configs(("BN", "BK"), [
    (32, 32, 4, 3), (32, 32, 8, 3), (64, 32, 4, 2), (64, 32, 4, 3), (64, 64, 4, 3),
    (64, 32, 8, 3), (64, 64, 8, 3), (128, 32, 8, 3), (128, 64, 8, 3), (128, 64, 8, 4),
    (128, 64, 4, 4), (128, 32, 4, 5), (256, 32, 8, 3), (256, 64, 8, 3),
]), key=["K", "N", "MODE", "BM", "bucket", "tf32"], prune_configs_by={"early_config_prune": _keep_precision})
@triton.jit
def _grouped_mm(A, W, Out, Aux, Slots, BlockExpert, n_slots, n_experts, K, N, div,
                sa0, sa1, swe, swk, swn, so0, saux, bucket, tf32,
                MODE: tl.constexpr, PREC: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Out[s] = A[s // div] @ W[e]. MODE 1 fuses SwiGLU; MODE 2 fuses its backward."""
    block = tl.program_id(0)
    expert = tl.load(BlockExpert + block)
    if expert < n_experts:
        slot = tl.load(Slots + block * BM + tl.arange(0, BM))
        valid = slot < n_slots
        row = slot // div
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        if MODE == 1:
            acc_gate = tl.zeros((BM, BN), tl.float32)
        for start in range(0, K, BK):
            k = start + tl.arange(0, BK)
            a = tl.load(A + row[:, None] * sa0 + k[None, :] * sa1,
                        valid[:, None] & (k[None, :] < K), other=0)
            w_ptr = W + expert * swe + k[:, None] * swk + n[None, :] * swn
            w_mask = (k[:, None] < K) & (n[None, :] < N)
            acc = tl.dot(a, tl.load(w_ptr, w_mask, other=0), acc=acc, input_precision=PREC)
            if MODE == 1:
                # Gate rows follow the N up-projection rows in the stacked weight.
                acc_gate = tl.dot(a, tl.load(w_ptr + N * swn, w_mask, other=0),
                                  acc=acc_gate, input_precision=PREC)
        mask = valid[:, None] & (n[None, :] < N)
        out = Out + slot[:, None] * so0 + n[None, :]
        if MODE == 0:
            tl.store(out, acc, mask)
        elif MODE == 1:
            # Aux keeps [up | gate] for backward.
            aux = Aux + slot[:, None] * saux + n[None, :]
            tl.store(aux, acc, mask)
            tl.store(aux + N, acc_gate, mask)
            tl.store(out, acc_gate / (1.0 + tl.exp(-acc_gate)) * acc, mask)
        else:
            # acc is the SwiGLU output gradient; Out receives [d_up | d_gate].
            aux = Aux + slot[:, None] * saux + n[None, :]
            up = tl.load(aux, mask, other=0).to(tl.float32)
            g = tl.load(aux + N, mask, other=0).to(tl.float32)
            sigmoid = 1.0 / (1.0 + tl.exp(-g))
            tl.store(out, acc * g * sigmoid, mask)
            tl.store(out + N, acc * up * sigmoid * (1.0 + g * (1.0 - sigmoid)), mask)


@triton.autotune(configs=_fp32_configs(("BM", "BN", "BK"), [
    (64, 64, 32, 4, 2), (64, 64, 32, 4, 3), (64, 64, 64, 4, 3), (128, 64, 32, 4, 3),
    (64, 128, 32, 4, 3), (128, 128, 32, 8, 2), (128, 128, 32, 8, 3), (128, 128, 64, 8, 3),
    (128, 128, 64, 8, 4), (128, 256, 32, 8, 3), (256, 128, 32, 8, 3),
]), key=["M", "N", "bucket", "tf32"], prune_configs_by={"early_config_prune": _keep_precision})
@triton.jit
def _expert_wgrad(A, B, Out, Order, Start, Count, M, N, div_a, div_b, splits,
                  sa0, sa1, sb0, sb1, sop, so0, so1, bucket, tf32,
                  PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Out[split, e] = sum of A[s // div_a].T @ B[s // div_b] over one split of e's slots."""
    expert, split = tl.program_id(0) // splits, tl.program_id(0) % splits
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    start = tl.load(Start + expert)
    count = tl.load(Count + expert)
    chunk = tl.cdiv(tl.cdiv(count, splits), BK) * BK
    lo = split * chunk
    hi = tl.minimum(lo + chunk, count)
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(lo, hi, BK):
        r = i + tl.arange(0, BK)
        active = r < hi
        slot = tl.load(Order + start + r, active, other=0)
        a = tl.load(A + (slot // div_a)[None, :] * sa0 + m[:, None] * sa1,
                    active[None, :] & (m[:, None] < M), other=0)
        b = tl.load(B + (slot // div_b)[:, None] * sb0 + n[None, :] * sb1,
                    active[:, None] & (n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc=acc, input_precision=PREC)
    tl.store(Out + split * sop + expert * so0 + m[:, None] * so1 + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def _route(indices, n_experts, BM):
    """Sort slots by expert and pad each expert's segment to whole BM-row tiles."""
    flat = indices.reshape(-1)
    n_slots = flat.numel()
    # int32 indices halve the registers spent on per-row offsets inside the tiles.
    order = flat.argsort(stable=True).int()
    # scatter_add_ instead of bincount: CUDA bincount synchronizes to size its output.
    counts = flat.new_zeros(n_experts).scatter_add_(0, flat, t.ones_like(flat))
    starts = counts.cumsum(0) - counts
    padded = (counts + BM - 1) // BM * BM
    padded_ends = padded.cumsum(0)
    expert = flat[order]
    rank = t.arange(n_slots, device=flat.device) - starts[expert]
    capacity = triton.cdiv(n_slots + n_experts * (BM - 1), BM) * BM
    slots = order.new_full((capacity,), n_slots)
    slots[padded_ends[expert] - padded[expert] + rank] = order
    # Tail blocks past the last padded segment map to n_experts and exit early.
    block_expert = t.searchsorted(padded_ends, t.arange(0, capacity, BM, device=flat.device),
                                  right=True, out_int32=True)
    return slots, block_expert, order, starts.int(), counts.int()


def _tf32():
    return t.backends.cuda.matmul.fp32_precision == "tf32"


@lru_cache(maxsize=None)
def _sm_count(device):
    return t.cuda.get_device_properties(device).multi_processor_count


def _splits(device, tiles, n_slots):
    """Split each expert's slots until about four waves of programs fill the GPU,
    keeping at least 512 slots (summed over experts) per split."""
    return max(1, min(triton.cdiv(4 * _sm_count(device), tiles), n_slots // 512))


def _weight_grads(x, w_up, w_gate, w_down, rw, grad, order, starts, counts, k):
    E, H, C = w_up.shape
    splits = _splits(x.device, E * triton.cdiv(H, 32), order.numel())  # 32-wide hidden chunks estimate the grid
    parts = t.empty((3, splits, E, H, C), device=x.device, dtype=t.float32)
    grid = lambda meta: (E * splits, triton.cdiv(H, meta["BH"]))
    _mlp_bwd_dw[grid](
        x, w_up, w_gate, w_down, rw, grad, parts[0], parts[1], parts[2], order, starts, counts, C, H, k, splits,
        *x.stride(), *w_up.stride(), *w_down.stride(), *grad.stride(),
        _bucket(order.numel()), BC=_bc(C),
    )
    d = parts.sum(1)  # DWd partials are stored [E, C, H] inside the same-sized slots.
    return d[0].to(w_up.dtype), d[1].to(w_gate.dtype), d[2].view(E, C, H).to(w_down.dtype)


def _mm(a, div, w, swk, swn, K, N, out, aux, mode, slots, block_expert, n_slots, BM):
    grid = lambda meta: (slots.numel() // BM, triton.cdiv(N, meta["BN"]))
    _grouped_mm[grid](
        a, w, out, out if aux is None else aux, slots, block_expert, n_slots, w.shape[0], K, N, div,
        *a.stride(), w.stride(0), swk, swn, out.stride(0), 0 if aux is None else aux.stride(0),
        _bucket(n_slots), _tf32(), MODE=mode, BM=BM,
    )


def _wgrad(a, div_a, b, div_b, order, starts, counts, n_experts, dtype):
    M, N = a.shape[1], b.shape[1]
    splits = _splits(a.device, n_experts * triton.cdiv(M, 64) * triton.cdiv(N, 64), order.numel())  # 64x64 output tiles
    out = t.empty((splits, n_experts, M, N), device=a.device, dtype=t.float32)
    grid = lambda meta: (n_experts * splits, triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    _expert_wgrad[grid](
        a, b, out, order, starts, counts, M, N, div_a, div_b, splits,
        *a.stride(), *b.stride(), *out.stride()[:3], _bucket(order.numel()), _tf32(),
    )
    return out.sum(0).to(dtype)


class _MoEFused(t.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_up, w_gate, w_down, indices, weights):
        n_tokens, C = x.shape
        E, H = w_up.shape[:2]
        k = indices.shape[1]
        S = n_tokens * k
        bm = _TILE
        route = _route(indices, E, bm)
        slots, block_expert = route[:2]
        rw = weights.reshape(-1).contiguous()
        out = x.new_empty(S, C)
        with t.cuda.device(x.device):
            _mlp_fwd[(slots.numel() // bm,)](
                x, w_up, w_gate, w_down, rw, out, slots, block_expert, S, E, C, H, k,
                *x.stride(), *w_up.stride(), *w_down.stride(), _bucket(S), BM=bm, BC=_bc(C),
            )
        ctx.save_for_backward(x, w_up, w_gate, w_down, rw, *route)
        ctx.k, ctx.bm = k, bm
        return out if k == 1 else out.view(n_tokens, k, C).sum(1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, w_up, w_gate, w_down, rw, slots, block_expert, order, starts, counts = ctx.saved_tensors
        n_tokens, C = x.shape
        E, H = w_up.shape[:2]
        k, bm = ctx.k, ctx.bm
        S = n_tokens * k
        dx = d_up = d_gate = d_down = d_weights = None
        with t.cuda.device(x.device):
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[5]:
                dx_slots = x.new_empty(S, C)
                drw = t.empty(S, device=x.device, dtype=t.float32)
                _mlp_bwd_dx[(slots.numel() // bm,)](
                    x, w_up, w_gate, w_down, rw, grad, dx_slots, drw, slots, block_expert, S, E, C, H, k,
                    *x.stride(), *w_up.stride(), *w_down.stride(), *grad.stride(),
                    _bucket(S), BM=bm, BC=_bc(C),
                )
                dx = dx_slots if k == 1 else dx_slots.view(n_tokens, k, C).sum(1)
                d_weights = drw.view(n_tokens, k).to(rw.dtype)
            if any(ctx.needs_input_grad[1:4]):
                d_up, d_gate, d_down = _weight_grads(x, w_up, w_gate, w_down, rw, grad, order, starts, counts, k)
        return dx, d_up, d_gate, d_down, None, d_weights


class _MoEGrouped(t.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_ug, w_down, indices):
        n_tokens, C = x.shape
        E, H = w_down.shape[0], w_down.shape[2]
        k = indices.shape[1]
        S = n_tokens * k
        bm = _TILE
        route = _route(indices, E, bm)
        slots, block_expert = route[:2]
        ug, act, y = x.new_empty(S, 2 * H), x.new_empty(S, H), x.new_empty(S, C)
        with t.cuda.device(x.device):
            _mm(x, k, w_ug, w_ug.stride(2), w_ug.stride(1), C, H, act, ug, 1, slots, block_expert, S, bm)
            _mm(act, 1, w_down, w_down.stride(2), w_down.stride(1), H, C, y, None, 0, slots, block_expert, S, bm)
        ctx.save_for_backward(x, w_ug, w_down, ug, act, *route)
        ctx.k, ctx.bm = k, bm
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        x, w_ug, w_down, ug, act, slots, block_expert, order, starts, counts = ctx.saved_tensors
        n_tokens, C = x.shape
        E, H = w_down.shape[0], w_down.shape[2]
        k, bm, S = ctx.k, ctx.bm, grad_y.shape[0]
        dx = dw_ug = dw_down = None
        with t.cuda.device(x.device):
            d_ug = ug.new_empty(S, 2 * H)
            _mm(grad_y, 1, w_down, w_down.stride(1), w_down.stride(2), C, H, d_ug, ug, 2, slots, block_expert, S, bm)
            if ctx.needs_input_grad[0]:
                dx_slot = x.new_empty(S, C)
                _mm(d_ug, 1, w_ug, w_ug.stride(1), w_ug.stride(2), 2 * H, C, dx_slot, None, 0, slots, block_expert, S, bm)
                dx = dx_slot.view(n_tokens, k, C).sum(1)
            if ctx.needs_input_grad[1]:
                dw_ug = _wgrad(d_ug, 1, x, k, order, starts, counts, E, w_ug.dtype)
            if ctx.needs_input_grad[2]:
                dw_down = _wgrad(grad_y, 1, act, 1, order, starts, counts, E, w_down.dtype)
        return dx, dw_ug, dw_down, None


def moe_swiglu(x, w_up, w_gate, w_down, indices, weights):
    """Weighted top-k SwiGLU mixture: [N,C] tokens, [E,H,C] up/gate, [E,C,H] down, [N,k] weights."""
    w_ug = t.cat((w_up, w_gate), dim=1)  # one layout: the kernels read up and gate with shared strides
    # The fused kernel keeps whole token rows on-chip, which only fits narrow models.
    if w_up.dtype == t.float32 or x.shape[1] > 256:
        y = _MoEGrouped.apply(x.to(w_up.dtype), w_ug, w_down, indices)
        return (y.view(*indices.shape, -1) * weights.unsqueeze(-1)).sum(1)
    H = w_up.shape[1]
    return _MoEFused.apply(x, w_ug[:, :H], w_ug[:, H:], w_down, indices, weights)
