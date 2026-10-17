"""Decoder-only causal forecasting over independently pretrained JEPA vectors."""

from typing import NamedTuple

import torch as t
from torch import nn
from torch.nn import functional as F
from torch.distributions import StudentT

from jepa_transformer.config import TransformerConfig
from jepa_transformer.kernels.transformer import Linear as TritonLinear
from jepa_transformer.models.embeddings import RoPE
from jepa_transformer.models.feedforward import MoE, SwiGlu
from jepa_transformer.models.forecasting import StudentTHead


class MHA(nn.Module):
    """Full causal MHA with per-head QK-RMSNorm before RoPE."""

    def __init__(self, config):
        super().__init__()
        self.d = config.d
        self.o = config.out_proj
        self.H = config.n_heads
        d = self.d
        o = self.o
        dH = self.o // self.H
        Linear = TritonLinear if config.kernel_backend == "triton" else nn.Linear
        self.q = Linear(d, o, bias=False)
        self.k = Linear(d, o, bias=False)
        self.v = Linear(d, o, bias=False)
        self.linear_out = Linear(self.o, self.d, bias=False)
        self.q_norm = nn.RMSNorm(dH, eps=config.norm_eps)
        self.k_norm = nn.RMSNorm(dH, eps=config.norm_eps)
        self.pos = RoPE(dH, config.rope_base)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input, cache=None, *, use_cache=False):
        H, o = self.H, self.o
        B, T, _ = input.shape
        q = self.q(input).view(B, T, H, o // H).transpose(1, 2)
        k = self.k(input).view(B, T, H, o // H).transpose(1, 2)
        v = self.v(input).view(B, T, H, o // H).transpose(1, 2)
        past_len = 0
        if cache is not None:
            past_len = cache["k"].shape[2]
            use_cache = True

        q = self.pos(q, past_len, norm=self.q_norm)
        k = self.pos(k, past_len, norm=self.k_norm)
        if cache is not None:
            k = t.cat([cache["k"], k], dim=2)
            v = t.cat([cache["v"], v], dim=2)

        # FlashAttention on sm80+ for FP16/BF16 (FP32 softmax inside, scores never stored). Cached chunks
        # attend to every past key plus their own causal prefix: query i sees keys <= past_len + i.
        mask = None if past_len == 0 or T == 1 else t.ones(T, past_len + T, dtype=t.bool, device=q.device).tril(past_len)
        attn_out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask,
                                                  dropout_p=self.dropout.p if self.training else 0.0,
                                                  is_causal=past_len == 0 and T > 1)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, o)
        new_cache = {"k": k.detach(), "v": v.detach()} if use_cache else None
        return self.linear_out(attn_out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        self.attention = MHA(config)
        self.ln1 = nn.RMSNorm(config.d, eps=config.norm_eps)
        self.ln2 = nn.RMSNorm(config.d, eps=config.norm_eps)
        self.ffw = MoE(config) if idx in config.moe_layers else SwiGlu(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input, cache=None, *, use_cache=False):
        hidden_state = self.ln1(input)
        hidden_state, cache = self.attention(hidden_state, cache, use_cache=use_cache)
        res = input + self.dropout(hidden_state)
        hidden_state = self.ln2(res)
        aux_loss = input.new_zeros(())
        if isinstance(self.ffw, MoE):
            hidden_state, aux_loss = self.ffw(hidden_state)
        else:
            hidden_state = self.ffw(hidden_state)
        return res + self.dropout(hidden_state), cache, aux_loss


class TransformerOutput(NamedTuple):
    pred: t.Tensor | StudentT
    cache: list[dict[str, t.Tensor]] | None
    aux_loss: t.Tensor


class Transformer(nn.Module):
    """Forecast next-day targets from dense, chronological [B,T,input_dim] inputs.

    No tokenizer, learned positional embedding, cross-attention, or JEPA model is
    included. Inputs must be unpadded windows from one asset/continuity segment.
    Prediction at position t is supervised by that day's next-day target.
    The decoder has no EMA teacher or masked-representation prediction objective.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        Linear = TritonLinear if config.kernel_backend == "triton" else nn.Linear
        self.embedder = Linear(config.input_dim, config.d, bias=False)
        self.blocks = nn.ModuleList([TransformerBlock(config, idx) for idx in range(config.n_layers)])
        self.ln = nn.RMSNorm(config.d, eps=config.norm_eps)
        self.linear = (StudentTHead(config) if config.head == "student_t"
                       else nn.Linear(config.d, config.target_dim))

    def forward(self, input, cache=None, *, use_cache=False):
        if cache is not None:
            use_cache = True
        else:
            cache = [None] * self.config.n_layers
        x = self.embedder(input).to(input.dtype)  # keep the residual stream in the input dtype under autocast
        new_cache = [] if use_cache else None
        aux_loss = x.new_zeros(())
        for block, b_cache in zip(self.blocks, cache, strict=True):
            x, b_cache, block_loss = block(x, b_cache, use_cache=use_cache)
            aux_loss = aux_loss + block_loss
            if use_cache:
                new_cache.append(b_cache)
        if self.config.moe_layers:
            aux_loss = aux_loss / len(self.config.moe_layers)
        return TransformerOutput(self.linear(self.ln(x)), new_cache, aux_loss)
