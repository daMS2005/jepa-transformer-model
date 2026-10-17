"""Dense SwiGLU and dropless, token-wise expert routing."""

import torch as t
from torch import nn
from torch.nn import functional as F

from jepa_transformer.kernels.transformer import Linear as TritonLinear, moe_swiglu


class SwiGlu(nn.Module):
    def __init__(self, config):
        super().__init__()
        Linear = TritonLinear if config.kernel_backend == "triton" else nn.Linear
        self.up_proj = Linear(config.d, config.hidden_dim, bias=False)
        self.gate_proj = Linear(config.d, config.hidden_dim, bias=False)
        self.actf = nn.SiLU()
        self.down_proj = Linear(config.hidden_dim, config.d, bias=False)

    def forward(self, input):
        return self.down_proj(self.actf(self.gate_proj(input)) * self.up_proj(input))


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = SwiGlu(config)

    def forward(self, input):
        return self.net(input)


class MoE(nn.Module):
    """Top-k SwiGLU experts and an unweighted load-balancing loss.

    Gates are probabilities over all experts, without renormalizing the selected
    subset. A top-1 gate therefore retains a forecasting-loss gradient. There is
    no batch-dependent capacity limit, token dropping, or cross-token routing.
    """

    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.k = config.moe_topk
        self.n_experts = config.n_experts
        self.kernel_backend = config.kernel_backend
        self.router = nn.Linear(config.d, config.n_experts, bias=False)

    def forward(self, input):
        B, T, C = input.shape
        input = input.reshape(-1, C)
        routed = self.router(input)
        dtype = t.float64 if routed.dtype == t.float64 else t.float32
        probs = t.softmax(routed.to(dtype), dim=-1)
        topk_weights, topk_indices = t.topk(probs, self.k, dim=-1)
        if self.kernel_backend == "triton":
            nets = [expert.net for expert in self.experts]
            output = moe_swiglu(
                input, t.stack([net.up_proj.weight for net in nets]),
                t.stack([net.gate_proj.weight for net in nets]),
                t.stack([net.down_proj.weight for net in nets]), topk_indices, topk_weights,
            )
        else:
            output = t.zeros_like(input)
            for idx, expert in enumerate(self.experts):
                token_indices, k_indices = (topk_indices == idx).nonzero(as_tuple=True)
                if token_indices.numel() == 0:
                    continue
                expert_input = input[token_indices]
                expert_output = expert(expert_input)
                weights = topk_weights[token_indices, k_indices].unsqueeze(-1)
                output.index_add_(0, token_indices, (expert_output * weights).to(output.dtype))

        load = F.one_hot(topk_indices, self.n_experts).float().mean(dim=(0, 1))
        aux_loss = self.n_experts * (load.detach() * probs.mean(dim=0)).sum()
        return output.view(B, T, C), aux_loss
