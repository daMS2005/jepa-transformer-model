"""Next-day conditional distributions built on Transformer representations."""

import torch as t
from torch import nn
from torch.distributions import StudentT
from torch.nn import functional as F


class StudentTHead(nn.Module):
    """Independent Student-t targets with positive scale and finite variance."""

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.d, 3 * config.target_dim)

    def forward(self, input):
        params = self.linear(input)
        # Distribution arithmetic is not evaluated in FP16/BF16.
        if params.dtype in (t.float16, t.bfloat16):
            params = params.float()
        mu, scale, df = params.chunk(3, dim=-1)
        scale = F.softplus(scale) + 1e-4
        df = 2.0 + F.softplus(df) + 1e-4
        return StudentT(df=df, loc=mu, scale=scale)
