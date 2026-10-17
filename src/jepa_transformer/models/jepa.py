"""JEPA representation learning for independent daily feature vectors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from jepa_transformer.models.masking import (
    apply_feature_mask,
    observed_features,
    sample_feature_mask,
)


class Encoder(nn.Module):
    """Encode one day's available features into a fixed-size representation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        *,
        feature_groups: tuple[tuple[str, int], ...] = (),
        input_encoder_dim: int | None = None,
        input_encoder_depth: int = 1,
        residual: bool = False,
    ) -> None:
        super().__init__()
        # Group stems are opt-in. Group boundaries may still be supplied solely
        # for block masking without silently changing the signed-off MLP width.
        self.input_dims = tuple(size for _, size in feature_groups) if input_encoder_dim is not None else ()
        model_dim = input_encoder_dim
        self.input_encoders = nn.ModuleList()
        for input_size in self.input_dims:
            input_layers: list[nn.Module] = []
            for layer_index in range(input_encoder_depth):
                input_layers.extend(
                    [
                        nn.Linear(input_size if layer_index == 0 else model_dim, model_dim),
                        nn.SiLU(),
                        nn.LayerNorm(model_dim),
                        nn.Dropout(dropout),
                    ]
                )
            self.input_encoders.append(nn.Sequential(*input_layers))

        layers: list[nn.Module] = []
        current_in = len(self.input_dims) * model_dim if self.input_dims else input_dim
        for layer_index in range(depth):
            block = nn.Sequential(
                nn.Linear(current_in, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            if residual and layer_index:
                layers.append(_Residual(block))
            else:
                layers.extend(block)
            current_in = hidden_dim
        layers.extend(
            [
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, input: Tensor, availability_mask: Tensor, feature_mask: Tensor) -> Tensor:
        hidden_state = apply_feature_mask(input, feature_mask, availability_mask)
        if self.input_dims:
            groups = hidden_state.split(self.input_dims, dim=-1)
            hidden_state = torch.cat([encoder(group) for encoder, group in zip(self.input_encoders, groups, strict=True)], dim=-1)
        return self.net(hidden_state)


class Predictor(nn.Module):
    """Predict a full-view latent representation from a masked-view latent."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class _Residual(nn.Module):
    """Residual wrapper for same-width JEPA hidden blocks."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, input: Tensor) -> Tensor:
        return input + self.block(input)


@dataclass(frozen=True)
class JEPAOutput:
    """Outputs needed to calculate and inspect the JEPA training objective."""

    pred: Tensor
    target: Tensor
    hidden_state: Tensor
    feature_mask: Tensor
    availability_mask: Tensor
    valid_samples: Tensor


class JEPA(nn.Module):
    """Learn daily feature representations using a masked EMA teacher target.

    The context encoder receives a partially masked daily feature vector. The
    target encoder receives the complete point-in-time available vector and is
    updated only through exponential moving averages of context parameters.
    After pretraining, :meth:`encode` produces the daily representations used by
    the separate temporal Transformer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        predictor_hidden_dim: int | None = None,
        dropout: float,
        feature_groups: tuple[tuple[str, int], ...] = (),
        input_encoder_dim: int | None = None,
        input_encoder_depth: int = 1,
        residual: bool = False,
        mask_units: tuple[int, ...] = (),
        mask_ratio: float = 0.50,
        target_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_ratio = mask_ratio
        self.target_momentum = target_momentum
        # Feature widths of the masking units (one per feature; an embedding's components form one unit).
        assert not mask_units or sum(mask_units) == input_dim, "mask units must sum to input_dim"
        units = torch.repeat_interleave(torch.arange(len(mask_units)), torch.tensor(mask_units, dtype=torch.long))
        self.register_buffer("mask_units", units if mask_units else None, persistent=False)

        self.encoder = Encoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            feature_groups=feature_groups,
            input_encoder_dim=input_encoder_dim,
            input_encoder_depth=input_encoder_depth,
            residual=residual,
        )
        self.target_encoder = deepcopy(self.encoder)
        self.predictor = Predictor(
            output_dim=output_dim,
            hidden_dim=predictor_hidden_dim or hidden_dim,
            dropout=dropout,
        )
        self._freeze_target_encoder()

    def _freeze_target_encoder(self) -> None:
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> JEPA:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(
        self,
        input: Tensor,
        *,
        availability_mask: Tensor | None = None,
        feature_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> JEPAOutput:
        availability = observed_features(input, availability_mask)

        if feature_mask is None:
            feature_mask = sample_feature_mask(availability, self.mask_ratio, generator=generator, units=self.mask_units)

        hidden_state = self.encoder(
            input,
            availability_mask=availability,
            feature_mask=feature_mask,
        )
        pred = self.predictor(hidden_state)

        full_view_mask = torch.zeros_like(availability)
        with torch.no_grad():
            target = self.target_encoder(
                input,
                availability_mask=availability,
                feature_mask=full_view_mask,
            )

        visible_mask = availability & ~feature_mask
        valid_samples = feature_mask.any(dim=-1) & visible_mask.any(dim=-1)
        return JEPAOutput(
            pred=pred,
            target=target,
            hidden_state=hidden_state,
            feature_mask=feature_mask,
            availability_mask=availability,
            valid_samples=valid_samples,
        )

    def encode(
        self,
        input: Tensor,
        *,
        availability_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode complete daily feature vectors for the downstream Transformer."""
        availability = observed_features(input, availability_mask)

        full_view_mask = torch.zeros_like(availability)
        return self.encoder(
            input,
            availability_mask=availability,
            feature_mask=full_view_mask,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """Update target parameters after an optimizer step on the context path."""
        momentum = self.target_momentum if momentum is None else momentum

        for context_parameter, target_parameter in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            target_parameter.mul_(momentum).add_(
                context_parameter,
                alpha=1.0 - momentum,
            )

        
