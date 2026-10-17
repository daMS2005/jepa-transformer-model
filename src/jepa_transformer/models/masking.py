"""Feature masking for independent-day JEPA representation learning."""

from __future__ import annotations

import torch
from torch import Tensor


def observed_features(
    features: Tensor,
    availability_mask: Tensor | None = None,
) -> Tensor:
    """Return a boolean mask for finite, point-in-time available features."""
    finite = torch.isfinite(features)
    if availability_mask is None:
        return finite
    return availability_mask & finite


def sample_feature_mask(
    availability_mask: Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
    units: Tensor | None = None,
) -> Tensor:
    """Mask available features independently for each daily sample.

    Rows with at least two observed features retain at least one context feature
    and mask at least one target feature. Rows with fewer than two observed
    features are left unmasked and should be excluded from the JEPA loss.
    ``units`` maps each feature to a masking unit; a unit's features (e.g. one
    embedding's components) are masked together, and the counts apply to units.
    """
    if units is not None:
        members = torch.nn.functional.one_hot(units, int(units.max()) + 1).to(availability_mask.device, torch.float32)
        unit_available = (availability_mask.float() @ members) > 0
        return sample_feature_mask(unit_available, mask_ratio, generator=generator)[..., units] & availability_mask
    available = availability_mask.sum(-1, keepdim=True)
    count = (available * mask_ratio).round().clamp(min=1).minimum(available - 1)
    # Rank available features in random order; unavailable ones rank last and are never masked.
    scores = torch.rand(availability_mask.shape, device=availability_mask.device, generator=generator)
    rank = scores.masked_fill(~availability_mask, 2.0).argsort(-1).argsort(-1)
    return rank < count


def apply_feature_mask(
    features: Tensor,
    feature_mask: Tensor,
    availability_mask: Tensor,
) -> Tensor:
    """Replace masked or unavailable features with zeros, so they contribute nothing downstream."""
    return features.masked_fill(~(availability_mask & ~feature_mask), 0.0)
