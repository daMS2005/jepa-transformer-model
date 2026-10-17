"""Datasets over an exported cache, in the shapes the JEPA and forecasting engines consume.

``JEPADataset`` yields one asset-day feature vector ``[F]``; a ``DataLoader``
stacks them into the ``[B, F]`` batches ``train_jepa_epoch`` takes. Unavailable
features are NaN, which JEPA treats as unobserved (or pass ``fill_value``).

``SequenceDataset`` yields ``(input [T, D], true [T, H], valid_mask [T])`` windows
of one asset's continuity segment in date order, stacked by the default collate
into the ``(input, true, valid_mask)`` batches of ``train_transformer_epoch``.
``input`` rows are raw features unless ``inputs`` (e.g. from :func:`encode_rows`)
supplies per-row JEPA representations; the forecaster needs finite inputs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

GROUP_CODES = {"core": 0, "extended": 1, "holdout": 2}


class CacheSplit:
    """Keys and labels of one split in memory; features memory-mapped across shards."""

    def __init__(self, root: str | Path, split: str) -> None:
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        shards = [root / split / name for name in self.meta["splits"][split]]
        assert shards, f"split {split} has no rows"

        def load(name: str) -> np.ndarray:
            return np.concatenate([np.load(shard / f"{name}.npy") for shard in shards])

        self.con_id, self.day, self.seg, self.k = load("con_id"), load("day"), load("seg"), load("k")
        self.group, self.target, self.labels = load("group"), load("target"), load("labels")
        self._features = [np.load(shard / "features.npy", mmap_mode="r") for shard in shards]
        self._offsets = np.cumsum([0] + [len(f) for f in self._features])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def features(self, rows: np.ndarray) -> np.ndarray:
        """Gather feature rows by position in this split's storage order."""
        rows = np.asarray(rows)
        out = np.empty((len(rows), self._features[0].shape[1]), dtype=np.float32)
        shard = np.searchsorted(self._offsets, rows, side="right") - 1
        for s in np.unique(shard):
            hit = shard == s
            out[hit] = self._features[s][rows[hit] - self._offsets[s]]
        return out

    def targets(self, groups: Sequence[str] | None) -> np.ndarray:
        keep = self.target.copy()
        if groups is not None:
            keep &= np.isin(self.group, [GROUP_CODES[g] for g in groups])
        return keep


class JEPADataset(Dataset):
    """Independent member asset-days of one split."""

    def __init__(self, root, split: str, *, groups: Sequence[str] | None = None, fill_value: float | None = None):
        self.rows = CacheSplit(root, split)
        self.index = np.flatnonzero(self.rows.targets(groups))
        self.fill_value = fill_value

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Tensor:
        x = torch.from_numpy(self.rows.features(self.index[i : i + 1])[0])
        return x if self.fill_value is None else x.nan_to_num(self.fill_value)


class SequenceDataset(Dataset):
    """Fixed-length chronological windows that never cross an asset, segment, or row gap.

    Windows end on target rows and step back by ``stride``; each supervises the
    targets in its last ``stride`` positions, so every target is supervised once.
    Earlier positions (including non-target lookback rows) are context only.
    Segments shorter than ``sequence_length`` cannot form a window and are counted
    in ``skipped_targets``. ``valid_mask`` also requires every chosen horizon.
    """

    def __init__(
        self,
        root,
        split: str,
        sequence_length: int,
        *,
        horizons: Sequence[int] | None = None,
        stride: int | None = None,
        groups: Sequence[str] | None = None,
        inputs: np.ndarray | None = None,
    ) -> None:
        self.rows = rows = CacheSplit(root, split)
        self.length = sequence_length
        self.stride = stride or sequence_length
        assert 0 < self.stride <= sequence_length, "stride must be in [1, sequence_length]"
        all_horizons = rows.meta["horizons"]
        self.columns = [all_horizons.index(h) for h in (horizons or all_horizons)]
        assert inputs is None or len(inputs) == len(rows), "inputs must align with the split's rows"
        self.inputs = inputs

        self.order = np.lexsort((rows.day, rows.con_id))
        con_id, seg, k = rows.con_id[self.order], rows.seg[self.order], rows.k[self.order]
        breaks = np.flatnonzero((np.diff(con_id) != 0) | (np.diff(seg) != 0) | (np.diff(k) != 1)) + 1
        target = rows.targets(groups)[self.order]
        self.supervised = target
        starts, ends, self.skipped_targets = [], [], 0
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(self.order)], strict=True):
            positions = a + np.flatnonzero(target[a:b])
            if not len(positions):
                continue
            if b - a < sequence_length:
                self.skipped_targets += len(positions)
                continue
            end = positions[-1]
            while end >= positions[0]:
                starts.append(max(a, end - sequence_length + 1))
                ends.append(end)
                earlier = positions[positions <= end - self.stride]
                if not len(earlier):
                    break
                end = earlier[-1]
        self.starts, self.ends = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        start, end = self.starts[i], self.ends[i]
        positions = np.arange(start, start + self.length)
        rows = self.order[positions]
        x = self.rows.features(rows) if self.inputs is None else np.asarray(self.inputs[rows], dtype=np.float32)
        y = self.rows.labels[rows][:, self.columns]
        valid = self.supervised[positions] & (positions > end - self.stride) & (positions <= end)
        valid &= np.isfinite(y).all(axis=1)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valid)


@torch.no_grad()
def encode_rows(root, split: str, encode: Callable[[Tensor], Tensor], *, batch_size: int = 4096) -> np.ndarray:
    """Apply ``encode`` (e.g. ``jepa.eval().encode``) to every row of a split, in storage order."""
    rows = CacheSplit(root, split)
    chunks = [
        encode(torch.from_numpy(rows.features(np.arange(i, min(i + batch_size, len(rows)))))).float().cpu().numpy()
        for i in range(0, len(rows), batch_size)
    ]
    return np.concatenate(chunks)
