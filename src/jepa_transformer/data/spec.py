"""Feature specification: blocks of point-in-time SQL channels and their transforms.

The SQL checks below are a tripwire, not a proof: they reject known leaky
columns and forward-looking window shapes, but any new source SQL still needs
review for point-in-time correctness.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Columns whose presence or value encodes later delisting, continuity decisions
# or collection state (look-ahead), and panel bookkeeping columns. None may
# appear in feature-producing SQL.
FORBIDDEN = re.compile(
    r"\b(source|provider_mismatch|provider_checked|quality_flagged|bar_count\w*|weighted_average_price\w*|vwap\w*"
    r"|trade_size\w*|next_link_valid|continuity_break|continuity_segment|segment_start|valid_to|status"
    r"|last_zero_bar_date|model_start_date|\w+_masked|news_keep_rate|retrieved_at|loaded_at"
    r"|collection_gap|request_hash|\w*raw_id|\w*provider_status|comparative_backfill\w*|parsed_filing\w*"
    r"|fwd_\d+|end_\d+|split|grp|member|era_end|ctx_(?:train|val|test))\b",
    re.IGNORECASE,
)
# Window shapes that read rows after t, or rank a whole partition.
FUTURE = re.compile(r"\b(lead|following|desc|percent_rank|cume_dist|ntile)\b|\blag\s*\([^)]*,\s*-", re.IGNORECASE)
# A window spec in parentheses; without ORDER BY it spans the whole partition,
# unless it extends a named window (whose own definition is checked).
WINDOW = re.compile(r"\b(OVER|AS)\s*\(([^()]*)\)", re.IGNORECASE)
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class Feature:
    """One input channel, or one embedding expanded to ``width`` channels.

    Transforms run in order: ``log`` ("ln" or "log1p"; invalid inputs become
    missing), ``clip`` to fixed bounds, then ``norm``: "train_z" (training-window
    mean/sd), "cs_rank" or "cs_z" (cross-section of non-holdout members on date t),
    or "none". A ``mask`` names a mask channel; where it is 0 this value is missing.
    """

    name: str
    sql: str
    kind: str = "value"
    log: str | None = None
    clip: tuple[float, float] | None = None
    norm: str = "none"
    mask: str | None = None
    dim: int = 0
    dtype: str = "float16"
    pca: int = 0

    def __post_init__(self) -> None:
        assert NAME.match(self.name), f"feature name {self.name!r} must be a lowercase identifier"
        assert self.kind in {"value", "mask", "embedding"}, f"{self.name}: unknown kind {self.kind}"
        assert self.norm in {"none", "train_z", "cs_rank", "cs_z"}, f"{self.name}: unknown norm {self.norm}"
        assert self.log in {None, "ln", "log1p"}, f"{self.name}: log must be ln or log1p"
        assert self.clip is None or self.clip[0] < self.clip[1], f"{self.name}: clip must be [low, high]"
        if self.kind == "embedding":
            assert self.dim > 0 and 0 <= self.pca <= self.dim, f"{self.name}: embedding needs dim and pca <= dim"
            assert self.norm == "none" and self.log is None and self.clip is None, f"{self.name}: embeddings use pca only"
        if self.kind == "mask":
            assert self.norm == "none" and self.mask is None, f"{self.name}: mask channels are raw 0/1"

    @property
    def width(self) -> int:
        return (self.pca or self.dim) if self.kind == "embedding" else 1

    @property
    def channels(self) -> list[str]:
        if self.kind != "embedding":
            return [self.name]
        return [f"{self.name}_{i:02d}" for i in range(self.width)]


@dataclass(frozen=True, slots=True)
class Block:
    """Features sharing one JEPA input group and, optionally, one joined source.

    Source kinds: "asset" joins on (con_id, trading_date); "asof" takes the latest
    row with trading_date on or before t; "date" joins on trading_date.
    """

    name: str
    features: tuple[Feature, ...]
    models: tuple[str, ...] = ("A", "B")
    source_kind: str | None = None
    source_sql: str | None = None
    optional: bool = False
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert NAME.match(self.name), f"block name {self.name!r} must be a lowercase identifier"
        assert self.features, f"block {self.name} has no features"
        assert set(self.models) <= {"A", "B"}, f"block {self.name}: models must be A and/or B"
        assert (self.source_kind is None) == (self.source_sql is None), f"block {self.name}: source needs kind and sql"
        assert self.source_kind in {None, "asset", "asof", "date"}, f"block {self.name}: unknown source kind"


@dataclass(frozen=True, slots=True)
class Splits:
    """Company groups (core/extended/holdout fractions) and time-era boundaries."""

    salt: str
    core_fraction: float
    extended_fraction: float
    liquidity_end: date
    train_end: date
    val_end: date
    test_end: date
    extended_train_end: date
    fit_end: date
    phase: str = "select"  # select: all groups train <= train_end; final: core <= val_end, extended <= extended_train_end

    def __post_init__(self) -> None:
        assert self.phase in ("select", "final"), "phase is select or final"
        assert 0 < self.core_fraction and 0 <= self.extended_fraction and self.core_fraction + self.extended_fraction < 1
        assert self.train_end < self.val_end < self.test_end, "eras must be ordered train < val < test"
        assert self.train_end <= self.extended_train_end < self.test_end, "extended training must end inside the test era"
        assert self.fit_end <= self.train_end and self.liquidity_end <= self.train_end, "fits use the training era only"


@dataclass(frozen=True, slots=True)
class Spec:
    members: str
    context: str
    lookback: int
    splits: Splits
    horizons: tuple[int, ...]
    z_clip: float
    price_stages: tuple[dict[str, str], ...]
    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        assert self.lookback >= 0 and self.z_clip > 0 and self.horizons and min(self.horizons) >= 1
        features = [f for b in self.blocks for f in b.features]
        names = [f.name for f in features] + [b.name for b in self.blocks]
        assert len(set(names)) == len(names), "feature and block names must be unique"
        kinds = {f.name: f.kind for f in features}
        for f in features:
            assert f.mask is None or kinds.get(f.mask) == "mask", f"{f.name}: mask {f.mask} is not a mask feature"
        stage_sql = [sql for stage in self.price_stages for sql in stage.values()]
        for sql in stage_sql + [b.source_sql or "" for b in self.blocks] + [f.sql for f in features]:
            if bad := FORBIDDEN.search(sql) or FUTURE.search(sql) or _unordered_window(sql):
                raise ValueError(f"spec SQL uses {bad.group(0)!r}, which leaks look-ahead information: {sql!r}")

    def blocks_for(self, model: str) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if model in b.models)


def _unordered_window(sql: str) -> re.Match | None:
    for match in WINDOW.finditer(sql):
        body = match.group(2).strip().lower()
        named = match.group(1).lower() == "over" and (body.split() or [""])[0] not in {"", "partition", "rows", "range", "groups"}
        is_window = match.group(1).lower() == "over" or body.startswith(("partition", "order"))
        if is_window and "order by" not in body and not named:
            return match
    return None


def load_spec(path: str | Path) -> Spec:
    """Read a TOML feature specification."""
    with open(path, "rb") as file:
        return spec_from_dict(tomllib.load(file))


def spec_from_dict(raw: dict) -> Spec:
    blocks = []
    for block in raw["block"]:
        source = block.get("source", {})
        features = tuple(
            Feature(**{**f, "clip": tuple(f["clip"]) if "clip" in f else None}) for f in block["feature"]
        )
        blocks.append(
            Block(
                name=block["name"],
                features=features,
                models=tuple(block.get("models", ("A", "B"))),
                source_kind=source.get("kind"),
                source_sql=source.get("sql"),
                optional=block.get("optional", False),
                requires=tuple(block.get("requires", ())),
            )
        )
    universe = raw["universe"]
    return Spec(
        members=universe["members"],
        context=universe["context"],
        lookback=universe.get("lookback_sessions", 252),
        splits=Splits(**raw["splits"]),
        horizons=tuple(raw["labels"]["horizons"]),
        z_clip=raw.get("normalization", {}).get("z_clip", 5.0),
        price_stages=tuple(raw.get("price_stage", ())),
        blocks=tuple(blocks),
    )
