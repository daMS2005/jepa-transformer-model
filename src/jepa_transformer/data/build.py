"""Materialize point-in-time features, labels and splits from a DuckDB replica into array shards.

The replica is attached read-only. Intermediate tables live in a work database
inside the output directory; arrays are exported per split in date-range chunks,
so memory stays bounded by one chunk. Each split directory holds shards of
``con_id``, ``day`` (days since 1970-01-01), ``seg``, ``k`` (row index in the
continuity segment), ``group`` (0 core, 1 extended, 2 holdout), ``target`` (a
member row of this split), ``labels`` [N, horizons] and ``features`` [N, F]
(float32, NaN where unavailable). Non-target rows are lookback context; their
labels are NaN.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np

from jepa_transformer.data.spec import Feature, Spec, load_spec
from jepa_transformer.data.splits import SPLITS, company_splits_sql, labels_sql, literal, panel_sql

logger = logging.getLogger(__name__)
SEGMENT = "WINDOW w AS (PARTITION BY con_id, seg ORDER BY trading_date)"
CS_REFERENCE = "member AND grp IN ('core', 'extended')"


def build_cache(
    spec_path: str | Path,
    database: str | Path,
    output: str | Path,
    model: str,
    *,
    chunk_days: int = 366,
    memory_limit: str = "1500MB",
    threads: int = 2,
    keep_work: bool = False,
    phase: str | None = None,
) -> dict:
    """Build the array cache for ``model`` ("A" or "B") and return its metadata."""
    spec = load_spec(spec_path)
    if phase:
        spec = replace(spec, splits=replace(spec.splits, phase=phase))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work.duckdb"
    work.unlink(missing_ok=True)
    con = duckdb.connect(config={"memory_limit": memory_limit, "threads": threads, "temp_directory": str(output / "tmp")})
    try:
        con.execute(f"ATTACH {literal(Path(database).resolve())} AS src (READ_ONLY)")
        con.execute(f"ATTACH {literal(work)} AS w")
        con.execute("USE w")
        con.execute("SET search_path = 'w.main,src.main'")
        con.execute("SET preserve_insertion_order = false")

        blocks, skipped = _available_blocks(con, spec, model)
        features = [f for b in blocks for f in b.features]
        _prepare(con, spec, blocks)
        stats, pca = _fit(con, spec, features)
        meta = {
            "model": model,
            "phase": spec.splits.phase,
            "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
            "features": [c for f in features for c in f.channels],
            "feature_groups": [[b.name, sum(f.width for f in b.features)] for b in blocks],
            "mask_units": [f.width for f in features],
            "horizons": list(spec.horizons),
            "lookback_sessions": spec.lookback,
            "skipped_blocks": skipped,
            "splits": {},
        }
        for split in SPLITS:
            meta["splits"][split] = _export(con, spec, features, stats, pca, output / split, split, chunk_days)
    finally:
        con.close()
    if not keep_work:
        work.unlink(missing_ok=True)
        shutil.rmtree(output / "tmp", ignore_errors=True)
    (output / "stats.json").write_text(json.dumps(stats, indent=1))
    if pca:
        np.savez(output / "pca.npz", **{f"{k}_{part}": v for k, p in pca.items() for part, v in p.items()})
    shutil.copyfile(spec_path, output / "spec.toml")
    (output / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _available_blocks(con, spec: Spec, model: str):
    tables = {name for (name,) in con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = 'src' "
        "UNION SELECT view_name FROM duckdb_views() WHERE database_name = 'src'"
    ).fetchall()}
    blocks, skipped = [], []
    for block in spec.blocks_for(model):
        missing = [t for t in block.requires if t not in tables]
        if not missing:
            blocks.append(block)
        elif block.optional:
            logger.warning("skipping optional block %s: missing %s", block.name, ", ".join(missing))
            skipped.append(block.name)
        else:
            raise ValueError(f"block {block.name} requires missing tables: {', '.join(missing)}")
    return blocks, skipped


def _prepare(con, spec: Spec, blocks) -> None:
    con.execute(f"""
CREATE TABLE member_days AS
SELECT DISTINCT p.con_id, p.trading_date
FROM prices_daily_model_v1 p JOIN universe_membership u
  ON u.con_id = p.con_id AND u.universe_name = {literal(spec.members)}
 AND p.trading_date BETWEEN u.valid_from AND coalesce(u.valid_to, DATE '9999-12-31')""")
    con.execute(f"CREATE TABLE ctx AS SELECT DISTINCT con_id, symbol FROM universe_membership "
                f"WHERE universe_name = {literal(spec.context)}")
    # Only these price columns are visible to features; flags and provider fields are not.
    stages = [f"""base AS (
  SELECT con_id, trading_date, open, high, low, close, volume, coalesce(split_ratio, 1) AS split_ratio, previous_link_valid,
         sum(CASE WHEN previous_link_valid THEN 0 ELSE 1 END) OVER (PARTITION BY con_id ORDER BY trading_date) AS seg
  FROM prices_daily_model_v1
  WHERE con_id IN (SELECT con_id FROM member_days UNION SELECT con_id FROM ctx))""",
              f"s0 AS (SELECT *, row_number() OVER w AS k FROM base {SEGMENT})"]
    for i, stage in enumerate(spec.price_stages, 1):
        columns = ", ".join(f"{sql} AS {name}" for name, sql in stage.items() if name != "join")
        join = stage.get("join", "").replace("{prev}", f"s{i - 1}")
        stages.append(f"s{i} AS (SELECT *, {columns} FROM s{i - 1} {join} {SEGMENT})")
    con.execute(f"CREATE TABLE px AS WITH {', '.join(stages)} SELECT * FROM s{len(spec.price_stages)} "
                "ORDER BY trading_date, con_id")
    con.execute(company_splits_sql(spec))
    con.execute(panel_sql(spec))
    con.execute(labels_sql(spec))

    joins = []
    for block in blocks:
        if block.source_kind is None:
            continue
        table = f"src_{block.name}"
        keys = "trading_date" if block.source_kind == "date" else "con_id, trading_date"
        con.execute(f"CREATE TABLE {table} AS {block.source_sql}")
        (duplicates,) = con.execute(f"SELECT count(*) - count(DISTINCT ({keys})) FROM {table}").fetchone()
        if duplicates:
            raise ValueError(f"source {block.name} has {duplicates} duplicate ({keys}) rows")
        joins.append(f"{'ASOF ' if block.source_kind == 'asof' else ''}LEFT JOIN {table} USING ({keys})")

    labels = ", ".join(f"CASE WHEN l.end_{h} <= era_end THEN l.fwd_{h} END AS fwd_{h}" for h in spec.horizons)
    by_name = {f.name: f for b in blocks for f in b.features}
    columns = ", ".join(f"{_expression(f, by_name)} AS f_{f.name}" for f in by_name.values())
    con.execute(f"""
CREATE VIEW feat AS
SELECT con_id, trading_date, datediff('day', DATE '1970-01-01', trading_date) AS day, seg, k, member, grp, split,
       {', '.join(f'ctx_{s}' for s in SPLITS)}, {labels}, {columns}
FROM panel JOIN px USING (con_id, trading_date) LEFT JOIN labels l USING (con_id, trading_date) {' '.join(joins)}""")


def _expression(f: Feature, by_name: dict[str, Feature]) -> str:
    """SQL for a feature's value after its log and clip transforms, NULL where its mask is 0.

    Masking here, before fitting and cross-sectional referencing, keeps masked
    rows out of normalization statistics and comparison sets.
    """
    if f.kind == "mask":
        return f"CAST(coalesce(CAST(({f.sql}) AS BOOLEAN), false) AS DOUBLE)"
    x = f"({f.sql})" if f.kind == "embedding" else f"CAST(({f.sql}) AS DOUBLE)"
    if f.mask is not None:
        x = f"CASE WHEN {_expression(by_name[f.mask], by_name)} = 1 THEN {x} END"
    if f.kind == "embedding":
        return x
    if f.log == "ln":
        x = f"CASE WHEN {x} > 0 THEN ln({x}) END"
    elif f.log == "log1p":
        x = f"CASE WHEN {x} > -1 THEN ln(1 + {x}) END"
    if f.clip is not None:
        # greatest/least ignore NULLs in DuckDB, so missing values must stay missing explicitly.
        x = f"CASE WHEN {x} IS NOT NULL THEN least(greatest({x}, {f.clip[0]}), {f.clip[1]}) END"
    return x


def _fit(con, spec: Spec, features: list[Feature]):
    """Normalization and PCA parameters from training-split member rows up to ``fit_end``."""
    fit = f"member AND split = 'train' AND trading_date <= DATE {literal(spec.splits.fit_end)}"
    zs = [f for f in features if f.norm == "train_z"]
    stats = {}
    if zs:
        row = con.execute(
            "SELECT " + ", ".join(f"avg(f_{f.name}), stddev_pop(f_{f.name})" for f in zs) + f" FROM feat WHERE {fit}"
        ).fetchone()
        for i, f in enumerate(zs):
            mean, std = row[2 * i], row[2 * i + 1]
            if mean is None:
                raise ValueError(f"{f.name}: no training rows to fit its normalization")
            stats[f.name] = {"mean": mean, "std": std if std else 1.0}
    pca = {}
    for f in (f for f in features if f.kind == "embedding" and f.pca):
        # Effective-dated document embeddings repeat on every daily row. Fit
        # each training-era vector once instead of weighting it by how long the
        # filing remained current (and multiplying PCA cost by millions).
        cursor = con.execute(
            f"SELECT DISTINCT f_{f.name} FROM feat WHERE {fit} AND f_{f.name} IS NOT NULL ORDER BY 1"
        )
        n, total, gram = 0, np.zeros(f.dim), np.zeros((f.dim, f.dim))
        while batch := cursor.fetchmany(20_000):
            x = _decode([b for (b,) in batch], f).astype(np.float64)
            n, total, gram = n + len(x), total + x.sum(0), gram + x.T @ x
        if n <= f.pca:
            raise ValueError(f"{f.name}: {n} training rows cannot fit {f.pca} components")
        mean = total / n
        values, vectors = np.linalg.eigh(gram / n - np.outer(mean, mean))
        top = np.argsort(values)[::-1][: f.pca]
        components = vectors[:, top]
        components *= np.sign(components[np.abs(components).argmax(0), np.arange(f.pca)])
        pca[f.name] = {"mean": mean, "components": components, "scale": np.sqrt(np.maximum(values[top], 1e-12))}
    return stats, pca


def _export(con, spec, features, stats, pca, directory: Path, split: str, chunk_days: int) -> list[str]:
    """Write one split in date chunks; return the shard names in date order."""
    first, last = con.execute(f"SELECT min(trading_date), max(trading_date) FROM panel WHERE ctx_{split}").fetchone()
    cs = [f for f in features if f.norm in ("cs_rank", "cs_z")]
    keys = (
        "con_id, day, CAST(seg AS INTEGER) AS seg, CAST(k AS INTEGER) AS k, "
        "CASE grp WHEN 'core' THEN 0 WHEN 'extended' THEN 1 ELSE 2 END AS grp, "
        f"coalesce(split = '{split}', false) AS target"
    )
    shutil.rmtree(directory, ignore_errors=True)
    shards = []
    start = first
    while first is not None and start <= last:
        end = min(start + timedelta(days=chunk_days - 1), last)
        dates = f"trading_date BETWEEN DATE '{start}' AND DATE '{end}'"
        data = con.execute(
            f"SELECT {keys}, {', '.join(f'fwd_{h}' for h in spec.horizons)}, "
            f"{', '.join(f'f_{f.name}' for f in features)} FROM feat WHERE ctx_{split} AND {dates} ORDER BY con_id, trading_date"
        ).fetchnumpy()
        reference = con.execute(
            f"SELECT day, {', '.join(f'f_{f.name}' for f in cs)} FROM feat WHERE {CS_REFERENCE} AND {dates}"
        ).fetchnumpy() if cs else {}
        start = end + timedelta(days=1)
        if not len(data["con_id"]):
            continue
        target = np.asarray(data["target"], dtype=bool)
        labels = np.stack([_float(data[f"fwd_{h}"]) for h in spec.horizons], axis=1).astype(np.float32)
        labels[~target] = np.nan
        arrays = {
            "con_id": np.asarray(data["con_id"], dtype=np.int64),
            "day": np.asarray(data["day"], dtype=np.int32),
            "seg": np.asarray(data["seg"], dtype=np.int32),
            "k": np.asarray(data["k"], dtype=np.int32),
            "group": np.asarray(data["grp"], dtype=np.int8),
            "target": target,
            "labels": labels,
            "features": _transform(data, reference, features, stats, pca, spec.z_clip),
        }
        shard = directory / f"{len(shards):04d}"
        shard.mkdir(parents=True, exist_ok=True)
        for key, value in arrays.items():
            np.save(shard / f"{key}.npy", value)
        shards.append(shard.name)
    return shards


def _float(values) -> np.ndarray:
    out = np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


def _decode(blobs, f: Feature) -> np.ndarray:
    x = np.frombuffer(b"".join(bytes(b) for b in blobs), dtype=np.dtype(f.dtype).newbyteorder("<"))
    return x.reshape(len(blobs), f.dim)


def _transform(data, reference, features, stats, pca, z_clip) -> np.ndarray:
    columns = {}
    for f in features:
        raw = data[f"f_{f.name}"]
        if f.kind == "embedding":
            values = np.ma.asarray(raw)
            present = ~np.ma.getmaskarray(values) & np.array([v is not None for v in values.data])
            out = np.full((len(present), f.width), np.nan)
            if present.any():
                x = _decode(values.data[present], f).astype(np.float64)
                if f.pca:
                    p = pca[f.name]
                    x = (x - p["mean"]) @ p["components"] / p["scale"]
                out[present] = x
            columns[f.name] = out
            continue
        x = _float(raw)
        if f.norm == "train_z":
            x = np.clip((x - stats[f.name]["mean"]) / stats[f.name]["std"], -z_clip, z_clip)
        elif f.norm in ("cs_rank", "cs_z"):
            x = _cross_section(x, data["day"], _float(reference[f"f_{f.name}"]), reference["day"], f.norm, z_clip)
        columns[f.name] = x[:, None]
    return np.concatenate([columns[f.name] for f in features], axis=1).astype(np.float32)


def _cross_section(x, day, ref, ref_day, norm, z_clip) -> np.ndarray:
    """Rank in [-1, 1] or z-score of each value against the reference rows of its own date."""
    out = np.full_like(x, np.nan)
    keep = np.isfinite(ref)
    ref, ref_day = ref[keep], np.asarray(ref_day)[keep]
    order = np.lexsort((ref, ref_day))
    ref, ref_day = ref[order], ref_day[order]
    rows_by_day = np.flatnonzero(np.isfinite(x))
    rows_by_day = rows_by_day[np.argsort(np.asarray(day)[rows_by_day], kind="stable")]
    days = np.asarray(day)[rows_by_day]
    for d in np.unique(days):
        rows = rows_by_day[np.searchsorted(days, d, "left"):np.searchsorted(days, d, "right")]
        r = ref[np.searchsorted(ref_day, d, "left"):np.searchsorted(ref_day, d, "right")]
        if not len(r):
            continue
        v = x[rows]
        if norm == "cs_rank":
            out[rows] = (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / len(r) - 1.0
        elif r.std() > 0:
            out[rows] = np.clip((v - r.mean()) / r.std(), -z_clip, z_clip)
    return out
