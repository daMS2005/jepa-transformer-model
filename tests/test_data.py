"""Data layer tests on a small synthetic DuckDB replica (no real data)."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import tomllib
import unittest
from datetime import date, timedelta
from importlib.util import find_spec
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1] / "configs" / "data" / "example.toml"
CUTOFF = "2023-12-31"  # equals splits.fit_end in the shipped spec
VAL_END = "2024-12-31"  # equals splits.val_end
HAS_EXTRAS = bool(find_spec("duckdb") and find_spec("numpy") and find_spec("torch"))

if HAS_EXTRAS:
    import duckdb
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from jepa_transformer.config import JEPAConfig, ModelConfig, RuntimeConfig, TransformerConfig
    from jepa_transformer.data.build import build_cache
    from jepa_transformer.data.dataset import CacheSplit, JEPADataset, SequenceDataset, encode_rows
    from jepa_transformer.data.spec import load_spec, spec_from_dict
    from jepa_transformer.data.splits import company_splits_sql
    from jepa_transformer.models import JEPA, Transformer
    from jepa_transformer.training.run import evaluate, train_jepa, train_transformer
    from jepa_transformer.training.transformer import forecast_ic


def make_replica(path: Path) -> None:
    """Twelve stocks, fourteen context ETFs, fundamentals, macro, and text-description embeddings."""
    con = duckdb.connect(str(path))
    u = lambda salt: f"((hash(con_id, trading_date, {salt}) % 1000003) / 1000003.0)"  # noqa: E731
    con.execute("""
CREATE TABLE days AS SELECT CAST(d AS DATE) AS trading_date FROM range(DATE '2021-01-04', DATE '2025-07-01', INTERVAL 1 DAY) t(d)
WHERE dayofweek(d) BETWEEN 1 AND 5;
CREATE TABLE ids AS SELECT * FROM (VALUES
  (1, 'AAA', 1001), (2, 'AAB', 1001), (3, 'BBB', 1003), (4, 'CCC', 1004), (5, 'DDD', 1005), (6, 'EEE', 1006),
  (7, 'FFF', 1007), (8, 'GGG', 1008), (9, 'HHH', 1009), (10, 'III', 1010), (11, 'JJJ', 1011), (12, 'KKK', NULL),
  (101, 'SPY', NULL), (102, 'IWM', NULL), (103, 'QQQ', NULL),
  (104, 'XLB', NULL), (105, 'XLC', NULL), (106, 'XLE', NULL), (107, 'XLF', NULL),
  (108, 'XLI', NULL), (109, 'XLK', NULL), (110, 'XLP', NULL), (111, 'XLRE', NULL),
  (112, 'XLU', NULL), (113, 'XLV', NULL), (114, 'XLY', NULL)) t(con_id, symbol, cik);""")
    con.execute(f"""
CREATE TABLE raw AS
SELECT con_id, trading_date, {u(1)} AS u1, {u(2)} AS u2, {u(3)} AS u3, {u(4)} AS u4, {u(5)} AS u5,
       CASE WHEN con_id = 4 AND trading_date = DATE '2023-09-01' THEN 2.0 ELSE 1.0 END::DOUBLE AS split_ratio
FROM ids, days WHERE NOT (con_id = 9 AND trading_date > DATE '2023-08-31');
CREATE TABLE prices_daily_model_v1 AS
WITH c AS (
  SELECT *, (10 + con_id % 50) * exp(sum(0.03 * (u1 - 0.5)) OVER (PARTITION BY con_id ORDER BY trading_date))
            / exp(sum(ln(split_ratio)) OVER (PARTITION BY con_id ORDER BY trading_date)) AS close
  FROM raw
)
SELECT con_id, trading_date, close * exp(0.01 * (u2 - 0.5)) AS open,
       greatest(close, close * exp(0.01 * (u2 - 0.5))) * (1 + 0.02 * u3) AS high,
       least(close, close * exp(0.01 * (u2 - 0.5))) * (1 - 0.02 * u4) AS low,
       close, 1e6 * (1 + con_id) * (0.5 + u5) * split_ratio AS volume, split_ratio,
       NOT (row_number() OVER (PARTITION BY con_id ORDER BY trading_date) = 1
            OR (con_id = 3 AND trading_date = DATE '2023-06-15')) AS previous_link_valid,
       con_id = 3 AND trading_date = DATE '2023-06-15' AS continuity_break,
       TRUE AS trade_valid, CASE WHEN con_id % 2 = 0 THEN 'provider_a' ELSE 'provider_b' END AS source,
       CAST(100 * u5 AS BIGINT) AS bar_count, close AS weighted_average_price,
       FALSE AS provider_mismatch, FALSE AS quality_flagged
FROM c;
CREATE TABLE universe_membership AS
SELECT 'pit_top1000_v1' AS universe_name, con_id, symbol,
       CASE con_id WHEN 7 THEN DATE '2022-06-01' WHEN 8 THEN DATE '2023-03-01' ELSE DATE '2021-03-01' END AS valid_from,
       CASE con_id WHEN 7 THEN DATE '2024-06-28' WHEN 9 THEN DATE '2023-08-31' END AS valid_to,
       'test' AS source
FROM ids WHERE con_id < 100
UNION ALL SELECT 'context_etf_v1', con_id, symbol, DATE '2000-01-03', NULL, 'test' FROM ids WHERE con_id > 100;
CREATE TABLE sec_security_mappings AS SELECT con_id, cik, DATE '2010-01-04' AS valid_from FROM ids WHERE cik IS NOT NULL;
CREATE TABLE sec_companies AS SELECT DISTINCT cik, '3571' AS sic FROM ids WHERE cik IS NOT NULL;
CREATE TABLE securities AS SELECT con_id, 'Technology' AS industry FROM ids;
CREATE TABLE sec_daily_fundamentals_model_v1 AS
SELECT con_id, trading_date, i.cik, 'USD' AS currency, 20 + 0.1 * con_id + 0.01 * {u(6)} AS log_assets,
       0.05 + 0.1 * {u(7)} AS revenue_q_to_assets, 0.02 * ({u(8)} - 0.3) AS net_income_q_to_assets,
       0.03 * ({u(9)} - 0.2) AS ocf_q_to_assets, 0.01 * {u(10)} AS capex_q_to_assets, 0.1 * {u(11)} AS cash_to_assets,
       0.3 * {u(12)} AS debt_to_assets, 0.4 + 0.1 * {u(13)} AS equity_to_assets,
       1 + dayofyear(trading_date) % 90 AS assets_age_days,
       30 + dayofyear(trading_date) % 90 AS assets_period_age_days, 30 + dayofyear(trading_date) % 250 AS revenue_period_age_days,
       40 AS net_income_period_age_days, 40 AS ocf_period_age_days, 40 AS capex_period_age_days,
       40 AS cash_period_age_days, 300 AS debt_period_age_days, 40 AS equity_period_age_days,
       con_id = 6 AS any_amendment, FALSE AS any_revision, 'reported_total' AS debt_completeness,
       1e8 * con_id AS market_cap_shares, 1 + dayofyear(trading_date) % 90 AS shares_age_days,
       CAST(CASE WHEN con_id = 7 THEN 3 ELSE 0 END AS BIGINT) AS missing_mask,
       con_id = 8 AS any_alias_conflict, con_id <> 10 AS currency_consistent
FROM prices_daily_model_v1 JOIN ids i USING (con_id) WHERE con_id < 100 AND con_id <> 5;""")
    series = sorted(set(re.findall(r"series_id = '(\w+)'", SPEC.read_text())))
    con.execute(f"""
CREATE TABLE macro_daily_asof AS
SELECT trading_date, series_id, 2 + (hash(series_id, trading_date) % 1000) / 1000.0 + 0.001 * (trading_date - DATE '2021-01-01') AS value,
       1 + (trading_date - DATE '2021-01-01') % 30 AS age_days,
       CAST(CASE WHEN series_id = 'ICSA' THEN date_trunc('week', trading_date - 7)
                 ELSE date_trunc('month', trading_date - 35) END AS DATE) AS observation_date
FROM days, (SELECT unnest({series}) AS series_id);
""")
    con.execute("""
CREATE TABLE sec_description_embeddings_model_v1 (
  section_id VARCHAR PRIMARY KEY, available_date DATE, embedding BLOB
);
CREATE TABLE sec_description_asof_model_v1 (
  con_id BIGINT, section_id VARCHAR, valid_from DATE, valid_until DATE
);
""")
    descriptions = []
    intervals = []
    for con_id in range(1, 12):
        if con_id == 5:
            continue
        dates = [date(2021, 3, 15) + timedelta(days=90 * index) for index in range(10)]
        for index, available in enumerate(dates):
            section_id = f"description-{con_id}-{index}"
            vector = (
                np.sin(np.arange(768, dtype=np.float32) * (con_id + 1 + index / 10) / 97.0)
                + con_id / 100.0 + index / 1000.0
            ).astype("<f2").tobytes()
            descriptions.append((section_id, available, vector))
            intervals.append((
                con_id, section_id, available,
                dates[index + 1] if index + 1 < len(dates) else None,
            ))
    con.executemany("INSERT INTO sec_description_embeddings_model_v1 VALUES (?, ?, ?)", descriptions)
    con.executemany("INSERT INTO sec_description_asof_model_v1 VALUES (?, ?, ?, ?)", intervals)
    con.close()


def perturb(path: Path, cutoff: str, holdout: list[int]) -> None:
    """Change every value after ``cutoff``, and holdout rows at every date.

    Holdout closes and volumes move by a per-row factor that keeps close * volume
    (and so the liquidity strata of the split assignment) unchanged.
    """
    con = duckdb.connect(str(path))
    ids = ", ".join(map(str, holdout)) or "NULL"
    factor = "(1 + 0.2 * (hash(con_id, trading_date, 77) % 1000) / 1000.0)"
    con.execute(f"""
UPDATE prices_daily_model_v1 SET close = close * 1.7, open = open * 1.3, high = high * 2, volume = volume * 3 WHERE trading_date > DATE '{cutoff}';
UPDATE prices_daily_model_v1 SET close = close * {factor}, volume = volume / {factor}, open = open * 1.05, high = high * 1.3
WHERE con_id IN ({ids});
UPDATE sec_daily_fundamentals_model_v1 SET log_assets = log_assets + 3, revenue_q_to_assets = 0.9 WHERE trading_date > DATE '{cutoff}' OR con_id IN ({ids});
UPDATE macro_daily_asof SET value = value * 5 WHERE trading_date > DATE '{cutoff}';""")
    con.execute(f"""UPDATE sec_description_embeddings_model_v1 SET embedding = ? WHERE available_date > DATE '{cutoff}'
OR section_id IN (SELECT section_id FROM sec_description_asof_model_v1 WHERE con_id IN ({ids}))""",
                [np.full(768, 3.0, "<f2").tobytes()])
    con.close()


def splits_of(database: Path) -> list[tuple]:
    con = duckdb.connect()
    con.execute(f"ATTACH '{database}' AS src (READ_ONLY)")
    con.execute("ATTACH ':memory:' AS w")
    con.execute("USE w")
    con.execute("SET search_path = 'w.main,src.main'")
    con.execute(company_splits_sql(load_spec(SPEC)))
    return con.execute("SELECT con_id, company_key, split_group FROM splits ORDER BY con_id").fetchall()


@unittest.skipUnless(HAS_EXTRAS, "requires torch and the data extras (duckdb, numpy)")
class DataLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.db, cls.cache = root / "replica.duckdb", root / "cache"
        make_replica(cls.db)
        cls.meta = build_cache(SPEC, cls.db, cls.cache, "B", chunk_days=200, threads=1)
        cls.splits = splits_of(cls.db)
        cls.holdout = [c for c, _, g in cls.splits if g == "holdout"]
        cls.perturbed, cls.after_val = root / "cache_perturbed", root / "cache_after_val"
        for cache, cutoff, holdout in ((cls.perturbed, CUTOFF, cls.holdout), (cls.after_val, VAL_END, [])):
            shutil.copyfile(cls.db, root / "perturbed.duckdb")
            perturb(root / "perturbed.duckdb", cutoff, holdout)
            build_cache(SPEC, root / "perturbed.duckdb", cache, "B", chunk_days=200, threads=1)
        cls.names = cls.meta["features"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_spec_rejects_survivorship_and_future_sql(self) -> None:
        raw = tomllib.loads(SPEC.read_text())
        for sql in ("bar_count", "source = 'provider_a'", "close_masked", "continuity_break", "collection_gap", "source_raw_id", "comparative_backfill_mask", "fwd_5", "lead(r) OVER w",
                    "lag(r, -1) OVER w", "avg(r) OVER (PARTITION BY con_id)", "percent_rank() OVER w"):
            raw["block"][0]["feature"][0]["sql"] = sql
            with self.assertRaises(ValueError):
                spec_from_dict(raw)

    def test_example_spec_channels(self) -> None:
        spec = load_spec(SPEC)
        expected = {"ohlc_ok", "ret_1d", "mom_20", "vol_gk_20", "spy_ret", "sec_present", "log_assets", "cash_ta",
                    "short_rate", "vix", "sec_desc_ok", "sec_desc_age"} | {f"sec_pc_{i:02d}" for i in range(8)}
        for model in ("A", "B"):
            self.assertEqual({c for b in spec.blocks_for(model) for f in b.features for c in f.channels}, expected)
        self.assertEqual(len(self.names), 20)

    def test_optional_blocks_are_skipped_when_absent(self) -> None:
        self.assertEqual(self.meta["skipped_blocks"], [])
        self.assertEqual(sum(size for _, size in self.meta["feature_groups"]), len(self.names))
        spec = Path(self.tmp.name) / "absent.toml"  # the description block, optional, reading a table that does not exist
        spec.write_text(SPEC.read_text().replace(
            'requires = ["sec_description_asof_model_v1", "sec_description_embeddings_model_v1"]',
            'optional = true\nrequires = ["absent_table"]'))
        model_a = build_cache(spec, self.db, Path(self.tmp.name) / "cache_a", "A", chunk_days=400)
        self.assertEqual(model_a["skipped_blocks"], ["sec_descriptions"])
        self.assertEqual([name for name, _ in model_a["feature_groups"]][-1], "macro")

    def test_split_assignment_is_deterministic_and_company_level(self) -> None:
        self.assertEqual(self.splits, splits_of(self.db))
        groups = {c: g for c, _, g in self.splits}
        self.assertEqual(groups[1], groups[2])  # two listings of CIK 1001
        self.assertEqual(set(groups.values()), {"core", "extended", "holdout"})
        train = CacheSplit(self.cache, "train")
        self.assertFalse((train.target & (train.group == 2)).any())  # holdout is never trained
        cutoff = int(np.datetime64(CUTOFF, "D").astype(int))
        self.assertTrue((train.day[train.target & (train.group == 0)] <= cutoff).all())
        self.assertTrue((train.day[train.target & (train.group == 1)] <= cutoff).all())  # select phase: out-of-time val for all

    def test_final_phase_extends_training(self) -> None:
        build_cache(SPEC, self.db, Path(self.tmp.name) / "cache_final", "A", chunk_days=400, phase="final")
        train = CacheSplit(Path(self.tmp.name) / "cache_final", "train")
        cutoff, val_end = (int(np.datetime64(t, "D").astype(int)) for t in (CUTOFF, VAL_END))
        core, extended = train.day[train.target & (train.group == 0)], train.day[train.target & (train.group == 1)]
        self.assertTrue((core > cutoff).any() and (core <= val_end).all())  # core adds the validation year
        self.assertTrue((extended > val_end).any())  # extended trains on later years
        self.assertFalse((train.target & (train.group == 2)).any())
        with self.assertRaises(AssertionError):  # no validation era in the final phase
            CacheSplit(Path(self.tmp.name) / "cache_final", "val")

    def test_labels_follow_valid_links_only(self) -> None:
        con = duckdb.connect(str(self.db), read_only=True)
        prices = con.execute("SELECT con_id, CAST(trading_date - DATE '1970-01-01' AS INTEGER), close, split_ratio, "
                             "previous_link_valid FROM prices_daily_model_v1 WHERE con_id < 100 ORDER BY con_id, trading_date").fetchall()
        log_return, dates = {}, {}
        for (c0, _, p0, _, _), (c1, d1, p1, s1, ok) in zip(prices, prices[1:]):
            log_return[(c1, d1)] = np.log(p1 * s1 / p0) if c0 == c1 and ok else np.nan
        for c, d, *_ in prices:
            dates.setdefault(c, []).append(d)
        position = {(c, d): i for c, ds in dates.items() for i, d in enumerate(ds)}
        day = lambda text: int(np.datetime64(text, "D").astype(int))  # noqa: E731
        era_end = {("train", 0): day(CUTOFF), ("train", 1): day(CUTOFF), "val": day("2024-12-31"), "test": day("2026-10-02")}
        checked = 0
        for split in ("train", "val", "test"):
            rows = CacheSplit(self.cache, split)
            for i in np.flatnonzero(rows.target):
                c, d = int(rows.con_id[i]), int(rows.day[i])
                limit = era_end[(split, int(rows.group[i]))] if split == "train" else era_end[split]
                for j, h in enumerate(self.meta["horizons"]):
                    window = dates[c][position[(c, d)] + 1 : position[(c, d)] + 1 + h]
                    expected = np.nan if len(window) < h or window[-1] > limit else sum(log_return[(c, x)] for x in window)
                    if np.isnan(expected):
                        self.assertTrue(np.isnan(rows.labels[i, j]), (split, c, d, h))
                    else:
                        self.assertAlmostEqual(float(rows.labels[i, j]), expected, places=5)
                        checked += 1
        self.assertGreater(checked, 1000)
        train = CacheSplit(self.cache, "train")
        brk = day("2023-06-15")
        before = train.target & (train.con_id == 3) & (train.day < brk)
        self.assertTrue(np.isnan(train.labels[before][-1]).all())  # the last row before the break has no label
        self.assertTrue(np.isnan(train.labels[before][-20:, 2]).all())  # no 20-day label crosses it

    def test_features_ignore_later_data(self) -> None:
        cutoff = int(np.datetime64(CUTOFF, "D").astype(int))
        for split in ("train", "val", "test"):
            a, b = CacheSplit(self.cache, split), CacheSplit(self.perturbed, split)
            keep_a = (a.day <= cutoff) & ~np.isin(a.con_id, self.holdout)
            keep_b = (b.day <= cutoff) & ~np.isin(b.con_id, self.holdout)
            np.testing.assert_array_equal(a.con_id[keep_a], b.con_id[keep_b])
            np.testing.assert_array_equal(a.day[keep_a], b.day[keep_b])
            np.testing.assert_array_equal(a.features(np.flatnonzero(keep_a)), b.features(np.flatnonzero(keep_b)))

    def test_features_and_labels_ignore_data_after_their_era(self) -> None:
        a, b = CacheSplit(self.cache, "val"), CacheSplit(self.after_val, "val")
        np.testing.assert_array_equal(a.con_id, b.con_id)
        np.testing.assert_array_equal(a.labels, b.labels)
        np.testing.assert_array_equal(a.features(np.arange(len(a))), b.features(np.arange(len(b))))

    def test_normalization_uses_unmasked_training_rows(self) -> None:
        rows = CacheSplit(self.cache, "train")
        x = rows.features(np.flatnonzero(rows.target & (rows.day <= np.datetime64(CUTOFF, "D").astype(int))))
        fitted = [f.name for b in load_spec(SPEC).blocks for f in b.features if f.norm == "train_z"]
        for name in fitted:
            v = x[:, self.names.index(name)]
            v = v[np.isfinite(v)]
            if v.std() > 0:  # fit rows end up centred and scaled; masked rows (e.g. news-free days) are not among them
                self.assertAlmostEqual(float(v.mean()), 0.0, delta=0.05, msg=name)
                self.assertAlmostEqual(float(v.std()), 1.0, delta=0.05, msg=name)

    def test_normalization_fit_on_training_rows(self) -> None:
        self.assertEqual(json.loads((self.cache / "stats.json").read_text()),
                         json.loads((self.perturbed / "stats.json").read_text()))
        if (self.cache / "pca.npz").exists():
            first, second = np.load(self.cache / "pca.npz"), np.load(self.perturbed / "pca.npz")
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])

    def test_masks_and_missing_values(self) -> None:
        rows = CacheSplit(self.cache, "train")
        x = rows.features(np.arange(len(rows)))
        masks = [self.names.index(n) for n in ("ohlc_ok", "sec_present", "sec_desc_ok")]
        self.assertTrue(np.isin(x[:, masks], [0.0, 1.0]).all())
        self.assertTrue((x[rows.con_id == 5, self.names.index("sec_present")] == 0).all())  # no fundamentals rows
        self.assertTrue(np.isnan(x[rows.con_id == 5][:, [self.names.index("log_assets"), self.names.index("cash_ta")]]).all())
        self.assertTrue(np.isfinite(x[rows.target, self.names.index("spy_ret")]).mean() > 0.99)  # date-level source
        description_start = np.datetime64("2021-03-15", "D").astype(int)
        has_description = (rows.con_id < 12) & (rows.con_id != 5) & (rows.day >= description_start)
        description = [self.names.index(f"sec_pc_{index:02d}") for index in range(8)]
        self.assertTrue((x[~has_description, self.names.index("sec_desc_ok")] == 0).all())
        self.assertTrue(np.isnan(x[~has_description][:, description + [self.names.index("sec_desc_age")]]).all())
        self.assertTrue(np.isfinite(x[has_description & rows.target][:, description]).all())
        filled = JEPADataset(self.cache, "train", fill_value=0.0)
        self.assertTrue(all(torch.isfinite(filled[i]).all() for i in range(0, len(filled), 97)))

    def test_batches_match_model_contracts(self) -> None:
        width = len(self.names)
        jepa = JEPA(width, 16, hidden_dim=32, depth=1, dropout=0.0,
                    feature_groups=tuple((n, s) for n, s in self.meta["feature_groups"]),
                    mask_units=tuple(self.meta["mask_units"]))
        batch = next(iter(DataLoader(JEPADataset(self.cache, "train"), batch_size=64, shuffle=True)))
        self.assertEqual(batch.shape, (64, width))
        out = jepa(batch)
        self.assertTrue(out.valid_samples.all())
        pcs = out.feature_mask[:, [i for i, n in enumerate(self.names) if n.startswith("sec_pc_")]]
        self.assertTrue((pcs.all(1) | ~pcs.any(1)).all() and pcs.any())  # the 64 components are masked together

        data = SequenceDataset(self.cache, "val", 20, horizons=[1, 5], stride=10)
        rows = data.rows
        for start in data.starts:
            window = data.order[start:start + 20]
            self.assertEqual(len(set(rows.con_id[window])) + len(set(rows.seg[window])), 2)
            self.assertTrue((np.diff(rows.k[window]) == 1).all())
        supervised = sum(int((data.supervised[s:s + 20] & (np.arange(s, s + 20) > e - 10)
                              & (np.arange(s, s + 20) <= e)).sum()) for s, e in zip(data.starts, data.ends))
        self.assertEqual(supervised + data.skipped_targets, int(rows.target.sum()))

        x, y, valid = next(iter(DataLoader(data, batch_size=8)))
        self.assertEqual((x.shape, y.shape, valid.shape), ((8, 20, width), (8, 20, 2), (8, 20)))
        jepa.eval()
        inputs = jepa.encode(x.flatten(0, 1)).view(8, 20, -1)
        config = TransformerConfig(input_dim=16, sequence_lengths=(20,), target_dim=2, learning_rate=1e-3, num_epochs=1,
                                   jepa_checkpoint="test", d=32, out_proj=32, n_heads=2, n_layers=4, hidden_dim=64,
                                   moe_layers=(1, 2))
        self.assertEqual(Transformer(config)(inputs).pred.shape, (8, 20, 2))
        self.assertTrue(torch.isfinite(y[valid]).all())

        encoded = encode_rows(self.cache, "val", jepa.encode, batch_size=500)
        x, _, _ = SequenceDataset(self.cache, "val", 20, inputs=encoded)[0]
        self.assertEqual(x.shape, (20, 16))
        self.assertTrue(torch.isfinite(x).all())

    def test_rank_ic_of_a_perfect_forecast_is_one(self) -> None:
        class Echo(torch.nn.Module):  # forecasts exactly the labels it is fed as inputs
            def forward(self, x):
                return type("Result", (), {"pred": x})()

        labels = np.nan_to_num(CacheSplit(self.cache, "val").labels)
        ic = forecast_ic(Echo(), SequenceDataset(self.cache, "val", 20, inputs=labels), device=torch.device("cpu"))
        self.assertEqual(ic["horizons"], self.meta["horizons"])
        self.assertTrue(all(v > 0.999 for v in ic["ic"] + ic["per_stock_ic"]), ic)

    def test_training_stages_run_end_to_end(self) -> None:
        tiny = ModelConfig(
            runtime=RuntimeConfig(num_workers=0),
            jepa=JEPAConfig(input_dim=1, output_dim=64, hidden_dim=64, depth=2, residual=True, dropout=0.0,
                            mask_ratio=0.5, target_momentum=0.99, learning_rate=1e-3, num_epochs=2),
            transformer=TransformerConfig(input_dim=64, sequence_lengths=(20,), target_dim=1, learning_rate=1e-3,
                                          num_epochs=2, jepa_checkpoint="-", d=64, out_proj=64, n_heads=1, n_layers=4,
                                          hidden_dim=256, moe_layers=(1, 2), n_experts=2, max_seq=20),
        )
        out = Path(self.tmp.name) / "run"
        self.assertTrue(train_jepa(tiny, self.cache, out, batch_size=256, limit_batches=3).exists())
        self.assertTrue(train_transformer(tiny, self.cache, out, batch_size=4, accumulation_steps=1, eval_length=20,
                                          limit_batches=3).exists())
        metrics = evaluate(self.cache, out, length=20)  # raises if the Transformer was trained on another JEPA
        self.assertEqual(metrics["rank_ic"]["horizons"], self.meta["horizons"])

    def test_rows_are_members_plus_bounded_lookback(self) -> None:
        lookback = self.meta["lookback_sessions"]
        for split in ("train", "val", "test"):
            rows = CacheSplit(self.cache, split)
            for c in np.unique(rows.con_id):
                for s in np.unique(rows.seg[rows.con_id == c]):
                    mine = (rows.con_id == c) & (rows.seg == s)
                    k, target = rows.k[mine], rows.target[mine]
                    self.assertTrue(target.any())
                    gap = np.array([(k[target] >= kk).any() and (k[target][k[target] >= kk] - kk).min() <= lookback for kk in k])
                    self.assertTrue(gap.all(), (split, c, s))


if __name__ == "__main__":
    unittest.main()
