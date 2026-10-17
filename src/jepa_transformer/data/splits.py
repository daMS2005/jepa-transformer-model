"""Company-level split groups and per-row train/val/test eras, as DuckDB SQL.

Groups are assigned per company (latest mapped CIK, else con_id) so share classes
never straddle groups. Companies are stratified by SIC division x liquidity
tercile (median dollar volume up to ``liquidity_end``) and ordered within each
stratum by a salted md5; the first ``core_fraction`` are core, the next
``extended_fraction`` extended, the rest holdout.

Eras by ``phase``. select (model selection): every trained group trains through
``train_end``, validates to ``val_end`` and tests to ``test_end``, so validation is
out of time for all. final: core trains through ``val_end`` and extended through
``extended_train_end``; both and holdout test after their training to ``test_end``
(only dates after ``extended_train_end`` are out of time for every group).
Holdout is never trained.
"""

from __future__ import annotations

from jepa_transformer.data.spec import Spec

SPLITS = ("train", "val", "test")
GROUPS = ("core", "extended", "holdout")


def literal(value) -> str:
    """Quote a config value as a SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def company_splits_sql(spec: Spec) -> str:
    s = spec.splits
    return f"""
CREATE TABLE splits AS
WITH members AS (
  SELECT DISTINCT con_id FROM universe_membership WHERE universe_name = {literal(spec.members)}
), cik AS (
  SELECT con_id, arg_max(cik, valid_from) AS cik FROM sec_security_mappings GROUP BY con_id
), liq AS (
  SELECT p.con_id, median(p.close * p.volume) AS dollar_volume
  FROM prices_daily_model_v1 p JOIN members USING (con_id)
  WHERE p.trading_date <= DATE {literal(s.liquidity_end)} AND p.trade_valid
  GROUP BY p.con_id
), company AS (
  SELECT mb.con_id,
         coalesce('cik:' || cik.cik, 'con:' || mb.con_id) AS company_key,
         coalesce(CAST(floor(TRY_CAST(sc.sic AS INTEGER) / 1000) AS VARCHAR), 'industry:' || s.industry, 'unknown') AS sector,
         liq.dollar_volume
  FROM members mb
  LEFT JOIN cik USING (con_id)
  LEFT JOIN sec_companies sc ON sc.cik = cik.cik
  LEFT JOIN securities s ON s.con_id = mb.con_id
  LEFT JOIN liq USING (con_id)
), firms AS (
  SELECT company_key, min(sector) AS sector, max(coalesce(dollar_volume, 0)) AS dv FROM company GROUP BY company_key
), strata AS (
  SELECT *, ntile(3) OVER (ORDER BY dv, company_key) AS liq_tercile FROM firms
), ranked AS (
  SELECT *, percent_rank() OVER (
           PARTITION BY sector, liq_tercile ORDER BY md5(company_key || ':' || {literal(s.salt)})
         ) AS u
  FROM strata
)
SELECT c.con_id, c.company_key, r.sector, r.liq_tercile,
       CASE WHEN r.u < {s.core_fraction} THEN 'core'
            WHEN r.u < {s.core_fraction + s.extended_fraction} THEN 'extended'
            ELSE 'holdout' END AS split_group
FROM company c JOIN ranked r USING (company_key)
"""


def panel_sql(spec: Spec) -> str:
    """Member-days plus up to ``lookback`` earlier sessions of the same continuity segment.

    ``split`` is set on member rows only; ``ctx_<split>`` marks rows a split needs
    (its member rows and their lookback); ``era_end`` bounds that row's labels.
    """
    s = spec.splits
    d = {k: f"DATE {literal(getattr(s, k))}" for k in ("train_end", "val_end", "test_end", "extended_train_end")}
    final = s.phase == "final"
    # [core/holdout boundary, extended boundary]; holdout rows before the first are unused.
    train_end = [d["val_end"] if final else d["train_end"], d["extended_train_end"] if final else d["train_end"]]
    phase_select = "false" if final else "true"
    ctx = ",\n    ".join(
        f"coalesce(max(CAST(split = '{name}' AS INTEGER)) OVER f, 0) = 1 AS ctx_{name}" for name in SPLITS
    )
    return f"""
CREATE TABLE panel AS
WITH r AS (
  SELECT p.con_id, p.trading_date, p.seg, md.con_id IS NOT NULL AS member, s.split_group AS grp,
    CASE WHEN md.con_id IS NULL THEN NULL
         WHEN p.trading_date <= {train_end[0]} THEN (CASE WHEN s.split_group <> 'holdout' THEN 'train' END)
         WHEN p.trading_date <= {train_end[1]} AND s.split_group = 'extended' THEN 'train'
         WHEN p.trading_date <= {d['val_end']} AND {phase_select} THEN 'val'
         WHEN p.trading_date <= {d['val_end']} THEN NULL
         WHEN p.trading_date <= {d['test_end']} THEN 'test'
    END AS split
  FROM px p JOIN splits s USING (con_id)
  LEFT JOIN member_days md USING (con_id, trading_date)
), c AS (
  SELECT *,
    {ctx}
  FROM r
  WINDOW f AS (PARTITION BY con_id, seg ORDER BY trading_date ROWS BETWEEN CURRENT ROW AND {spec.lookback} FOLLOWING)
)
SELECT con_id, trading_date, member, grp, split, {', '.join(f'ctx_{name}' for name in SPLITS)},
       CASE split WHEN 'train' THEN (CASE grp WHEN 'extended' THEN {train_end[1]} ELSE {train_end[0]} END)
                  WHEN 'val' THEN {d['val_end']} WHEN 'test' THEN {d['test_end']} END AS era_end
FROM c
WHERE ctx_train OR ctx_val OR ctx_test
ORDER BY trading_date, con_id
"""


def labels_sql(spec: Spec) -> str:
    """Forward log price returns over closes t+1..t+h, NULL unless every link is valid.

    r_t = ln(close_t * split_ratio_t / close_{t-1}) only when ``previous_link_valid``,
    so a sum over h links never crosses a continuity break or missing session.
    ``end_h`` is the date of the last close the label uses.
    """
    windows = ", ".join(
        f"h{h} AS (PARTITION BY con_id ORDER BY trading_date ROWS BETWEEN 1 FOLLOWING AND {h} FOLLOWING)"
        for h in spec.horizons
    )
    columns = ", ".join(
        f"CASE WHEN count(lr) OVER h{h} = {h} THEN sum(lr) OVER h{h} END AS fwd_{h}, "
        f"lead(trading_date, {h}) OVER c AS end_{h}"
        for h in spec.horizons
    )
    return f"""
CREATE TABLE labels AS
WITH b AS (
  SELECT con_id, trading_date,
         CASE WHEN previous_link_valid AND close > 0 AND lag(close) OVER c > 0
              THEN ln(close * coalesce(split_ratio, 1) / lag(close) OVER c) END AS lr
  FROM prices_daily_model_v1
  WHERE con_id IN (SELECT con_id FROM splits)
  WINDOW c AS (PARTITION BY con_id ORDER BY trading_date)
)
SELECT con_id, trading_date, {columns}
FROM b
WINDOW c AS (PARTITION BY con_id ORDER BY trading_date), {windows}
"""
