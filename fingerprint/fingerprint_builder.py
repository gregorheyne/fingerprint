from typing import NamedTuple

import numpy as np
import pandas as pd

from fingerprint.category_composition import build_composition_features
from fingerprint.behavior_features import (
    build_directionality_features, build_concentration_features,
    build_complexity_features, build_intensity_features, build_temporal_features,
)


class FingerprintResult(NamedTuple):
    """`fp`: the normalized fingerprint (Bayesian-shrunk + CLR composition
    shares, log1p-transformed scale features) -- what PCA/UMAP/clustering
    should consume.
    `fp_raw`: the same features in plain, business-legible units (actual
    trade counts, dollar amounts, plain percentage shares, days) -- no
    Bayesian shrinkage, no CLR, no log transform. Where a feature is already
    scale-free (e.g. `direction:buy_share_dollar`), `fp_raw` holds the exact
    same value as `fp`; see `docs/fingerprint-features.md` for the full
    per-feature breakdown of which columns differ and which don't.
    Supports both `fp, fp_raw = build_fingerprint(...)` and
    `result.fp` / `result.fp_raw`.
    """
    fp: pd.DataFrame
    fp_raw: pd.DataFrame


# ---------------------------------------------------------------------------
# Full fingerprint: composition-of-trading + behavioral blocks
# ---------------------------------------------------------------------------

def build_fingerprint(trades: pd.DataFrame, client_col: str, dimensions: list[str],
                      value_col: str = "amount_usd", date_col: str = "trade_date",
                      side_col: str = "side", buy_label: str = "buy",
                      aum: pd.Series | None = None, alpha: float = 10.0,
                      keep: tuple[str, ...] = ("count", "gap"),
                      use_effective_n: bool = False) -> FingerprintResult:
    """Assemble the full client fingerprint for PCA/clustering.

    Combines `category_composition.build_composition_features` (composition
    across instrument categories) with five behavioral blocks from
    `behavior_features`:
      - directionality:  accumulator vs churner
      - concentration:   diversified vs concentrated (HHI/entropy)
      - complexity:      plain-vanilla vs derivative/option usage
      - intensity:       SCALE features (log-transformed, optionally
                          AUM-normalized via `aum`) -- see its docstring for
                          why this block is kept small
      - temporal:         steady vs bursty inter-trade timing

    Every block builder returns `(normalized, raw)` -- see
    `FingerprintResult` and `docs/fingerprint-features.md`. All blocks are
    client x feature matrices sharing the same client index contract, so
    each side (normalized / raw) concatenates directly. Any NaNs -- e.g.
    temporal features for clients with fewer than 2 trades -- are
    median-filled independently on each table so downstream PCA/clustering
    (on `fp`) and business reporting (on `fp_raw`) never see missing values.
    """
    blocks = [
        build_composition_features(trades, client_col, dimensions,
                                   alpha=alpha, value_col=value_col,
                                   keep=keep, use_effective_n=use_effective_n),
        build_directionality_features(trades, client_col, value_col, side_col, buy_label),
        build_concentration_features(trades, client_col, dimensions, value_col),
        build_complexity_features(trades, client_col, value_col=value_col),
        build_intensity_features(trades, client_col, date_col, value_col, aum),
        build_temporal_features(trades, client_col, date_col),
    ]
    norm_blocks, raw_blocks = zip(*blocks)
    fp = pd.concat(norm_blocks, axis=1)
    fp_raw = pd.concat(raw_blocks, axis=1)
    return FingerprintResult(fp=fp.fillna(fp.median()), fp_raw=fp_raw.fillna(fp_raw.median()))


def raw_threshold(fp_col: str, threshold: float) -> tuple[str, str, float | None]:
    """The single source of truth for the `fp` <-> `fp_raw` column-naming
    convention (see `FingerprintResult`/`docs/fingerprint-features.md`).
    Given a decision-tree split on an `fp` column, returns
    `(raw_col, kind, raw_threshold)`:

    - `kind="identical"`: direction/concentration/complexity columns and
      `temporal:cv_gap`. `raw_col == fp_col` and `raw_threshold == threshold`
      unchanged -- `fp`/`fp_raw` already hold the same value.
    - `kind="log1p"`: intensity/temporal `log_*` columns. `raw_col` has the
      `log_` segment stripped; `raw_threshold = expm1(threshold)`, an EXACT
      inverse since `log1p` is strictly increasing:
      `log1p(raw) <= t  <=>  raw <= expm1(t)`.
    - `kind="clr"`: composition/gap columns (`{dimension}:count/dollar/gap:
      {category}`). `raw_col == fp_col` (same name in `fp_raw`, different
      value) but `raw_threshold` is `None`: a CLR value depends on a
      client's whole category vector, not just this one column, so no
      single raw-share number reproduces the same split rule for every
      client. Callers needing a business-legible description for these must
      fall back to a data-driven summary (see
      `visualization.plot_decision_tree`'s `raw_composition_summary`).

    Every caller that needs the `fp_raw` column name for an `fp` column --
    exact conversion or not -- should go through this function rather than
    re-deriving or assuming the name, so the naming convention lives in
    exactly one place.
    """
    prefix = fp_col.split(":", 1)[0]
    if prefix in ("direction", "concentration", "complexity") or fp_col == "temporal:cv_gap":
        return fp_col, "identical", threshold
    if prefix in ("intensity", "temporal") and ":log_" in fp_col:
        return fp_col.replace(":log_", ":"), "log1p", float(np.expm1(threshold))
    return fp_col, "clr", None


'''
client_col = "client_id"
alpha=10.0
keep=("count", "gap")
value_col = "amount_usd"
value_col = None
use_effective_n = False
category_col = 'asset_class'
evidence_n = n_trades
'''
