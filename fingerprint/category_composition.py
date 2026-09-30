"""
Trading-fingerprint pipeline.

Design: two grains with a single seam.
  1. LONG trades  --groupby/agg-->  (client x category) matrix   [pandas' job]
  2. matrix  --shrink, CLR, gap-->  fingerprint                  [pure matrix functions]

Every function below the aggregation step is "composition matrix in, matrix out":
it does not know whether the input came from trade counts or dollar volumes, nor
which categorical dimension it belongs to. That is what makes them reusable across
both weightings and every dimension.

All matrices are pandas DataFrames indexed by client_id with category columns, so
index/column alignment is automatic and population broadcasting cannot misalign.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Grain 1 -> Grain 2: the single aggregation seam (pandas' split-apply-combine)
# ---------------------------------------------------------------------------

def aggregate(trades: pd.DataFrame, client_col: str, category_col: str,
              value_col: str | None = None) -> pd.DataFrame:
    """Long trades -> (client x category) matrix.

    value_col=None -> trade counts ("where attention goes").
    value_col set  -> summed volume ("where money goes").

    The category universe is inferred from `trades[category_col]` (sorted for a
    stable column order), so every client is reindexed against the same fixed
    set of columns and any category a client never traded materialises as a
    real zero column rather than a dropped one.
    """
    categories = sorted(trades[category_col].unique())
    if value_col is None:
        agg = trades.groupby([client_col, category_col]).size()
    else:
        agg = trades.groupby([client_col, category_col])[value_col].sum()
    return (agg.unstack(category_col)
               .reindex(columns=categories)  # reindex() effectively reorders
               .fillna(0.0))


# ---------------------------------------------------------------------------
# Grain 2: pure matrix functions (reused for count AND dollar, every dimension)
# ---------------------------------------------------------------------------

def to_shares(mat: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise a count/volume matrix to shares summing to 1."""
    return mat.div(mat.sum(axis=1), axis=0)


def population_mean(mat: pd.DataFrame) -> pd.Series:
    """Pooled population composition (column totals / grand total).

    Must be strictly positive on every included category, otherwise the alpha*m
    prior term cannot lift a structural zero and CLR's log will blow up. Keep the
    category universe to categories that actually occur (the 'other' bin absorbs
    the rest), and this holds automatically.
    """
    tot = mat.sum(axis=0)
    return tot / tot.sum()


def shrink(shares: pd.DataFrame, evidence_n: pd.Series,
           population: pd.Series, alpha: float) -> pd.DataFrame:
    """Dirichlet / empirical-Bayes shrinkage, convex-combination form:

        shrunk = lambda * shares + (1 - lambda) * population,
        lambda = n / (n + alpha)

    `evidence_n` is the count that governs *trust* -- always trade count (or the
    Kish effective count), never dollar volume, even when `shares` are dollar
    shares. Decoupling "what is shrunk" from "how much evidence" is the whole
    point of using this form for the money-weighted block.

    Output sums to 1 per client and contains no zeros (CLR-safe).
    """
    lam = evidence_n.reindex(shares.index) / (evidence_n.reindex(shares.index) + alpha)
    prior = pd.DataFrame(np.outer(1.0 - lam.values, population.values),
                         index=shares.index, columns=shares.columns)
    return shares.mul(lam, axis=0) + prior


def clr(mat: pd.DataFrame) -> pd.DataFrame:
    """Centered log-ratio, computed in log space for stability:

        clr_i = log(x_i) - mean_j log(x_j)

    (mean of logs == log of geometric mean). Requires strictly positive input,
    which `shrink` guarantees.
    Log-space computation is just a numerically stable reformulation
    (avoids computing the product of possibly many small numbers before
    taking one big log) — mathematically identical to the direct formula,
    not an approximation.
    """
    logm = np.log(mat)
    return logm.sub(logm.mean(axis=1), axis=0)


def effective_n(trades: pd.DataFrame, client_col: str, value_col: str) -> pd.Series:
    """Kish effective sample size per client from trade sizes:

        n_eff = (sum v)^2 / sum(v^2)

    Equals the trade count when all trades are equal size, collapses toward 1
    when one whale dominates. Optional: pass as `evidence_n` for the dollar block
    so whale-driven compositions get shrunk harder.
    """
    grp = trades.groupby(client_col)[value_col]
    s1 = grp.sum() ** 2
    s2 = grp.apply(lambda v: (v ** 2).sum())
    return s1 / s2


# ---------------------------------------------------------------------------
# Block assembly and orchestration
# ---------------------------------------------------------------------------

def clr_block(trades: pd.DataFrame, client_col: str, category_col: str,
              alpha: float, evidence_n: pd.Series,
              value_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One CLR block for one dimension and one weighting.

    For the count block: value_col=None, evidence_n = trade counts.
    For the dollar block: value_col='amount_usd', evidence_n = trade counts
    (or effective_n) -- NOT dollar totals.

    Returns `(clr_df, raw_share_df)`: the CLR-transformed, Bayesian-shrunk
    result used for PCA/clustering, alongside the plain per-category share
    (`to_shares(mat)`, before shrinkage and before CLR) -- a business-legible
    "what fraction actually landed here" number, e.g. "32% of trades were in
    equities".
    """
    mat = aggregate(trades, client_col, category_col, value_col)
    shares = to_shares(mat)
    population = population_mean(mat)
    shrunk = shrink(shares, evidence_n, population, alpha)
    return clr(shrunk), shares


def build_composition_features(trades: pd.DataFrame, client_col: str,
                               dimensions: list[str],
                               alpha: float = 10.0, value_col: str = "amount_usd",
                               keep: tuple[str, ...] = ("count", "gap"),
                               use_effective_n: bool = False,
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the full composition-of-trading feature block (client x features).

    `dimensions`: list of category columns, e.g.
        ["asset_class", "product_class", "sector"]

    Each column's category universe is inferred from `trades` itself (see
    `aggregate`), so callers don't need to enumerate categories up front.

    `keep`: which of {"count", "dollar", "gap"} blocks to retain per dimension.
        Keep exactly TWO -- all three are exact linear redundancy because
        gap == clr(dollar) - clr(count). Default keeps count + gap.

    Columns are MultiIndex-prefixed "{dim}:{block}:{category}" so blocks stay
    identifiable and independently weightable downstream.

    Returns `(fp_block, fp_block_raw)`, sharing the same column names: the
    CLR/shrinkage-based block used for PCA/clustering, and a business-legible
    companion of plain, unshrunk shares (and, for "gap", a plain share-point
    difference rather than a CLR difference).
    """
    if len(set(keep)) != 2:
        raise ValueError("keep exactly two of {'count','dollar','gap'} "
                         "-- all three are linearly redundant")

    # Evidence counts govern *trust*, so both are in TRADE-COUNT units -- never
    # dollar volume. effective_n is still a trade count (Kish), it just discounts
    # whale-dominated clients. Passing dollar amount here would reintroduce the
    # magnitude-mismatch bug the convex-combination form exists to avoid.
    n_trades = trades.groupby(client_col).size()          # evidence, shared across dims
    evidence_dollar = (effective_n(trades, client_col, value_col)
                       if use_effective_n else n_trades)

    out, out_raw = [], []
    for category_col in dimensions:
        clr_count, raw_count = clr_block(trades, client_col, category_col,
                                         alpha, n_trades, value_col=None)
        clr_dollar, raw_dollar = clr_block(trades, client_col, category_col,
                                           alpha, evidence_dollar, value_col=value_col)
        gap = clr_dollar - clr_count
        raw_gap = raw_dollar - raw_count

        available = {"count": (clr_count, raw_count), "dollar": (clr_dollar, raw_dollar),
                    "gap": (gap, raw_gap)}
        for block_name in keep:
            norm_block, raw_block = available[block_name]
            prefix = f"{category_col}:{block_name}:"
            out.append(norm_block.add_prefix(prefix))
            out_raw.append(raw_block.add_prefix(prefix))

    return pd.concat(out, axis=1), pd.concat(out_raw, axis=1)
