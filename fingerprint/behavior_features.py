"""
Behavioral feature blocks that complement the category_composition features.

Each function below is "long trades in, client x feature matrix out" -- the same
shape contract as `category_composition.build_composition_features` -- so all
blocks concatenate cleanly on the client index in `fingerprint.build_fingerprint`.

Five blocks, matching five distinct behavioral questions:
  - directionality:   accumulator vs churner (buy-share, net/gross turnover)
  - concentration:    diversified vs concentrated (HHI, entropy, distinct count)
  - complexity:       plain-vanilla vs sophisticated (derivative/option usage)
  - intensity:        SCALE features (log-transformed, optionally AUM-normalized --
                       kept deliberately small, see its docstring)
  - temporal:         steady vs bursty trading rhythm (inter-trade gap CV)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from fingerprint.category_composition import aggregate, to_shares


# ---------------------------------------------------------------------------
# Directionality
# ---------------------------------------------------------------------------

def build_directionality_features(trades: pd.DataFrame, client_col: str,
                                  value_col: str = "amount_usd",
                                  side_col: str = "side",
                                  buy_label: str = "buy",
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buy-share of volume/count and net-over-gross turnover.

    `amount_usd` is a trade *magnitude* (always positive), so gross turnover is
    just the per-client dollar sum and net turnover is buy-dollar minus
    sell-dollar. net_over_gross sits in [-1, 1]: near +-1 means volume flows
    almost entirely one way (an accumulator or a liquidator), near 0 means
    buys and sells roughly cancel out (a churner). All three outputs are
    ratios, so -- unlike the intensity block -- they need no log-transform or
    AUM normalization to be scale-free.

    Returns `(out, out)`: these features are already raw, business-legible
    ratios, so the non-normalized companion is identical to the normalized
    block -- there is nothing to un-transform.
    """
    is_buy = trades[side_col] == buy_label
    signed = np.where(is_buy, trades[value_col], -trades[value_col])

    grp = trades.groupby(client_col)
    total_dollar = grp[value_col].sum()
    total_count = grp.size()

    buy_dollar = (trades.loc[is_buy].groupby(client_col)[value_col].sum()
                  .reindex(total_dollar.index).fillna(0.0))
    buy_count = (trades.loc[is_buy].groupby(client_col).size()
                 .reindex(total_count.index).fillna(0.0))
    net_dollar = (pd.Series(signed, index=trades.index)
                  .groupby(trades[client_col]).sum())

    out = pd.DataFrame({
        "buy_share_dollar": buy_dollar / total_dollar,
        "buy_share_count": buy_count / total_count,
        "net_over_gross": net_dollar / total_dollar,
    }, index=total_dollar.index)
    out = out.add_prefix("direction:")
    return out, out


# ---------------------------------------------------------------------------
# Concentration / diversification
# ---------------------------------------------------------------------------

def _normalized_entropy(shares: pd.DataFrame, n_categories: int) -> pd.Series:
    """Shannon entropy over category shares, normalized by log(K) to [0, 1] so
    it's comparable across dimensions with different category counts.
    0 = all volume in one category, 1 = perfectly uniform over all K.
    0*log(0) is taken as 0 by convention (shares from `to_shares` can be
    exactly 0 for a category a client never touched).
    """
    p = shares.values
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    entropy = -terms.sum(axis=1)
    denom = np.log(n_categories) if n_categories > 1 else 1.0
    return pd.Series(entropy / denom, index=shares.index)


def build_concentration_features(trades: pd.DataFrame, client_col: str,
                                 dimensions: list[str],
                                 value_col: str = "amount_usd",
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HHI, normalized Shannon entropy, and distinct-category count, per
    dimension, both count-weighted ("where attention goes") and
    dollar-weighted ("where money goes"). Reuses `aggregate`/`to_shares` from
    category_composition so the category universe and zero-filling logic stay
    in one place. HHI and entropy are two views of the same thing (HHI penalizes
    concentration more sharply); keeping both is cheap and lets PCA pick
    whichever shape fits the data.

    Returns `(out, out)`: none of these features go through the Bayesian
    shrinkage / CLR pipeline (HHI and entropy are already computed on plain,
    unshrunk shares, and n_distinct is a raw count), so there is no separate
    non-normalized companion to compute -- both tables hold the same values.
    """
    out = {}
    for dim in dimensions:
        count_mat = aggregate(trades, client_col, dim, value_col=None)
        dollar_mat = aggregate(trades, client_col, dim, value_col=value_col)
        n_categories = count_mat.shape[1]

        count_shares = to_shares(count_mat)
        dollar_shares = to_shares(dollar_mat)

        out[f"{dim}:hhi_count"] = (count_shares ** 2).sum(axis=1)
        out[f"{dim}:hhi_dollar"] = (dollar_shares ** 2).sum(axis=1)
        out[f"{dim}:entropy_count"] = _normalized_entropy(count_shares, n_categories)
        out[f"{dim}:entropy_dollar"] = _normalized_entropy(dollar_shares, n_categories)
        out[f"{dim}:n_distinct"] = (count_mat > 0).sum(axis=1).astype(float)

    out = pd.DataFrame(out).add_prefix("concentration:")
    return out, out


# ---------------------------------------------------------------------------
# Complexity / sophistication
# ---------------------------------------------------------------------------

def build_complexity_features(trades: pd.DataFrame, client_col: str,
                              product_class_col: str = "product_class",
                              derivative_labels: tuple[str, ...] = ("option", "structured"),
                              option_label: str = "option",
                              value_col: str = "amount_usd",
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of activity in derivative/structured products, plus an
    options-specific cut, as a rough sophistication proxy. Count- and
    dollar-weighted shares can diverge meaningfully here (a client who
    dabbles in options on a handful of trades but keeps most dollars in
    plain equity looks very different from one running large option
    positions), so both are kept.

    Returns `(out, out)`: these are already raw shares/flags with no
    shrinkage or transform applied, so the non-normalized companion is
    identical.
    """
    grp = trades.groupby(client_col)
    total_count = grp.size()
    total_dollar = grp[value_col].sum()

    is_deriv = trades[product_class_col].isin(derivative_labels)
    deriv_count = (trades.loc[is_deriv].groupby(client_col).size()
                   .reindex(total_count.index).fillna(0.0))
    deriv_dollar = (trades.loc[is_deriv].groupby(client_col)[value_col].sum()
                    .reindex(total_dollar.index).fillna(0.0))

    is_option = trades[product_class_col] == option_label
    option_count = (trades.loc[is_option].groupby(client_col).size()
                    .reindex(total_count.index).fillna(0.0))

    out = pd.DataFrame({
        "derivative_share_count": deriv_count / total_count,
        "derivative_share_dollar": deriv_dollar / total_dollar,
        "option_share_count": option_count / total_count,
        "uses_options": (option_count > 0).astype(float),
    }, index=total_count.index)
    out = out.add_prefix("complexity:")
    return out, out


# ---------------------------------------------------------------------------
# Intensity (scale -- include sparingly)
# ---------------------------------------------------------------------------

def build_intensity_features(trades: pd.DataFrame, client_col: str,
                             date_col: str = "trade_date",
                             value_col: str = "amount_usd",
                             aum: pd.Series | None = None,
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trade count, average trade size, total volume, and trading rate --
    the only block that is NOT scale-invariant by construction (it directly
    encodes wealth/activity level rather than trading style). All four are
    log1p-transformed to tame heavy right skew and keep them from dominating
    PCA purely on magnitude.

    If `aum` (a client_id -> assets-under-management Series) is supplied, an
    additional log(1 + volume / AUM) turnover-ratio feature is added -- this
    is the standard way to strip wealth back out of an activity measure, so
    a small account trading its whole book looks similar to a large account
    trading its whole book, rather than the large account just "winning" on
    raw volume.

    Returns `(out, out_raw)`: `out` holds the log1p-transformed features used
    for PCA/clustering (`intensity:log_*`); `out_raw` holds the same
    quantities before the log1p transform, under the same names with the
    `log_` prefix dropped (`intensity:n_trades`, `intensity:avg_trade_size`,
    `intensity:total_volume`, `intensity:trades_per_day`, and -- if `aum` is
    given -- `intensity:turnover_ratio`), e.g. an actual trade count or
    dollar figure a business reader can use directly.
    """
    grp = trades.groupby(client_col)
    n_trades = grp.size()
    total_volume = grp[value_col].sum()
    avg_trade_size = total_volume / n_trades

    span_days = (grp[date_col].max() - grp[date_col].min()).dt.total_seconds() / 86400.0
    span_days = span_days.clip(lower=1.0)  # avoid div-by-zero for single/same-day clients
    trades_per_day = n_trades / span_days

    out = pd.DataFrame({
        "log_n_trades": np.log1p(n_trades),
        "log_avg_trade_size": np.log1p(avg_trade_size),
        "log_total_volume": np.log1p(total_volume),
        "log_trades_per_day": np.log1p(trades_per_day),
    }, index=n_trades.index)
    out_raw = pd.DataFrame({
        "n_trades": n_trades,
        "avg_trade_size": avg_trade_size,
        "total_volume": total_volume,
        "trades_per_day": trades_per_day,
    }, index=n_trades.index)

    if aum is not None:
        aum = aum.reindex(out.index)
        turnover_ratio = total_volume / aum
        out["log_turnover_ratio"] = np.log1p(turnover_ratio)
        out_raw["turnover_ratio"] = turnover_ratio

    return out.add_prefix("intensity:"), out_raw.add_prefix("intensity:")


# ---------------------------------------------------------------------------
# Temporal dynamics
# ---------------------------------------------------------------------------

def build_temporal_features(trades: pd.DataFrame, client_col: str,
                            date_col: str = "trade_date",
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inter-trade gap statistics: mean gap (log-transformed, since it's a
    time-scale that correlates with trading intensity) and coefficient of
    variation of the gap (std/mean, scale-free by construction). CV << 1 is
    a metronomic/steady trader (e.g. periodic rebalancing); CV >= 1 is a
    bursty trader -- long quiet spells punctuated by clusters of activity.
    Clients with fewer than 2 trades have no defined gap and come back as
    NaN; the caller (`build_fingerprint`) is responsible for imputing.

    Returns `(out, out_raw)`: `out_raw` holds the mean gap before the log1p
    transform (`temporal:mean_gap_days`, an actual day count) and the same
    `cv_gap` value as `out` (already scale-free, nothing to un-transform).
    """
    ordered = trades[[client_col, date_col]].sort_values([client_col, date_col])
    gap_days = (ordered.groupby(client_col)[date_col].diff()
               .dt.total_seconds() / 86400.0)  # NaN for each client's first trade

    mean_gap = gap_days.groupby(ordered[client_col]).mean()
    std_gap = gap_days.groupby(ordered[client_col]).std(ddof=0)
    cv_gap = (std_gap / mean_gap).where(mean_gap > 0)

    out = pd.DataFrame({
        "log_mean_gap_days": np.log1p(mean_gap),
        "cv_gap": cv_gap,
    })
    out_raw = pd.DataFrame({
        "mean_gap_days": mean_gap,
        "cv_gap": cv_gap,
    })
    return out.add_prefix("temporal:"), out_raw.add_prefix("temporal:")
