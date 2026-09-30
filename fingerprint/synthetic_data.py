import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic trade data for exercising build_fingerprint end-to-end.
# ---------------------------------------------------------------------------

def generate_synthetic_trades(n_clients: int = 200, n: int = 5000, seed: int = 0,
                              n_outlier_clients: int = 0):
    """Generate synthetic trades with latent per-client personas.

    Each persona is a latent client archetype: a probability vector per
    dimension, a lognormal (mean, sigma) for trade size, plus the
    behavioral knobs -- buy_prob (directionality), activity (relative
    trading frequency, i.e. intensity), and burstiness (temporal rhythm).
    Without a client-level latent factor like this, every client's true
    distribution is identical and the fingerprint differences across
    clients are pure sampling noise -- PCA/clustering would have no real
    structure to recover.

    `n_outlier_clients` appends that many extra clients, each with its OWN
    randomly drawn parameters (same shape as a persona entry) instead of
    sharing an archetype with anyone else -- true one-offs, not a 5th
    hidden cluster. They're labeled "outlier" in `client_persona`. Useful
    for checking how clustering handles clients that don't fit any known
    persona, e.g. whether HDBSCAN correctly flags them as noise (-1)
    instead of k-means/GMM forcing them into the nearest cluster.

    Returns
    -------
    trades : pd.DataFrame
        Columns: client_id, amount_usd, trade_date, side, and one column
        per entry in `dimensions`.
    dimensions : list[str]
    client_persona : pd.Series
        Ground-truth persona label per client_id, indexed by client_id
        ("outlier" for the extra non-persona clients).
    client_aum : pd.Series
        Synthetic AUM per client_id, correlated with (but noisier and
        larger than) the client's typical trade size.
    persona_names : list[str]
        The real persona archetypes only (excludes "outlier").
    """
    rng = np.random.default_rng(seed)

    dimensions = ["asset_class", "product_class", "sector"]
    categories = {
        "asset_class":   ["equity", "bond", "fx", "other"],
        "product_class": ["stock", "option", "structured", "other"],
        "sector":        ["tech", "utilities", "financials", "none"],
    }

    personas = {
        "growth_equity": {
            "asset_class":   [0.70, 0.05, 0.05, 0.20],
            "product_class": [0.55, 0.25, 0.05, 0.15],
            "sector":        [0.65, 0.05, 0.10, 0.20],
            "amount":        (10.5, 1.8),
            "buy_prob":      0.75,   # accumulator
            "activity":      1.3,
            "burstiness":    "bursty",   # buys in clusters (dips/news)
        },
        "income_bond": {
            "asset_class":   [0.10, 0.70, 0.05, 0.15],
            "product_class": [0.20, 0.05, 0.55, 0.20],
            "sector":        [0.10, 0.55, 0.25, 0.10],
            "amount":        (9.5, 1.0),
            "buy_prob":      0.55,   # mostly coupon reinvestment
            "activity":      0.6,
            "burstiness":    "steady",   # periodic reinvestment schedule
        },
        "fx_hedger": {
            "asset_class":   [0.05, 0.05, 0.80, 0.10],
            "product_class": [0.10, 0.10, 0.60, 0.20],
            "sector":        [0.05, 0.05, 0.10, 0.80],
            "amount":        (11.0, 2.2),
            "buy_prob":      0.50,   # balanced buy/sell hedge rolls
            "activity":      1.8,
            "burstiness":    "steady",   # regular hedge-roll cadence
        },
        "diversified": {
            "asset_class":   [0.25, 0.25, 0.25, 0.25],
            "product_class": [0.25, 0.25, 0.25, 0.25],
            "sector":        [0.25, 0.25, 0.25, 0.25],
            "amount":        (10.0, 1.5),
            "buy_prob":      0.60,
            "activity":      1.0,
            "burstiness":    "bursty",   # irregular rebalancing
        },
    }
    persona_names = list(personas)
    persona_weights = [0.30, 0.25, 0.25, 0.20]

    total_clients = n_clients + n_outlier_clients
    outlier_ids = list(range(n_clients, total_clients))

    # Rejection-sample a dirichlet draw for one composition dimension so it
    # stays at least `min_dist` (L2) from EVERY persona's own probability
    # vector for that dimension. Plain `rng.dirichlet(np.ones(k))` has mean
    # equal to the uniform vector, so a large share of unconstrained draws
    # land right on top of the "diversified" persona's exact (0.25, 0.25,
    # ...) composition -- an accidental overlap that let outlier clients get
    # embedded inside an existing persona's neighborhood (and then absorbed
    # into that cluster by UMAP/HDBSCAN) instead of landing in genuinely
    # unclaimed space. If `max_tries` is exhausted without clearing
    # `min_dist` (persona vectors can cover most of a low-dimensional
    # simplex), fall back to the farthest draw actually found rather than
    # an arbitrary last attempt.
    def _draw_away_from_personas(persona_vecs: np.ndarray, k: int,
                                 min_dist: float = 0.25, max_tries: int = 200) -> np.ndarray:
        best_draw, best_dist = None, -1.0
        for _ in range(max_tries):
            draw = rng.dirichlet(np.ones(k))
            dist = np.min(np.linalg.norm(persona_vecs - draw, axis=1))
            if dist > best_dist:
                best_draw, best_dist = draw, dist
            if dist >= min_dist:
                return draw
        return best_draw

    client_persona = pd.Series(
        np.concatenate([
            rng.choice(persona_names, size=n_clients, p=persona_weights),
            np.full(n_outlier_clients, "outlier", dtype=object),
        ]),
        index=pd.RangeIndex(total_clients, name="client_id"),
        name="persona",
    )

    # Each outlier client draws its own one-off parameter set (composition
    # per dimension, plus its own amount/buy_prob/activity/burstiness) --
    # unlike the four personas above, no two outlier clients share these, so
    # they can't form a cluster with each other either.
    persona_vecs = {dim: np.array([personas[name][dim] for name in persona_names])
                    for dim in dimensions}
    outlier_params = {
        c: {
            **{dim: _draw_away_from_personas(persona_vecs[dim], len(categories[dim]))
               for dim in dimensions},
            "amount":     (rng.uniform(7.0, 13.0), rng.uniform(0.8, 3.0)),
            "buy_prob":   rng.uniform(0.1, 0.9),
            "activity":   rng.uniform(0.3, 2.5),
            "burstiness": rng.choice(["steady", "bursty"]),
        }
        for c in outlier_ids
    }

    def params_of(c: int) -> dict:
        persona = client_persona.iloc[c]
        return outlier_params[c] if persona == "outlier" else personas[persona]

    # Trade counts are not uniform across clients: sampling client_id with
    # probability proportional to each client's "activity" weight is what
    # gives build_intensity_features' trade-count/trading-rate features real
    # persona structure instead of pure Poisson noise.
    client_activity = np.array([params_of(c)["activity"] for c in range(total_clients)])
    client_probs = client_activity / client_activity.sum()
    client_id = rng.choice(total_clients, size=n, p=client_probs)
    trade_persona = client_persona.values[client_id]

    trade_dims = {dim: np.empty(n, dtype=object) for dim in dimensions}
    amount_usd = np.empty(n)
    side = np.empty(n, dtype=object)
    trade_date = np.empty(n, dtype="datetime64[ns]")

    window_start = pd.Timestamp("2025-01-01")
    window_days = 365

    for persona in persona_names:
        mask = trade_persona == persona
        n_p = int(mask.sum())
        p = personas[persona]
        for dim in dimensions:
            trade_dims[dim][mask] = rng.choice(categories[dim], size=n_p,
                                               p=p[dim])
        mu, sigma = p["amount"]
        amount_usd[mask] = rng.lognormal(mu, sigma, n_p)
        side[mask] = rng.choice(["buy", "sell"], size=n_p,
                                p=[p["buy_prob"], 1 - p["buy_prob"]])

    # Outlier clients don't share params with each other, so their trades
    # are generated one client at a time rather than in one grouped pass.
    for c in outlier_ids:
        mask = client_id == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        p = outlier_params[c]
        for dim in dimensions:
            trade_dims[dim][mask] = rng.choice(categories[dim], size=n_c, p=p[dim])
        mu, sigma = p["amount"]
        amount_usd[mask] = rng.lognormal(mu, sigma, n_c)
        side[mask] = rng.choice(["buy", "sell"], size=n_c,
                                p=[p["buy_prob"], 1 - p["buy_prob"]])

    # Per-client inter-trade timing: "steady" clients get near-evenly-spaced
    # trades (low coefficient of variation of gaps), "bursty" clients get
    # trades clustered around a handful of activity windows (high CV) -- this
    # is what gives build_temporal_features' cv_gap real persona structure to
    # recover instead of measuring pure sampling noise.
    for c in range(total_clients):
        positions = np.where(client_id == c)[0]
        k = positions.size
        if k == 0:
            continue
        p = params_of(c)
        if p["burstiness"] == "steady":
            centers = np.linspace(0, window_days, k, endpoint=False)
            jitter = rng.normal(0, window_days / (4 * k), size=k)
            offsets = np.clip(centers + jitter, 0, window_days)
        else:  # bursty
            n_bursts = max(1, k // 5)
            burst_centers = rng.uniform(0, window_days, size=n_bursts)
            offsets = np.clip(burst_centers[rng.integers(0, n_bursts, size=k)]
                              + rng.normal(0, 2.0, size=k), 0, window_days)
        trade_date[positions] = (window_start
                                 + pd.to_timedelta(offsets, unit="D")).values

    # Synthetic AUM per client, correlated with (but noisier and larger than)
    # the client's typical trade size -- lets build_intensity_features'
    # optional AUM-normalized turnover ratio be demonstrated downstream.
    aum_mu = np.array([params_of(c)["amount"][0] + 2.0 for c in range(total_clients)])
    client_aum = pd.Series(rng.lognormal(aum_mu, 0.8), index=client_persona.index,
                           name="aum_usd")

    trades = pd.DataFrame({"client_id": client_id, "amount_usd": amount_usd,
                           "trade_date": trade_date, "side": side, **trade_dims})

    return trades, dimensions, client_persona, client_aum, persona_names
