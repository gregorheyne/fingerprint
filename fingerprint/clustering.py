from __future__ import annotations
import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_samples


# ---------------------------------------------------------------------------
# Clustering
#
# HDBSCAN, typically paired with the UMAP embedding from embedding.py rather
# than raw or PCA'd features -- see its docstring for why.
#
# Uses the standalone `hdbscan` package rather than sklearn's HDBSCAN: only
# the standalone package computes `relative_validity_` (an internal,
# MST-based DBCV-style validity index) as a byproduct of fitting, which
# sklearn's implementation doesn't expose at all. Note this means cluster
# assignments can shift slightly vs. sklearn's HDBSCAN even with identical
# parameters, since it's a different (the original) implementation.
# ---------------------------------------------------------------------------

def hdbscan_cluster(X: pd.DataFrame, min_cluster_size: int = 5,
                    min_samples: int | None = None) -> tuple[pd.Series, float | None]:
    """HDBSCAN density-based clustering. Meant to be run on a UMAP embedding
    rather than raw or PCA'd CLR features: "density" is a much more
    meaningful notion of neighborhood in a handful of nonlinear embedding
    dimensions than in high-dimensional, near-collinear CLR space, and this
    UMAP-then-HDBSCAN pairing is the de facto standard for this style of
    cluster exploration.

    Strengths: does not need the number of clusters fixed up front -- it
    infers cluster count from density structure; handles arbitrarily shaped
    (non-convex) clusters that k-means/GMM cannot; explicitly labels
    ambiguous, mixed-persona clients as noise (label -1) instead of forcing
    them into the nearest cluster, which is often the more honest answer for
    real trading behavior that blends personas.
    Weaknesses: `min_cluster_size` needs tuning to the data -- too small
    fragments one persona into several clusters, too large merges distinct
    personas or calls everything noise; can leave a nontrivial fraction of
    clients unlabeled, useful for exploration but inconvenient if every
    client needs an assignment downstream; noise points make evaluation
    metrics like silhouette score require care (must exclude label -1).

    `min_samples` is a second, separate knob from `min_cluster_size` and is
    easy to conflate with it: `min_cluster_size` is the minimum size a group
    of core points must reach to be reported as a cluster at all, while
    `min_samples` controls how many neighbors a point itself needs to count
    as "core" -- i.e. how strict the density estimate is. scikit-learn's
    default, when `min_samples` is left as `None` (as here), is
    `min_samples = min_cluster_size` -- the most LENIENT setting available,
    since it takes the fewest neighbors for a point to be treated as core.
    That leniency is exactly what let one-off/outlier clients get folded
    into the nearest real persona cluster as border members instead of
    being labeled noise (-1): with few nearby points required, a client that
    merely leans toward one persona's neighborhood in the UMAP embedding is
    enough to qualify. Passing `min_samples` explicitly, well above
    `min_cluster_size`, makes HDBSCAN demand a denser neighborhood before
    calling anything core, which makes it far more willing to call sparse,
    one-off points noise -- at the cost of also pushing some genuinely
    boundary-region persona clients into noise too, so it's a tradeoff to
    tune, not a strictly-better setting.

    Returns `(labels, relative_validity)`. `labels` are renumbered by
    descending cluster size -- `0` is always the largest cluster, `-1` is
    always noise -- so cluster IDs are meaningfully ordered instead of
    reflecting HDBSCAN's arbitrary internal discovery order; downstream
    consumers only ever use labels as opaque group keys (groupby/crosstab/
    boolean masks), never as positional indices, so this renumbering is
    safe. Ties in size are broken by the original label, ascending, for
    determinism. `relative_validity` is `None` when fewer than 2 clusters
    are found (mirrors the silhouette guard elsewhere).
    """
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            gen_min_span_tree=True)
    raw_labels = model.fit_predict(X.values)
    labels = pd.Series(raw_labels, index=X.index, name="cluster")

    counts = labels[labels != -1].value_counts()  # index=old label, values=size
    # descending by size, ties broken by the original label ascending (for determinism)
    order = counts.index.values[np.lexsort((counts.index.values, -counts.values))]
    remap = {old: new for new, old in enumerate(order)}
    remap[-1] = -1
    labels = labels.map(remap)

    relative_validity = model.relative_validity_ if counts.size >= 2 else None
    return labels, relative_validity


def cluster_silhouette(embedding: pd.DataFrame, labels: pd.Series, cluster_id: int,
                       exclude_noise: bool = True) -> pd.Series:
    """Per-member silhouette score for ONE cluster, in the space HDBSCAN
    actually clustered on -- a geometric answer to "how cohesive/well
    separated is this cluster" that, unlike everything in
    `cluster_explainability.py`, deliberately does NOT take `fp` (original
    features). Silhouette is a statement about the embedding's geometry
    (distance to own cluster vs. nearest other cluster), not about which
    original features drove it -- that's what `cluster_explainability.py`'s
    functions are for.

    `embedding` -- named to match `fingerprint.py`'s
    `_hdbscan_report(fp, embedding, ...)` -- is the higher-dimensional
    embedding HDBSCAN was fit on (e.g. `umap_embedding`), NOT the original `fp`
    and NOT the separate 2D embedding used only for plotting. There's no
    reliable runtime way to tell `fp` and an embedding apart (both are plain
    client x float DataFrames), so this relies on the parameter name alone,
    same as the existing `_hdbscan_report` call site -- pass the wrong one
    and you'll get a silently-wrong-but-plausible score, not an error.

    A silhouette score is inherently relative to ALL clusters at once (each
    point's score depends on its distance to its own cluster AND to the
    nearest *other* cluster), so this still runs `silhouette_samples` over
    every kept cluster -- it just returns only `cluster_id`'s slice
    afterward, rather than computing something cluster-local from scratch.

    Returns the raw per-member scores as a `Series` (not a pre-aggregated
    number) so the caller can take `.mean()`, `.median()`, `.describe()`, or
    plot a histogram, without this function needing an opinion about which
    summary matters. Since `sklearn.metrics.silhouette_score` is defined as
    the mean of `silhouette_samples` over all points, the size-weighted
    average of this function's `.mean()` across every real cluster must
    equal the aggregate `silhouette_score` computed elsewhere (e.g.
    `fingerprint.py:_hdbscan_report`'s `metrics["silhouette"]`) exactly --
    a useful invariant to check this against.

    `cluster_id=-1` (noise) is always rejected, even with
    `exclude_noise=False` -- noise is not a cluster to score.
    `exclude_noise=True` (default) drops label -1 before scoring, so "the
    nearest other cluster" never means "the noise points"; `exclude_noise=False`
    keeps noise in as just another label silhouette can be computed against.
    """
    if cluster_id == -1:
        raise ValueError("cluster_id=-1 is the noise label, not a cluster to score")
    embedding = embedding.reindex(labels.index)
    if exclude_noise:
        keep = labels != -1
        embedding, labels = embedding[keep], labels[keep]
    if cluster_id not in set(labels.unique()):
        raise ValueError(f"unknown cluster_id: {cluster_id!r} -- not present in labels "
                         f"(after exclude_noise filtering)")

    scores = silhouette_samples(embedding.values, labels.values)
    scores = pd.Series(scores, index=embedding.index, name="silhouette")
    return scores[labels == cluster_id]
