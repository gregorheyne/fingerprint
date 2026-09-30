from __future__ import annotations
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Dimensionality reduction
#
# PCA only rotates/decorrelates the fingerprint and drops low-variance axes --
# it does not itself group clients. Clustering the PCA scores (see
# clustering.py) is the standard way to check whether the retained variance
# actually corresponds to discrete groups rather than a smooth continuum, and
# it's cheaper/more stable than clustering the full high-dimensional CLR
# space directly.
# ---------------------------------------------------------------------------

def pca_reduce(fp: pd.DataFrame, n_components: int = 2,
               standardize: bool = True, random_state: int = 0) -> tuple[pd.DataFrame, PCA]:
    """PCA on the fingerprint matrix (client x feature -> client x PC).

    Standardizing to unit variance is on by default: the count and gap blocks
    can differ in scale, and without standardizing PCA would rank axes partly
    by which block happens to have larger CLR spread rather than by which
    axis actually separates clients.

    Returns (scores, fitted_model) -- the model exposes
    `explained_variance_ratio_` so callers can judge how much of the
    fingerprint's variance the retained components capture.
    """
    X = StandardScaler().fit_transform(fp.values) if standardize else fp.values
    model = PCA(n_components=n_components, random_state=random_state)
    scores = model.fit_transform(X)
    cols = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(scores, index=fp.index, columns=cols), model


def pca_loadings(pca_model: PCA, feature_names: pd.Index) -> pd.DataFrame:
    """Each PC's loadings on the original (standardized) fingerprint features.

    A PC is by construction a linear combination of the input features, and
    `PCA.components_` holds exactly those combination weights (one row per
    PC, one column per feature). This just transposes that into the more
    readable `feature x PC` orientation, matching the columns `pca_reduce`
    fit on -- pass its input's `.columns`.

    A loading's sign/magnitude says how much moving along that PC shifts a
    client's value on that (standardized) feature; it says nothing about
    where any individual client actually falls, so pair with
    `top_pca_loadings` to see each PC's dominant features, and with
    `cluster_feature_profile` (cluster_explainability.py) to see what
    actually distinguishes a cluster.
    """
    cols = [f"PC{i + 1}" for i in range(pca_model.components_.shape[0])]
    return pd.DataFrame(pca_model.components_.T, index=feature_names, columns=cols)


def top_pca_loadings(loadings: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """The `n` largest-magnitude feature loadings per PC, long-format.

    `pca_loadings` returns every feature x every PC, which is unwieldy to
    read once the fingerprint has more than a handful of features -- most
    entries are near zero and not worth reporting. This ranks by |loading|
    within each PC and keeps only the top `n`, e.g. to answer "PC1 is mostly
    driven by which features, and in which direction?"

    Returns columns [pc, feature, loading], sorted by PC then descending
    |loading|.
    """
    rows = []
    for pc in loadings.columns:
        top = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index[:n])
        for feature, loading in top.items():
            rows.append({"pc": pc, "feature": feature, "loading": loading})
    return pd.DataFrame(rows)


def umap_reduce(fp: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15,
                 min_dist: float = 0.1, random_state: int = 0) -> pd.DataFrame:
    """Nonlinear embedding via UMAP -- good for 2D visualization and as a
    pre-clustering step (see clustering.py's module note on pairing it with
    HDBSCAN). Requires `pip install umap-learn`; imported lazily here so the
    rest of this module doesn't need it installed.
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap_reduce requires umap-learn: pip install umap-learn") from e
    model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                       min_dist=min_dist, random_state=random_state)
    embedding = model.fit_transform(fp.values)
    cols = [f"UMAP{i + 1}" for i in range(n_components)]
    return pd.DataFrame(embedding, index=fp.index, columns=cols)
