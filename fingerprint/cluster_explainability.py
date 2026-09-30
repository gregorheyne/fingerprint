from __future__ import annotations
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif


def cluster_feature_profile(fp: pd.DataFrame, labels: pd.Series,
                            exclude_noise: bool = True) -> pd.DataFrame:
    """Per-cluster mean of each ORIGINAL fingerprint feature, as a z-score
    against the whole client population -- the model-agnostic way to
    explain a cluster in terms of the features that actually produced it.

    Unlike `pca_loadings`/`top_pca_loadings` (embedding.py, which only
    explain PCA's own axes), this operates on `fp` directly, so it works
    identically whether `labels` came from PCA+k-means, PCA+GMM, or
    UMAP+HDBSCAN -- UMAP has no linear "loadings" of its own (see
    `cluster_surrogate_importance` for why and what to use instead), but
    "what do this cluster's members look like on the original features" is
    always a well-posed question.

    Z-scoring (population mean/std, not per-cluster) makes features of very
    different scales comparable in one table: an entry of +1.5 means that
    cluster's average client sits 1.5 population standard deviations above
    the overall mean on that feature; -0.3 means slightly below average.

    `exclude_noise=True` (default) drops HDBSCAN's -1 label before
    computing cluster means -- noise is explicitly "didn't fit a cluster",
    not a group whose average is meaningful.

    Returns a `cluster x feature` DataFrame of z-scores.
    """
    fp = fp.reindex(labels.index)
    if exclude_noise:
        keep = labels != -1
        fp, labels = fp[keep], labels[keep]
    std = fp.std(ddof=0).replace(0, 1)  # constant features would otherwise divide by zero
    z = (fp - fp.mean()) / std
    return z.groupby(labels).mean()


def profile_feature_spread(profile: pd.DataFrame, method: str = "range") -> pd.Series:
    """Rank fingerprint features by how much their `cluster_feature_profile`
    z-scores diverge across clusters -- i.e. which features actually
    separate the clusters, at a glance, without fitting a classifier.

    `cluster_feature_profile` returns a `cluster x feature` table of
    z-scored means; a feature where every cluster sits near 0 is one none
    of them differ on, while one where clusters sit at e.g. +1.8 and -1.5
    clearly splits them. This collapses that table's cluster axis into a
    single per-feature spread score and sorts descending, so the top rows
    are the features most worth quoting when describing what separates the
    clusters.

    method="range" (default): max - min across cluster means. Simple and
    robust to cluster count, but anchored only on the two most extreme
    clusters -- ignores where the rest sit.
    method="std": population std across cluster means. Sensitive to how
    spread out ALL cluster means are relative to each other, not just the
    extremes, but pulled toward 0 if most clusters agree and only one or
    two are outliers.

    This only measures BETWEEN-cluster mean spread -- unlike an ANOVA
    F-statistic, it does not discount for WITHIN-cluster variance, so it
    can rank a feature highly even where the clusters heavily overlap on
    it. Treat it as a fast first pass, and use `cluster_surrogate_importance`
    for a model-based, interaction-aware view of the same question.
    """
    if method == "range":
        spread = profile.max() - profile.min()
    elif method == "std":
        spread = profile.std(ddof=0)
    else:
        raise ValueError(f"unknown method: {method!r} -- expected 'range' or 'std'")
    return spread.sort_values(ascending=False)


def cluster_anova_importance(fp: pd.DataFrame, labels: pd.Series,
                             exclude_noise: bool = True) -> pd.DataFrame:
    """Per-feature one-way ANOVA F-statistic for how well each original
    fingerprint feature separates the cluster labels.

    `profile_feature_spread` ranks features by between-cluster mean spread
    alone (the F-statistic's numerator) -- it has no way to tell a feature
    with widely-spread cluster means but also huge within-cluster scatter
    (means could differ by chance) from one with the same spread but tight,
    consistent clusters (means differ for real). The F-statistic is exactly
    that ratio:

        F = (between-cluster variance) / (within-cluster variance)

    so a large F means the clusters are genuinely separated on that feature
    relative to their own internal noise, not just spread out on paper.
    Uses `sklearn.feature_selection.f_classif`, which computes this
    per-feature in closed form (no model fit needed).

    Like `profile_feature_spread`, this is still a per-feature (marginal)
    view -- it does not account for how features jointly/interactively
    separate clusters the way `cluster_surrogate_importance`'s classifier
    does. It also carries classical ANOVA's assumptions (roughly normal,
    equal-variance-across-clusters feature distributions), which CLR/log
    fingerprint features won't satisfy exactly, so treat F as a ranking
    heuristic rather than a rigorous hypothesis test -- and be wary of the
    accompanying p-values in particular, since testing many features at
    once inflates the odds of a low p-value by chance alone.

    `exclude_noise=True` (default) drops HDBSCAN's -1 label first --
    noise isn't a group to test against the others.

    Returns a DataFrame indexed by feature, columns [F, p_value], sorted by
    F descending.
    """
    fp = fp.reindex(labels.index)
    if exclude_noise:
        keep = labels != -1
        fp, labels = fp[keep], labels[keep]
    f_stat, p_value = f_classif(fp.values, labels.values)
    result = pd.DataFrame({"F": f_stat, "p_value": p_value}, index=fp.columns)
    return result.sort_values("F", ascending=False)


def cluster_surrogate_importance(fp: pd.DataFrame, labels: pd.Series,
                                 method: str = "tree", exclude_noise: bool = True,
                                 max_depth: int = 3, random_state: int = 0):
    """Which original fingerprint features best predict cluster membership,
    via a small interpretable classifier trained on (features -> cluster
    label) -- the practical substitute for "loadings" when the embedding
    that produced `labels` is nonlinear (UMAP), where "what is UMAP axis 1
    made of" is not even a well-posed question the way it is for PCA.

    Both `method` options are deliberately shallow: the goal is a
    human-readable explanation of what separates the clusters, not a
    high-accuracy model, and an unconstrained classifier would fit the
    labels too well to explain them.

    method="tree" (default): `DecisionTreeClassifier(max_depth=max_depth)`.
    Returns (importances, model) where importances is a Series of Gini-based
    `feature_importances_` sorted descending, and model is the fitted tree
    -- inspect its actual split rules with
    `sklearn.tree.export_text(model, feature_names=list(fp.columns))` for
    the most literal explanation ("cluster 2 is clients with
    intensity_log > 1.3 and gap_entropy <= 0.4").
    Weakness: only ranks features by predictive usefulness, not direction
    (a feature can be important via either high or low values).

    method="logistic": multinomial L1 `LogisticRegression` on standardized
    features. Returns (importances, model) where importances is a
    `cluster x feature` DataFrame of signed coefficients -- so unlike the
    tree, this also gives direction (positive coefficient: higher values of
    that feature push a client toward that cluster). L1 additionally zeroes
    out features the model finds unnecessary, giving a sparser explanation
    than the tree's importances.
    Weakness: assumes roughly linear separability between clusters in
    feature space, which won't hold as well as the tree does for clusters
    UMAP found precisely because they're nonlinearly separable in the
    original features.

    `exclude_noise=True` (default) drops HDBSCAN's -1 label before fitting
    -- noise isn't a class worth teaching the classifier to predict.
    """
    fp = fp.reindex(labels.index)
    if exclude_noise:
        keep = labels != -1
        fp, labels = fp[keep], labels[keep]

    if method == "tree":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        model.fit(fp.values, labels.values)
        importances = pd.Series(model.feature_importances_, index=fp.columns,
                                name="importance").sort_values(ascending=False)
    elif method == "logistic":
        model = LogisticRegression(penalty="l1", solver="saga", max_iter=5000,
                                   random_state=random_state)
        X = StandardScaler().fit_transform(fp.values)
        model.fit(X, labels.values)
        importances = pd.DataFrame(model.coef_, index=[f"cluster {c}" for c in model.classes_],
                                   columns=fp.columns)
    else:
        raise ValueError(f"unknown method: {method!r} -- expected 'tree' or 'logistic'")
    return importances, model


def cluster_cohesion_profile(fp: pd.DataFrame, labels: pd.Series, cluster_id: int,
                             exclude_noise: bool = True) -> pd.DataFrame:
    """Deep-dive on ONE cluster: per-feature z-score of ITS mean vs. the
    population, alongside a within-cluster dispersion ratio -- the two
    questions "how different is this cluster's average member" and "how
    tightly do its members agree with each other" are easy to conflate but
    answer different things, and a cluster can score high on one and low on
    the other.

    `mean_z` is exactly `cluster_feature_profile(fp, labels, exclude_noise)
    .loc[cluster_id]` (same z-scoring convention, computed directly here
    rather than slicing that table, to stay self-contained like every other
    function in this file) -- how far this cluster's average sits from the
    overall population, in population standard deviations.

    `dispersion_ratio` is new: this cluster's own std on that feature,
    divided by the population's std. Near 0 means members agree tightly on
    that feature -- a genuinely defining trait, even if `mean_z` isn't
    extreme. Near/above 1 means the cluster is about as scattered as the
    whole population on that feature -- its mean may be an artifact of a
    few members rather than something the group shares, even if `mean_z`
    ranks it highly.

    `exclude_noise=True` (default) drops HDBSCAN's -1 label before
    computing the population and the cluster. `cluster_id=-1` itself is
    always rejected, even with `exclude_noise=False` -- noise isn't a
    cohesive group to profile.

    Returns a `feature x [mean_z, dispersion_ratio]` DataFrame sorted by
    `mean_z`'s magnitude descending.
    """
    if cluster_id == -1:
        raise ValueError("cluster_id=-1 is the noise label, not a cluster to profile")
    fp = fp.reindex(labels.index)
    if exclude_noise:
        keep = labels != -1
        fp, labels = fp[keep], labels[keep]
    if cluster_id not in set(labels.unique()):
        raise ValueError(f"unknown cluster_id: {cluster_id!r} -- not present in labels "
                         f"(after exclude_noise filtering)")

    members = fp[labels == cluster_id]
    pop_std = fp.std(ddof=0).replace(0, 1)  # constant features would otherwise divide by zero
    mean_z = (members.mean() - fp.mean()) / pop_std
    dispersion_ratio = members.std(ddof=0) / pop_std
    result = pd.DataFrame({"mean_z": mean_z, "dispersion_ratio": dispersion_ratio})
    return result.reindex(result["mean_z"].abs().sort_values(ascending=False).index)


def cluster_exemplars(fp: pd.DataFrame, labels: pd.Series, cluster_id: int,
                      n_closest: int = 5, n_farthest: int = 0,
                      exclude_noise: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    """Deep-dive on ONE cluster via its most representative and most
    atypical real members, for qualitative inspection -- a complement to the
    statistical views above, since "pull up these 5 actual clients and read
    their raw fingerprint" is often the fastest way to build intuition for
    what a cluster actually is.

    Standardizes `fp` the same way as `cluster_feature_profile` (population
    z-score, so features of different scales are comparable in one distance
    metric), then takes `centroid` = the cluster's mean position in that
    standardized space -- itself identical to
    `cluster_feature_profile(fp, labels, exclude_noise).loc[cluster_id]`, a
    free cross-check. Every member's standardized Euclidean distance to
    `centroid` is then ranked: the closest real member is the "medoid" --
    a more inspectable stand-in for the centroid than the centroid itself,
    since it's an actual client rather than an average of one -- the next
    `n_closest - 1` are "closest" (typical members), and the `n_farthest`
    most distant are "farthest" (atypical/boundary members, still inside
    this cluster).

    `n_closest`/`n_farthest` are clipped to the cluster's size; if their sum
    exceeds it, the closest and farthest sets can overlap for small
    clusters -- not specially handled, since avoiding that would need an
    arbitrary tie-breaking rule. `n_closest=0` produces no "medoid" row.

    `exclude_noise=True` (default) drops HDBSCAN's -1 label first.
    `cluster_id=-1` itself is always rejected, even with
    `exclude_noise=False` -- noise isn't a cluster to find exemplars in.

    Returns `(centroid, exemplars)`: `centroid` is a `Series` indexed by
    feature; `exemplars` is a DataFrame indexed by client id with columns
    `[role, rank, distance]`, `role` one of `{"medoid", "closest", "farthest"}`.
    """
    if cluster_id == -1:
        raise ValueError("cluster_id=-1 is the noise label, not a cluster to find exemplars in")
    fp = fp.reindex(labels.index)
    if exclude_noise:
        keep = labels != -1
        fp, labels = fp[keep], labels[keep]
    if cluster_id not in set(labels.unique()):
        raise ValueError(f"unknown cluster_id: {cluster_id!r} -- not present in labels "
                         f"(after exclude_noise filtering)")

    std = fp.std(ddof=0).replace(0, 1)  # constant features would otherwise divide by zero
    z = (fp - fp.mean()) / std
    members_z = z[labels == cluster_id]
    centroid = members_z.mean()
    dist = ((members_z - centroid) ** 2).sum(axis=1) ** 0.5
    dist = dist.sort_values()

    n_closest = min(n_closest, len(dist))
    n_farthest = min(n_farthest, len(dist))
    closest = dist.iloc[:n_closest]
    farthest = dist.iloc[-n_farthest:][::-1] if n_farthest else dist.iloc[0:0]

    rows = []
    for rank, (client_id, d) in enumerate(closest.items(), start=1):
        role = "medoid" if rank == 1 else "closest"
        rows.append((client_id, role, rank, d))
    for rank, (client_id, d) in enumerate(farthest.items(), start=1):
        rows.append((client_id, "farthest", rank, d))

    exemplars = pd.DataFrame(rows, columns=["client_id", "role", "rank", "distance"])
    exemplars = exemplars.set_index("client_id")
    return centroid, exemplars


def cluster_vs_rest_importance(fp: pd.DataFrame, labels: pd.Series, cluster_id: int,
                               method: str = "tree", exclude_noise: bool = True,
                               max_depth: int = 3, random_state: int = 0):
    """Which original fingerprint features best distinguish ONE cluster from
    every other (real) cluster -- the one-vs-rest binary analogue of
    `cluster_surrogate_importance`, for when you've already picked a cluster
    (e.g. the largest) and want its own crisp "what separates this group"
    answer instead of one shared multiclass model's view across all of them.

    Same shallow-classifier philosophy as `cluster_surrogate_importance`:
    the goal is a human-readable explanation, not a high-accuracy model.
    Target is binary -- `y = 1` for `cluster_id`'s members, `y = 0` for
    everyone else kept after `exclude_noise` filtering.

    method="tree" (default): `DecisionTreeClassifier(max_depth=max_depth)`.
    Returns (importances, model) where importances is a Series of Gini-based
    `feature_importances_` sorted descending, and model is the fitted tree
    -- inspect its rules with `sklearn.tree.export_text(model,
    feature_names=list(fp.columns), class_names=["rest", f"cluster
    {cluster_id}"])` for a literal binary rule ("cluster 2 is clients with
    intensity_log > 1.3"). Same directionless weakness as the multiclass
    version: ranks features by usefulness, not which way they push.

    method="logistic": L1 `LogisticRegression` on standardized features.
    Unlike `cluster_surrogate_importance`'s multiclass logistic branch --
    which returns a `cluster x feature` DataFrame, one row per class -- a
    *binary* logistic model has a single coefficient vector (log-odds of
    "in cluster_id" vs. "rest"), so this returns a signed `Series`, sorted
    by magnitude descending, not a DataFrame. This shape difference from
    `cluster_surrogate_importance` is deliberate, not an oversight.

    `exclude_noise=True` (default) drops HDBSCAN's -1 label first, so
    "rest" means "every other real cluster," not "everything that isn't
    cluster_id including ambiguous noise points." `cluster_id=-1` itself is
    always rejected, even with `exclude_noise=False` -- noise isn't a
    cluster to explain.
    """
    if cluster_id == -1:
        raise ValueError("cluster_id=-1 is the noise label, not a cluster to explain")
    fp = fp.reindex(labels.index)
    if exclude_noise:
        keep = labels != -1
        fp, labels = fp[keep], labels[keep]
    if cluster_id not in set(labels.unique()):
        raise ValueError(f"unknown cluster_id: {cluster_id!r} -- not present in labels "
                         f"(after exclude_noise filtering)")

    y = (labels == cluster_id).astype(int)
    if method == "tree":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        model.fit(fp.values, y.values)
        importances = pd.Series(model.feature_importances_, index=fp.columns,
                                name="importance").sort_values(ascending=False)
    elif method == "logistic":
        model = LogisticRegression(penalty="l1", solver="saga", max_iter=5000,
                                   random_state=random_state)
        X = StandardScaler().fit_transform(fp.values)
        model.fit(X, y.values)
        coef = pd.Series(model.coef_[0], index=fp.columns, name="coefficient")
        importances = coef.reindex(coef.abs().sort_values(ascending=False).index)
    else:
        raise ValueError(f"unknown method: {method!r} -- expected 'tree' or 'logistic'")
    return importances, model
