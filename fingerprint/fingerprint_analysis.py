from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.tree import export_text

from fingerprint.config import AnalysisConfig, HDBSCANConfig
from fingerprint.embedding import pca_reduce, umap_reduce, pca_loadings, top_pca_loadings
from fingerprint.clustering import hdbscan_cluster, cluster_silhouette
from fingerprint.cluster_explainability import (
    cluster_feature_profile, profile_feature_spread, cluster_anova_importance,
    cluster_surrogate_importance, cluster_cohesion_profile, cluster_exemplars,
    cluster_vs_rest_importance,
)
from fingerprint.visualization import plot_cluster_results, plot_decision_tree_with_importances


# ---------------------------------------------------------------------------
# Full analysis pipeline: fingerprint -> PCA/UMAP -> clustering -> reporting
# ---------------------------------------------------------------------------

def _single_cluster_reports(fp: pd.DataFrame, embedding: pd.DataFrame, clusters: pd.Series,
                            label: str, report_first_n: int = 10,
                            results_dir: Path | None = None,
                            fp_raw: pd.DataFrame | None = None,
                            raw_composition_summary: bool = True) -> None:
    """Deep-dive report for each cluster, up to `report_first_n` of them --
    demonstrates the single-cluster functions in
    `cluster_explainability.py`/`clustering.py`, which the all-clusters-at-once
    report above (`cluster_feature_profile`/`cluster_surrogate_importance`
    etc.) can't answer on their own: one-vs-rest surrogate tree rules for
    each cluster (and, when `results_dir` is given, its PDF rendering).

    `report_first_n` (default 10) caps how many clusters get a deep dive --
    cluster ids are already numbered by descending size (see
    `clustering.hdbscan_cluster`), so this covers the `report_first_n`
    biggest clusters, i.e. cluster ids `0` through `report_first_n - 1`.

    `results_dir` (default `None` = don't save): when given, also renders
    each cluster's one-vs-rest surrogate tree + feature importances to a
    single PDF there (one page per cluster) via
    `plot_decision_tree_with_importances`.

    `fp_raw` (default `None`): when given, the PDF's split labels render in
    business-legible raw units instead of `fp`'s normalized ones -- see
    `visualization.plot_decision_tree`'s docstring, and `raw_composition_summary`
    to toggle just the composition/gap splits between an empirical raw-value
    summary (default) and today's technical CLR threshold. `fp_raw` is
    reindexed/noise-filtered here the same way `cluster_vs_rest_importance`
    filters `fp` internally, so the two line up sample-for-sample with what
    each tree was actually fit on.
    """
    cluster_ids = sorted(clusters[clusters != -1].unique())[:report_first_n]

    fp_raw_fit = None
    if fp_raw is not None:
        fp_raw_fit = fp_raw.reindex(clusters.index)
        fp_raw_fit = fp_raw_fit[clusters != -1]
    fp_fit = fp.reindex(clusters.index)
    fp_fit = fp_fit[clusters != -1]

    pdf_canvas = None
    fname = None
    if results_dir is not None:
        from reportlab.pdfgen import canvas as pdfcanvas
        fname = results_dir / f"cluster_surrogate_trees_{label.lower().replace(' + ', '_')}.pdf"
        pdf_canvas = pdfcanvas.Canvas(str(fname))

    for cluster_id in cluster_ids:
        print(f"\n=== single-cluster deep dive: cluster {cluster_id} ({label}) ===")

        # cohesion = cluster_cohesion_profile(fp, clusters, cluster_id)
        # print(f"\ncohesion profile (mean_z vs population, dispersion_ratio; "
        #      f"top 8 by |mean_z|, cluster {cluster_id}, {label}):")
        # print(cohesion.head(8).round(3))

        # _centroid, exemplars = cluster_exemplars(fp, clusters, cluster_id, n_closest=5, n_farthest=3)
        # print(f"\nexemplar members -- closest/medoid + farthest by distance to centroid "
        #      f"(cluster {cluster_id}, {label}):")
        # print(exemplars.round(2))

        importances, tree = cluster_vs_rest_importance(fp, clusters, cluster_id, method="tree")
        print(f"\none-vs-rest surrogate tree importances -- cluster {cluster_id} vs. rest ({label}):")
        print(importances.head(8))
        class_names = ["rest", f"cluster {cluster_id}"]
        print(export_text(tree, feature_names=list(fp.columns), class_names=class_names, show_weights=True))
        if pdf_canvas is not None:
            plot_decision_tree_with_importances(tree, feature_names=list(fp.columns), class_names=class_names,
                                                importances=importances, highlight_classes=class_names,
                                                fp=fp_fit, fp_raw=fp_raw_fit,
                                                raw_composition_summary=raw_composition_summary,
                                                title=f'Cluster {cluster_id}', canvas=pdf_canvas)

        # sil = cluster_silhouette(embedding, clusters, cluster_id)
        # print(f"silhouette within cluster {cluster_id} ({label}): "
        #      f"mean={sil.mean():.3f}, median={sil.median():.3f}, min={sil.min():.3f}")

    if pdf_canvas is not None:
        pdf_canvas.save()
        print(f"wrote {fname}")


def _hdbscan_report(fp: pd.DataFrame, embedding: pd.DataFrame, label: str,
                    persona: pd.Series | None,
                    hdbscan_config: HDBSCANConfig,
                    results_dir: Path | None = None,
                    fp_raw: pd.DataFrame | None = None,
                    raw_composition_summary: bool = True) -> tuple[pd.Series, dict]:
    """Run HDBSCAN on `embedding` and print its cluster diagnostics:
    found count/noise count/noise fraction, biggest-cluster sizes,
    silhouette (noise excluded), HDBSCAN's own relative validity index,
    persona crosstab (if `persona` given), feature profile, feature-spread
    ranking, ANOVA importance, surrogate-tree importance, and single-cluster
    deep dives (see `_single_cluster_reports`) on the biggest clusters found.
    Shared between the PCA and UMAP variants below since both go through the
    same reporting.

    `embedding` is deliberately not the 2D embedding used for
    plotting -- see `run_fingerprint_analysis`'s docstring for why
    clustering and visualization use separate embeddings.

    Returns `(clusters, metrics)`, `metrics` being `{"n_clusters", "n_noise",
    "noise_fraction", "silhouette", "relative_validity"}` -- the same
    numbers that get printed, but as data so a sweep across many configs
    (see `run_sweep`) has something to compare.

    `hdbscan_config.min_samples` defaults to 3x `min_cluster_size` rather
    than the lenient `min_samples=min_cluster_size` HDBSCAN default -- see
    `hdbscan_cluster`'s docstring for why the default folds outlier clients
    into the nearest persona cluster instead of flagging them noise.

    `fp_raw`/`raw_composition_summary` are forwarded to `_single_cluster_reports`
    for business-legible surrogate tree PDFs -- see its docstring.
    """
    clusters, relative_validity = hdbscan_cluster(
        embedding, min_cluster_size=hdbscan_config.min_cluster_size,
        min_samples=hdbscan_config.min_samples)
    n_found = clusters[clusters != -1].nunique()
    n_total = len(clusters)
    n_noise = int((clusters == -1).sum())
    noise_fraction = n_noise / n_total
    print(f"{label} found {n_found} clusters "
         f"({n_noise} of {n_total} clients labeled noise, {noise_fraction:.1%})")
    sizes = clusters[clusters != -1].value_counts().sort_values(ascending=False)
    print(f"\n10 biggest clusters by size ({label}):")
    print(sizes.head(10))
    silhouette = None
    if n_found >= 2:
        not_noise = clusters != -1
        silhouette = silhouette_score(embedding[not_noise], clusters[not_noise])
        print(f"silhouette ({label}, noise excluded, in clustering space):", silhouette)
        print(f"HDBSCAN relative validity (~DBCV, {label}):", relative_validity)
    if persona is not None:
        print(pd.crosstab(clusters, persona))
    if n_found >= 2:
        print(f"\ncluster feature profile (z-score vs population, {label}):")
        profile = cluster_feature_profile(fp, clusters)
        print(profile.round(2))
        print(f"\nfeatures ranked by cluster-mean spread ({label}):")
        print(profile_feature_spread(profile).round(2).head(8))
        print(f"\nfeatures ranked by ANOVA F-statistic ({label}):")
        print(cluster_anova_importance(fp, clusters).round(2).head(8))

        importances, tree = cluster_surrogate_importance(fp, clusters, method="tree")
        print(f"\nsurrogate decision tree feature importances ({label}):")
        print(importances.head(8))

        _single_cluster_reports(fp, embedding, clusters, label, results_dir=results_dir,
                                fp_raw=fp_raw, raw_composition_summary=raw_composition_summary)
    metrics = {"n_clusters": n_found, "n_noise": n_noise, "noise_fraction": noise_fraction,
              "silhouette": silhouette, "relative_validity": relative_validity}
    return clusters, metrics


def run_fingerprint_analysis(fp: pd.DataFrame,
                             fp_raw: pd.DataFrame | None = None,
                             client_persona: pd.Series | None = None,
                             client_aum: pd.Series | None = None,
                             config: AnalysisConfig | None = None,
                             save_plots: bool = True,
                             raw_composition_summary: bool = True) -> dict:
    """Run the full embedding -> clustering -> reporting pipeline on a
    pre-built fingerprint `fp` (see `fingerprint.build_fingerprint`) and
    return the per-method results and metrics.

    `config` (an `AnalysisConfig` from `config.py`) holds every PCA/UMAP/
    HDBSCAN hyperparameter; defaults to `AnalysisConfig()` if not given. Load
    one from YAML with `load_config(path)`, or generate many with
    `sweep_configs(base, ...)` to compare across a parameter grid via
    `run_sweep`.

    `save_plots=False` skips writing PNGs to `results/` -- useful when this
    is called many times over a sweep, since today's filenames aren't unique
    per config.

    Clustering and visualization are deliberately disentangled, for both
    PCA and UMAP: squeezing everything down to 2D is great for plotting but
    throws away structure a higher-dimensional embedding preserves, and
    since clustering doesn't need to be *seen*, there's no reason to
    bottleneck HDBSCAN at 2 dimensions. So each method fits an embedding
    with `config.{pca,umap}.n_components` dimensions, HDBSCAN runs on that,
    and a *separate* 2D embedding -- used only for the scatter plots -- is
    colored by the cluster labels the higher-dimensional run found:
      - PCA: components are nested (PC1/PC2 of a 10-component fit are
        identical to a standalone 2-component fit), so the 2D plot just
        reuses the first two columns of the higher-dimensional scores --
        no second fit needed.
      - UMAP: components are NOT nested (it's a nonlinear, stochastic
        embedding, so a 2D UMAP is a different fit, not a projection of a
        10D one), so a second `umap_reduce(..., n_components=2)` call is
        required purely for the plot.

    Clustering is HDBSCAN throughout -- see `clustering.hdbscan_cluster`'s
    docstring for why it's normally paired with UMAP. Unlike k-means/GMM,
    HDBSCAN infers its own cluster count, so no `n_clusters` is needed.

    The PCA + HDBSCAN branch is currently disabled below (commented out,
    not deleted) -- `embedding.pca_reduce`/`pca_loadings`/`top_pca_loadings`
    and `clustering.hdbscan_cluster` are unaffected and still usable; only
    the call to run PCA + HDBSCAN as part of this pipeline is switched off.

    `client_persona` and `client_aum` are ground-truth / enrichment data
    only available for synthetic data -- both optional so this can run on
    real data without them. `client_persona` (if given) enables crosstabs
    against known personas and aligned cluster coloring in the plots;
    `client_aum` (if given) is used to produce revenue-bucket-colored plots
    (AUM-normalizing intensity features, if wanted, happens earlier when
    `fp` is built via `build_fingerprint`).

    `fp_raw` (default `None`, the `FingerprintResult.fp_raw` companion from
    the same `build_fingerprint` call): when given, surrogate-tree PDFs
    render their split labels in business-legible raw units instead of
    `fp`'s normalized ones -- see `_single_cluster_reports`/
    `visualization.plot_decision_tree`. `raw_composition_summary` (default
    `True`) toggles just the composition/gap splits between an empirical
    raw-value summary and today's technical CLR-threshold display, e.g. to
    compare the two by calling this twice with different values.
    """
    config = config or AnalysisConfig()

    print(fp.shape)      # (n_clients, n_features)
    # print(fp.head())

    persona = client_persona.reindex(fp.index) if client_persona is not None else None

    results: dict = {}
    metrics: dict = {}

    results_dir = None
    if save_plots:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

    # PCA + HDBSCAN branch temporarily disabled -- see docstring above.
    # `embedding.pca_reduce`/`clustering.hdbscan_cluster` are untouched and
    # still fully usable; only this call is switched off.
    # n_pca = min(config.pca.n_components, fp.shape[1], fp.shape[0] - 1)
    # pca_scores, pca_model = pca_reduce(fp, n_components=n_pca, standardize=config.pca.standardize,
    #                                    random_state=config.pca.random_state)
    # print(f"explained variance ratio ({n_pca} PCs used for clustering):",
    #      pca_model.explained_variance_ratio_)
    # print("\ntop PCA loadings (what each PC is a linear combination of):")
    # print(top_pca_loadings(pca_loadings(pca_model, fp.columns), n=5))
    #
    # pca_clusters, pca_metrics = _hdbscan_report(fp, pca_scores, "PCA + HDBSCAN", persona, config.hdbscan)
    # # 2D plotting view: the first two columns of the same (nested) PCA fit,
    # # not a separate reduction -- see docstring above.
    # pca_scores_2d = pca_scores.iloc[:, :2]
    # results["PCA + HDBSCAN"] = (pca_scores_2d, pca_clusters)
    # metrics["PCA + HDBSCAN"] = pca_metrics

    n_umap = min(config.umap.n_components, fp.shape[1])
    umap_embedding = umap_reduce(fp, n_components=n_umap, n_neighbors=config.umap.n_neighbors,
                                min_dist=config.umap.min_dist, random_state=config.umap.random_state)
    umap_clusters, umap_metrics = _hdbscan_report(fp, umap_embedding, "UMAP + HDBSCAN", persona,
                                                    config.hdbscan, results_dir=results_dir,
                                                    fp_raw=fp_raw,
                                                    raw_composition_summary=raw_composition_summary)
    # Separate 2D UMAP fit purely for plotting -- UMAP isn't nested like
    # PCA, so this is a different embedding, not a projection of the
    # `n_umap`-dimensional one clustering ran on. Cluster *labels* still
    # come from the higher-dimensional run above.
    umap_embedding_2d = umap_reduce(fp, n_components=2, n_neighbors=config.umap.n_neighbors,
                                    min_dist=config.umap.min_dist, random_state=config.umap.random_state)
    results["UMAP + HDBSCAN"] = (umap_embedding_2d, umap_clusters)
    metrics["UMAP + HDBSCAN"] = umap_metrics

    if save_plots:
        # One file per (embedding, clustering) result rather than one combined
        # small-multiples figure, so each variant can be viewed/shared on its own.
        for title, (embedding, clusters) in results.items():
            fname = results_dir / f"cluster_results_{title.lower().replace(' + ', '_')}.png"
            plot_cluster_results({title: (embedding, clusters)}, true_labels=persona,
                                 savepath=fname)
            print(f"wrote {fname}")

        # Same cluster layouts, but colored by revenue (AUM) bucket instead of
        # persona -- illustrates `color_by` for checking whether an unrelated
        # client dimension is smeared across clusters or concentrated in some.
        if client_aum is not None:
            revenue_bucket = pd.qcut(client_aum.reindex(fp.index), q=4,
                                     labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
            for title, (embedding, clusters) in results.items():
                fname = results_dir / f"cluster_results_{title.lower().replace(' + ', '_')}_by_revenue.png"
                plot_cluster_results({title: (embedding, clusters)}, color_by=revenue_bucket,
                                     savepath=fname)
                print(f"wrote {fname}")

    return {"results": results, "metrics": metrics}


def run_sweep(fp: pd.DataFrame, configs: Iterable[AnalysisConfig],
             client_persona: pd.Series | None = None,
             client_aum: pd.Series | None = None) -> pd.DataFrame:
    """Run `run_fingerprint_analysis` once per config in `configs` (e.g. from
    `sweep_configs`) on the same pre-built fingerprint `fp`, and return a
    tidy `pd.DataFrame`, one row per (config, method), with the flattened
    config fields (`pca_n_components`, `umap_n_neighbors`,
    `hdbscan_min_cluster_size`, etc.) alongside that run's
    `n_clusters`/`n_noise`/`silhouette` -- sortable/plottable to see how a
    swept parameter moves cluster quality. Plots are skipped (`save_plots=
    False`) since today's plot filenames aren't unique per config.
    """
    rows = []
    for config in configs:
        flat_config = {
            f"{section}_{key}": value
            for section, section_config in asdict(config).items()
            for key, value in section_config.items()
        }
        outcome = run_fingerprint_analysis(fp, client_persona=client_persona,
                                           client_aum=client_aum, config=config, save_plots=False)
        for method, method_metrics in outcome["metrics"].items():
            rows.append({**flat_config, "method": method, **method_metrics})
    return pd.DataFrame(rows)
