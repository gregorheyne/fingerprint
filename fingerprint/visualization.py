from __future__ import annotations
import io
import itertools
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from fingerprint.fingerprint_builder import raw_threshold


# ---------------------------------------------------------------------------
# Visualization
#
# One small-multiples figure: one panel per (embedding, clustering) result,
# each a 2D scatter colored by cluster assignment. Colors come from a fixed
# qualitative palette assigned in a fixed order -- never cycled, never chosen
# per-plot -- so a given color always denotes the same thing everywhere it
# appears. The first 7 entries are the colorblind-safe Okabe-Ito palette
# (used whenever there are <= 7 clusters); the remaining 13 are a tab20-
# derived extension to keep colors distinct up to 20 clusters.
# ---------------------------------------------------------------------------

_OKABE_ITO = [
    "#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442",
    "#D62728", "#FF9896", "#9467BD", "#C5B0D5", "#8C564B", "#C49C94",
    "#E377C2", "#F7B6D2", "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5", "#AEC7E8",
]
_NOISE_COLOR = "#9E9E9E"  # HDBSCAN label -1: deliberately outside the categorical set
_OTHER_COLOR = "#4D4D4D"  # clusters beyond `max_clusters`, lumped together -- darker than noise


def _cluster_colors(labels: pd.Series, true_labels: pd.Series | None,
                    max_clusters: int | None = None) -> pd.Series:
    """Map each cluster label to a display name + fixed palette color.

    Without ground truth, clusters are just numbered in sorted order (0, 1, ...)
    and colored accordingly -- consistent *within* one panel, but the mapping is
    arbitrary across panels since e.g. PCA+HDBSCAN's "cluster 0" and UMAP+HDBSCAN's
    "cluster 0" need not be the same real-world group.

    With `true_labels` (e.g. known personas), each cluster is instead named and
    colored after its OWN majority ground-truth label -- so the same real-world
    group renders in the same color in every panel of the figure, regardless of
    which algorithm found it or how it happened to number its clusters. This is
    only a plotting aid (majority vote, not a real assignment metric); it can
    mislabel a cluster if two personas are about equally represented in it.

    Noise (-1, from HDBSCAN) is always rendered in a fixed gray outside the
    categorical palette, never as "just another cluster."

    `max_clusters` (only applied when `true_labels` is None -- with
    `true_labels`, coloring is keyed by persona name, not raw cluster id, so
    a cap on cluster *ids* isn't the relevant axis there): clusters are
    already numbered by descending size (see `clustering.hdbscan_cluster`),
    so keeping only the first `max_clusters` ids keeps the biggest ones;
    everything past that is lumped into a single "other clusters" name/color
    instead of being dropped, so no data silently disappears from the plot.
    """
    is_noise = labels == -1
    if true_labels is not None:
        ref = true_labels.reindex(labels.index)
        majority = pd.crosstab(labels[~is_noise], ref[~is_noise]).idxmax(axis=1)
        # fixed color assignment keyed by persona name, not by cluster id, so
        # order is identical across every panel regardless of which personas
        # a given algorithm happened to find first
        all_names = sorted(true_labels.unique())
        color_of = dict(zip(all_names, _OKABE_ITO))
        name = labels.map(majority).astype(object)
        color = name.map(color_of)
    else:
        cluster_ids = sorted(labels[~is_noise].unique())
        if max_clusters is not None:
            kept_ids = set(cluster_ids[:max_clusters])
        else:
            kept_ids = set(cluster_ids)
        color_of = dict(zip([c for c in cluster_ids if c in kept_ids], _OKABE_ITO))
        name = labels.map(lambda c: f"cluster {c}" if c in kept_ids else "other clusters")
        color = labels.map(lambda c: color_of[c] if c in kept_ids else _OTHER_COLOR)

    name = name.where(~is_noise, "noise")
    color = color.where(~is_noise, _NOISE_COLOR)
    return name, color


def _direct_colors(values: pd.Series, palette=_OKABE_ITO) -> tuple[pd.Series, pd.Series]:
    """Map each point directly to its own value's name + color, bypassing
    cluster identity entirely.

    Unlike `_cluster_colors` (which names/colors a whole cluster by its
    majority vote against ground truth), this is for coloring the *same*
    embedding by an unrelated external, per-client dimension -- e.g. revenue
    bucket -- to see how that dimension is distributed across the clusters a
    given (embedding, labels) pair actually found. Every point keeps its own
    `values` entry as both legend name and color key, independent of which
    cluster it landed in.

    Colors cycle through `palette` if `values` has more unique levels than
    the palette provides. Missing values render as "n/a" in the fixed noise
    gray, same convention as HDBSCAN noise in `_cluster_colors`.
    """
    is_na = values.isna()
    all_names = sorted(values[~is_na].unique(), key=str)
    color_of = dict(zip(all_names, itertools.cycle(palette)))
    name = values.astype(object).where(~is_na, "n/a")
    color = values.map(color_of).where(~is_na, _NOISE_COLOR)
    return name, color


def plot_cluster_results(results: dict[str, tuple[pd.DataFrame, pd.Series]],
                         true_labels: pd.Series | None = None,
                         color_by: pd.Series | None = None,
                         ncols: int = 2,
                         panel_size: tuple[float, float] = (4.6, 4.2),
                         max_clusters: int = 15,
                         savepath: str | None = None) -> Figure:
    """Small-multiples scatter of 2D embedding x cluster assignment, one
    panel per entry of `results`.

    `results`: {panel title -> (embedding, labels)}, e.g.
        {"PCA + HDBSCAN":  (pca_scores, pca_hdb_clusters),
         "UMAP + HDBSCAN": (umap_embedding, umap_hdb_clusters)}
    `embedding` must have exactly 2 columns (e.g. from `embedding.pca_reduce`/
    `embedding.umap_reduce`); `labels` a cluster-id Series aligned to it
    (-1 = noise).

    `embedding` and `labels` need not come from the same fit -- e.g.
    `run_fingerprint_analysis` clusters on a higher-dimensional embedding
    and passes a separate 2D embedding here purely for plotting (see its
    docstring). This function only plots; it doesn't care where the labels
    came from. The silhouette score in each panel's title is computed on
    the 2D `embedding` shown, as a visual-separation sanity check -- it is
    NOT the silhouette of the clustering decision itself when that ran in a
    different-dimensional space; that one is reported by the caller
    (`_hdbscan_report` in fingerprint.py) at clustering time.

    `true_labels`: optional client_id -> ground-truth group Series (e.g.
    known personas). When given, cluster colors are aligned across panels
    (see `_cluster_colors`) and silhouette score is computed against the
    true 2D embedding; each panel's title also reports it.

    `color_by`: optional client_id -> label Series for a dimension unrelated
    to the clustering itself (e.g. revenue bucket, region, tenure). When
    given, it takes over point naming/coloring completely (`true_labels` is
    ignored for coloring purposes) -- every point is labeled/colored by its
    own `color_by` value rather than by its cluster's identity or majority
    vote. Use this to check how some other client dimension is distributed
    across clusters found on a different axis, e.g. plotting a PCA/UMAP
    cluster layout but coloring each point by revenue bucket. The panel
    title's cluster stats (k, noise count, silhouette) still describe the
    actual `labels`/embedding passed in -- `color_by` only changes what
    generates each point's legend name/color.

    `max_clusters`: only the `max_clusters` biggest clusters (by the
    size-descending numbering `clustering.hdbscan_cluster` already assigns)
    are drawn with their own color/legend entry; the rest are lumped into a
    single "other clusters" entry (see `_cluster_colors`). Each panel's
    title notes when it's been truncated this way.

    Returns the Figure; pass `savepath` to also write it to disk.
    """
    from sklearn.metrics import silhouette_score

    n = len(results)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False)

    for ax, (title, (embedding, labels)) in zip(axes.flat, results.items()):
        x_col, y_col = embedding.columns[:2]
        if color_by is not None:
            name, color = _direct_colors(color_by.reindex(embedding.index))
        else:
            name, color = _cluster_colors(labels, true_labels, max_clusters=max_clusters)

        not_noise = labels != -1
        n_clusters = labels[not_noise].nunique()
        n_noise = int((~not_noise).sum())
        sil = (silhouette_score(embedding[not_noise], labels[not_noise])
              if n_clusters >= 2 else float("nan"))

        # plot group-by-group (not one scatter with a color array) so each
        # group gets its own legend handle for free, in a fixed draw order
        # (real clusters, then the lumped "other clusters" bucket, then noise)
        for grp_name in sorted(name.unique(),
                              key=lambda g: (g == "noise", g == "other clusters", g)):
            mask = name == grp_name
            ax.scatter(embedding.loc[mask, x_col], embedding.loc[mask, y_col],
                      s=18, linewidths=0, alpha=0.85 if grp_name != "noise" else 0.4,
                      color=color[mask].iloc[0], label=grp_name)

        subtitle = f"{title}\nk={n_clusters}"
        if n_noise:
            subtitle += f", {n_noise} noise"
        if color_by is None and true_labels is None and n_clusters > max_clusters:
            subtitle += f" (showing top {max_clusters})"
        subtitle += f", 2D silhouette={sil:.2f}" if n_clusters >= 2 else ""
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(fontsize=8, markerscale=1.5, frameon=False)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Decision tree plotting
#
# Renders a fitted sklearn DecisionTreeClassifier via graphviz by walking its
# internal `tree_` arrays directly, rather than via `sklearn.tree.export_
# graphviz`/`plot_tree` -- both of those always render from the root, whereas
# walking the arrays ourselves lets `node_id` start anywhere, so the exact
# same function renders a whole tree (`node_id=0`, the default) or a
# sub-tree rooted at any internal node.
# ---------------------------------------------------------------------------

# Per-class leaf highlight styles -- distinguished from each other and from
# the fill colors by shape/pattern (border count, dash) rather than hue, so
# they read without relying on color perception at all.
_HIGHLIGHT_STYLES = [
    {"color": "black", "penwidth": "4", "peripheries": "2"},                    # double solid border
    {"color": "black", "penwidth": "4", "style": "rounded,filled,dashed"},      # dashed border
]


def _composition_split_summary(col: str, raw_col: str, thr: float, nid: int, node_paths,
                               fp: pd.DataFrame, fp_raw: pd.DataFrame) -> str:
    """Business-readable label for a composition/gap split, for which no
    single raw-share number reproduces the CLR split rule (see
    `fingerprint.raw_threshold`'s docstring). Instead, describes the split
    empirically: among the training clients that actually reach this node,
    the real (`fp_raw`) values on each side of `col <= thr`, as median + IQR.

    `raw_col` (from `fingerprint.raw_threshold`) is the `fp_raw` column name
    to read -- for `kind="clr"` this is always equal to `col`, but reading it
    from `raw_threshold` rather than reusing `col` directly keeps the naming
    convention in that one function, not re-assumed here.
    """
    mask = node_paths[:, nid].toarray().ravel().astype(bool)
    goes_left = fp[col].values <= thr
    left_vals = fp_raw.loc[fp.index[mask & goes_left], raw_col]
    right_vals = fp_raw.loc[fp.index[mask & ~goes_left], raw_col]
    is_gap = ":gap:" in col
    if is_gap:
        fmt = lambda s: (f"{s.median():+.0%} (IQR {s.quantile(.25):+.0%} "
                         f"to {s.quantile(.75):+.0%})")
    else:
        fmt = lambda s: (f"{s.median():.0%} (IQR {s.quantile(.25):.0%}"
                         f"–{s.quantile(.75):.0%})")
    return f"{raw_col} (raw)\n≤: {fmt(left_vals)}\n>: {fmt(right_vals)}"


def plot_decision_tree(model, feature_names: list[str], class_names: list[str],
                       node_id: int = 0, max_depth: int | None = None,
                       highlight_classes: list[str] | None = None,
                       show_class_label: bool = False,
                       fp: pd.DataFrame | None = None,
                       fp_raw: pd.DataFrame | None = None,
                       raw_composition_summary: bool = True,
                       savepath: str | Path | None = None):
    """Render `model` (a fitted `DecisionTreeClassifier`) -- or the sub-tree
    rooted at `node_id` -- as a graphviz `Digraph`, with every node's
    per-class sample counts shown in its label, and a legend mapping each
    class name to its fill color.

    `class_names` order must match `model.classes_` (e.g.
    `cluster_vs_rest_importance`'s binary tree uses `y = (labels ==
    cluster_id).astype(int)`, so `class_names=["rest", f"cluster
    {cluster_id}"]`).

    `node_id` (default 0, the root) picks which node's sub-tree to draw --
    pass any internal node id (e.g. from inspecting `model.tree_`) to zoom
    into one branch of a deeper tree instead of the whole thing.
    `max_depth` optionally caps how many levels below `node_id` are drawn.

    `highlight_classes`: for each class named here (each must be one of
    `class_names`), the leaf with the most members of that class (by raw
    count, not proportion) gets its own border style from
    `_HIGHLIGHT_STYLES`, e.g. `highlight_classes=["rest", f"cluster
    {cluster_id}"]` marks the leaf holding the most "rest" clients with a
    double border and the one holding the most `cluster_id` clients with a
    dashed border. If the same leaf wins for two classes, only the
    last-applied style is visible (dot merges repeated node attributes).

    `show_class_label`: whether each node's label also spells out
    `class = <name>` for its majority class. Default `False` since the
    legend already covers that via fill color -- set `True` for a version
    that's readable without the legend (e.g. once a subtree is cropped out
    of the full figure).

    `savepath`'s suffix (`.png`/`.pdf`/`.svg`) picks the render format; if
    omitted, nothing is written to disk and only the `Digraph` is returned
    (e.g. to display inline in a notebook). Always returns the `Digraph`.

    `fp`/`fp_raw`: the exact (already noise-filtered/reindexed) fingerprint
    tables the tree was fit on, needed to render split labels in business-
    legible raw units instead of `fp`'s normalized ones. When omitted
    (default), every split renders exactly as `f"{feature} <= {threshold}"`
    on the normalized value, as before. When given: intensity/temporal
    `log_*` splits and already-identical direction/concentration/complexity/
    `temporal:cv_gap` splits convert exactly (see `fingerprint.raw_threshold`);
    composition/gap splits have no exact single-number raw equivalent (a CLR
    value depends on a client's whole category vector, not just the split
    column), so they instead show an empirical description -- the real
    `fp_raw` values (median + IQR) of the training clients on each side of
    the split -- unless `raw_composition_summary=False`, which keeps those
    particular splits on today's technical CLR-threshold display (e.g. to
    compare the two side by side).
    """
    import graphviz

    t = model.tree_
    n_colors = len(_OKABE_ITO)
    highlight_idxs = ([class_names.index(c) for c in highlight_classes]
                      if highlight_classes is not None else [])
    best_leaves = {idx: {"nid": None, "count": -1} for idx in highlight_idxs}
    node_paths = model.decision_path(fp.values) if fp is not None else None

    dot = graphviz.Digraph()
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")

    def walk(nid: int, depth: int) -> None:
        # `tree_.value` holds each class's *proportion* within the node, not
        # raw counts -- multiply back out by the node's (weighted) sample
        # count to get per-class client counts.
        proportions = t.value[nid][0]
        counts_per_class = proportions * t.weighted_n_node_samples[nid]
        majority = int(proportions.argmax())
        is_leaf = t.children_left[nid] == -1
        depth_capped = max_depth is not None and depth >= max_depth

        lines = []
        if not is_leaf and not depth_capped:
            col = feature_names[t.feature[nid]]
            thr = t.threshold[nid]
            if fp is None or fp_raw is None:
                split_label = f"{col} <= {thr:.3g}"
            else:
                raw_col, kind, raw_thr = raw_threshold(col, thr)
                if kind == "clr" and raw_composition_summary:
                    split_label = _composition_split_summary(col, raw_col, thr, nid, node_paths, fp, fp_raw)
                elif kind == "clr":
                    split_label = f"{col} <= {thr:.3g}"  # technical CLR value, for comparison
                else:
                    split_label = f"{raw_col} <= {raw_thr:.3g}"
            lines.append(split_label)
        counts = ", ".join(f"{cn}={round(v)}" for cn, v in zip(class_names, counts_per_class))
        lines += [f"samples = {t.n_node_samples[nid]}", counts]
        if show_class_label:
            lines.append(f"class = {class_names[majority]}")

        dot.node(str(nid), "\n".join(lines),
                 fillcolor=_OKABE_ITO[majority % n_colors])

        if is_leaf:
            for idx in highlight_idxs:
                count = counts_per_class[idx]
                if count > best_leaves[idx]["count"]:
                    best_leaves[idx].update(nid=nid, count=count)

        if is_leaf or depth_capped:
            return
        left, right = t.children_left[nid], t.children_right[nid]
        dot.edge(str(nid), str(left), label="True")
        walk(left, depth + 1)
        dot.edge(str(nid), str(right), label="False")
        walk(right, depth + 1)

    walk(node_id, 0)

    # Re-declaring the same node name adds/overrides these attributes (dot
    # merges repeated node statements) without needing to repeat its
    # label/fillcolor.
    for style_i, idx in enumerate(highlight_idxs):
        nid = best_leaves[idx]["nid"]
        if nid is not None:
            dot.node(str(nid), **_HIGHLIGHT_STYLES[style_i % len(_HIGHLIGHT_STYLES)])

    # Legend: one swatch + name per class, in an HTML-like table node so it
    # renders as a single self-contained key regardless of where dot decides
    # to place this (otherwise-disconnected) node in the layout.
    rows = "".join(
        f'<TR><TD BGCOLOR="{_OKABE_ITO[i % n_colors]}" WIDTH="18" HEIGHT="18"></TD>'
        f'<TD ALIGN="LEFT">{cn}</TD></TR>'
        for i, cn in enumerate(class_names))
    dot.node("legend", "<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
                       f'CELLPADDING="4">{rows}</TABLE>>', shape="plain")

    if savepath is not None:
        savepath = Path(savepath)
        fmt = savepath.suffix.lstrip(".") or "png"
        dot.format = fmt
        dot.render(savepath.with_suffix(""), cleanup=True)
    return dot


def _scale_drawing(drawing, target_width: float) -> None:
    """Scale a reportlab `Drawing` in place (preserving aspect ratio) so its
    width equals `target_width` points. `renderPDF.draw` doesn't consult
    `drawing.width`/`.height` at all -- it just renders whatever transform is
    on the drawing -- so besides applying the scale, this also updates those
    two attributes to match, purely so callers can use them for layout math
    (page size, stacking panels) afterwards."""
    factor = target_width / drawing.width
    drawing.scale(factor, factor)
    drawing.width *= factor
    drawing.height *= factor


def plot_decision_tree_with_importances(model, feature_names: list[str], class_names: list[str],
                                        importances: pd.Series, node_id: int = 0,
                                        max_depth: int | None = None,
                                        highlight_classes: list[str] | None = None,
                                        show_class_label: bool = False,
                                        fp: pd.DataFrame | None = None,
                                        fp_raw: pd.DataFrame | None = None,
                                        raw_composition_summary: bool = True,
                                        top_n: int = 10, title: str | None = None,
                                        savepath: str | Path | None = None,
                                        tree_width: float = 500,
                                        imp_width_frac: float = 0.85,
                                        canvas=None) -> bytes | None:
    """One PDF page combining `plot_decision_tree`'s diagram (top) with a
    smaller horizontal bar chart of `importances` (e.g. from
    `cluster_vs_rest_importance`/`cluster_surrogate_importance`, bottom) --
    so the tree and the ranking of what drove its splits ship as a single
    file instead of two.

    Both panels are rendered and composited as vector graphics (via SVG,
    then reportlab), never rasterized, so the result stays crisp at any zoom
    level -- unlike an earlier version of this function that rasterized the
    tree to PNG via `graphviz.Digraph.pipe` and embedded it as an image.
    That PNG path went through graphviz's `gd`/`pango` rendering plugins,
    which on some installs (notably Windows setups missing
    `gvplugin_pango.dll`) silently fall back to font metrics that don't
    match what graphviz used to size each node's box, so labels spilled
    outside their box. SVG output comes from graphviz's built-in `core`
    plugin instead -- no rendering plugin, no font-metric mismatch -- and
    text stays text (not pixels) all the way to the final PDF.

    The tree is rendered exactly as `plot_decision_tree` would (same
    `node_id`/`max_depth`/`highlight_classes`/`show_class_label`, including
    its class-color legend). `tree_width` is its rendered width in points
    (1/72"); the importances panel is scaled to `imp_width_frac` of that
    (default 85%), and the tree's own aspect ratio (from graphviz's layout)
    sets its height. Its y-axis labels (the feature names) are drawn at a
    smaller font than matplotlib's default, since `tight_layout` otherwise
    gives long feature names most of the panel's width and squeezes the
    bars themselves into a narrow strip.

    `title`, if given, is drawn centered above the tree panel.

    `fp`/`fp_raw`/`raw_composition_summary` are forwarded to
    `plot_decision_tree` as-is -- see its docstring for how they switch split
    labels from normalized to business-legible raw units.

    Requires `svglib` and `reportlab` (pure-Python, no system libraries) in
    addition to `graphviz`.

    `canvas`: an existing `reportlab.pdfgen.canvas.Canvas` to draw this page
    onto instead of creating a new one -- lets a caller combine several
    calls (e.g. one per cluster) into a single multi-page PDF by passing the
    same canvas each time and calling `.save()` itself once all pages are
    drawn. When given, `savepath` is ignored, this function only advances
    the page (`showPage`) rather than saving, and returns `None`.

    If `savepath` is given (and `canvas` is not), the PDF is written there
    and this returns `None`; otherwise nothing is written and the PDF bytes
    are returned.
    """
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    from reportlab.pdfgen import canvas as pdfcanvas

    dot = plot_decision_tree(model, feature_names, class_names, node_id=node_id,
                             max_depth=max_depth, highlight_classes=highlight_classes,
                             show_class_label=show_class_label, fp=fp, fp_raw=fp_raw,
                             raw_composition_summary=raw_composition_summary)
    tree_drawing = svg2rlg(io.BytesIO(dot.pipe(format="svg")))
    _scale_drawing(tree_drawing, tree_width)

    top = importances.head(top_n).iloc[::-1]  # ascending, so barh shows the biggest at the top
    imp_fig, ax_imp = plt.subplots(figsize=(tree_width / 72, max(2.0, 0.3 * len(top))))
    ax_imp.barh(top.index.astype(str), top.values, color=_OKABE_ITO[1])
    ax_imp.tick_params(axis="y", labelsize=8)
    ax_imp.set_xlabel("feature importance")
    ax_imp.set_title(f"top {len(top)} features", fontsize=10)
    imp_fig.tight_layout()
    imp_buf = io.BytesIO()
    imp_fig.savefig(imp_buf, format="svg")
    plt.close(imp_fig)
    imp_buf.seek(0)
    imp_drawing = svg2rlg(imp_buf)
    _scale_drawing(imp_drawing, imp_width_frac * tree_width)

    margin = 20
    title_h = 24 if title else 0
    gap = 12
    page_w = max(tree_drawing.width, imp_drawing.width) + 2 * margin
    page_h = title_h + tree_drawing.height + gap + imp_drawing.height + 2 * margin

    reuse_canvas = canvas is not None
    if reuse_canvas:
        c = canvas
        c.setPageSize((page_w, page_h))
    else:
        out = savepath if savepath is not None else io.BytesIO()
        c = pdfcanvas.Canvas(str(out) if savepath is not None else out, pagesize=(page_w, page_h))

    y = page_h - margin
    if title is not None:
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(page_w / 2, y - 14, title)
        y -= title_h

    y -= tree_drawing.height
    renderPDF.draw(tree_drawing, c, (page_w - tree_drawing.width) / 2, y)

    y -= gap + imp_drawing.height
    renderPDF.draw(imp_drawing, c, (page_w - imp_drawing.width) / 2, y)

    c.showPage()
    if reuse_canvas:
        return None
    c.save()
    return None if savepath is not None else out.getvalue()
