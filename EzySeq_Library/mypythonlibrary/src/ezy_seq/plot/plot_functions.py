from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from typing import List


def volcano(
    de_df,
    lfc_col="log2fc",
    p_col="padj",
    lfc_thresh=None,
    p_thresh=0.05,
    top_labels=30,
    title=None,
    ax=None,
    figsize=(6, 5),
    y_axis='score',
    y_axis_label=None,
    balance_labels=False,
    y_max=None,
    x_range=None,
    default_colors=None,
    significant_colors2=['green', 'purple'],
    significant_colors=['blue', 'red'],
    sig_markers=[None, None],
    return_highlights=False,
    highlight_genes=None,
    highlight_genes2=None,
    gene_colors=None,
):
    """
    Volcano plot of DE results with flexible highlighting and annotation.

    Parameters
    ----------
    de_df : pd.DataFrame
        DE results. Must contain `lfc_col`, `p_col`, and a 'gene' column.
    lfc_col : str
        Column name for log2 fold change.
    p_col : str
        Column name for adjusted p-value.
    lfc_thresh : float, optional
        Log2FC threshold; drawn as dashed vertical lines at ±lfc_thresh.
    p_thresh : float
        Significance threshold; drawn as a dashed horizontal line.
    top_labels : int
        Number of genes to annotate when highlight_genes is None.
    balance_labels : bool
        If True, annotate equal numbers of up- and down-regulated genes.
    y_axis : str
        Column to use for the y-axis (default 'score'). Falls back to
        -log10(padj) if the column is not present.
    highlight_genes : list of [neg_list, pos_list], optional
        Genes to highlight by direction. Overrides automatic top-N selection.
        Format: [list_of_down_genes, list_of_up_genes].
    highlight_genes2 : list of [neg_list, pos_list], optional
        A second gene set overlaid with different colors/markers.
    gene_colors : dict, optional
        Per-gene color overrides applied on top of group colors:
        {'GeneA': '#ff0000', 'GeneB': 'orange'}.
    significant_colors : list of 2 colors
        Colors for [positive, negative] highlighted genes (first set).
    significant_colors2 : list of 2 colors
        Colors for the second highlight set.
    sig_markers : list of 2 matplotlib marker strings
        Markers for [positive, negative] highlighted genes.
    y_max : float, optional
        Upper ceiling for the y-axis.
    x_range : float or (min, max), optional
        Symmetric float or explicit (min, max) limits for the x-axis.
    default_colors : list of 2 colors, optional
        Background scatter colors for [positive, negative] y-values.
        Defaults to ['salmon', 'lightblue'].
    ax : matplotlib Axes, optional
        Axes to draw on; creates a new figure if None.
    figsize : tuple
        Figure size used when ax is None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = de_df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=[lfc_col, p_col])
    df["neglog10p"] = -np.log10(np.clip(df[p_col], 1e-300, None))
    yvals = df[y_axis] if y_axis in df.columns else df["neglog10p"]
    eff_lfc = float(lfc_thresh) if lfc_thresh is not None else 0.0

    labels__df = pd.DataFrame()
    pos_pick, neg_pick = pd.DataFrame(), pd.DataFrame()
    df['_abs_score'] = yvals.abs()

    if highlight_genes is not None:
        h1_pos, h1_neg = set(highlight_genes[1]), set(highlight_genes[0])
        all_targets = h1_pos | h1_neg
        if highlight_genes2:
            all_targets |= set(highlight_genes2[1]) | set(highlight_genes2[0])

        labels__df = df[df["gene"].isin(all_targets)].copy()
        if labels__df.empty:
            print("Warning: highlight_genes provided, but no matches found in df['gene'].")

        m1_pos = df["gene"].isin(h1_pos)
        m1_neg = df["gene"].isin(h1_neg)
    else:
        is_sig = (abs(df[lfc_col]) > eff_lfc) & (df[p_col] < p_thresh)
        m1_pos = is_sig & (df[lfc_col] > 0)
        m1_neg = is_sig & (df[lfc_col] <= 0)

        if balance_labels:
            n = max(1, top_labels // 2)
            pool_pos = df[m1_pos] if m1_pos.any() else df[df[lfc_col] > 0]
            pool_neg = df[m1_neg] if m1_neg.any() else df[df[lfc_col] < 0]
            pos_pick = pool_pos.nlargest(n, columns=['_abs_score'])
            neg_pick = pool_neg.nlargest(n, columns=['_abs_score'])
            labels__df = pd.concat([pos_pick, neg_pick])
        else:
            if is_sig.any():
                labels__df = df[is_sig].nlargest(top_labels, columns=['_abs_score'])
            else:
                labels__df = df.nlargest(top_labels, columns=['_abs_score'])
            pos_pick = labels__df[labels__df[lfc_col] > 0]
            neg_pick = labels__df[labels__df[lfc_col] < 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    bg_colors = (
        ['salmon' if v > 0 else 'lightblue' for v in yvals]
        if default_colors is None
        else [default_colors[0] if v > 0 else default_colors[1] for v in yvals]
    )
    ax.scatter(df[lfc_col], yvals.abs(), s=10, alpha=0.3, c=bg_colors)

    def scatter_subset(mask, default_color, marker):
        if not mask.any():
            return
        subset_df = df.loc[mask]
        if gene_colors is not None and "gene" in subset_df.columns:
            c_vals = subset_df["gene"].map(gene_colors).fillna(default_color)
        else:
            c_vals = default_color
        ax.scatter(subset_df[lfc_col], yvals.loc[mask].abs(), s=13, c=c_vals,
                   marker=marker, alpha=1)

    if highlight_genes2 is None:
        scatter_subset(m1_pos, significant_colors[0], sig_markers[0])
        scatter_subset(m1_neg, significant_colors[1], sig_markers[1])
    else:
        h2_pos, h2_neg = set(highlight_genes2[1]), set(highlight_genes2[0])
        m2_pos, m2_neg = df["gene"].isin(h2_pos), df["gene"].isin(h2_neg)
        union_1 = m1_pos | m1_neg
        union_2 = m2_pos | m2_neg
        overlap  = union_1 & union_2

        scatter_subset(m1_pos & ~overlap, significant_colors[0], 'o')
        scatter_subset(m1_neg & ~overlap, significant_colors[1], 'o')
        scatter_subset(m2_pos & ~overlap, significant_colors2[0], sig_markers[0])
        scatter_subset(m2_neg & ~overlap, significant_colors2[1], sig_markers[1])
        scatter_subset(overlap & (m2_pos | (~m2_pos & ~m2_neg & m1_pos)), 'black', sig_markers[0])
        scatter_subset(overlap & (m2_neg | (~m2_pos & ~m2_neg & m1_neg)), 'black', sig_markers[1])

    if isinstance(lfc_thresh, (float, int)):
        ax.axvline(eff_lfc, linestyle="--", lw=1)
        ax.axvline(-eff_lfc, linestyle="--", lw=1)

    if "gene" in df.columns and not labels__df.empty:
        to_label = labels__df.copy()
        if lfc_thresh:
            to_label = to_label[to_label[lfc_col].abs() >= eff_lfc]
        if p_thresh:
            to_label = to_label[to_label[p_col] <= p_thresh]
        for _, r in to_label.iterrows():
            ax.annotate(
                r["gene"],
                (r[lfc_col], abs(r[y_axis] if y_axis in r else r["neglog10p"])),
                xytext=(3, 3), textcoords="offset points", fontsize=8,
            )

    if title is None:
        title = (
            f"Volcano: {df['case'].iloc[0]} vs {df['control'].iloc[0]}"
            if {"case", "control"}.issubset(df.columns)
            else "Volcano plot"
        )
    ax.set_title(title)
    ax.set_xlabel("log2 fold change")
    ax.axhline(0, linestyle="--", lw=1)

    if y_axis_label:
        ax.set_ylabel(y_axis_label)
    elif y_axis == 'score':
        ax.set_ylabel("Wilcoxon Score")
    else:
        ax.set_ylabel(
            "-log10(adj p-value)"
            if y_axis == 'neglog10p' or y_axis not in df.columns
            else y_axis
        )
        if y_axis not in df.columns or y_axis == 'neglog10p':
            ax.axhline(-np.log10(p_thresh), linestyle="--", lw=1)

    if y_max:
        ax.set_ylim(top=float(y_max))
    if x_range:
        try:
            if isinstance(x_range, (list, tuple)):
                ax.set_xlim(float(x_range[0]), float(x_range[1]))
            else:
                ax.set_xlim(-abs(float(x_range)), abs(float(x_range)))
        except Exception:
            pass

    plt.show()
    return ax


def rank_and_plot(
    adata,
    groupby: str,
    comparisons: str | list[tuple[str, str]] = "all_vs_ref",
    reference: str | None = None,
    groups_keep: list[str] | None = None,
    cell_key: str | None = None,
    cells_keep: list[str] | None = None,
    extra_filter: dict | None = None,
    show_title: bool = True,
    method: str = "wilcoxon",
    label_mode: str = "all",
    gene_list: list[str] | None = None,
    n_labels: int = 30,
    soft_logp_cap: float = 100000000,
    cap_logp: float = 100000000,
    cap_lfc_left=None,
    cap_lfc_right=None,
    label_x_min: float = None,
    label_y_min: float = None,
    figsize=(10, 8),
    show: bool = True,
    return_tables: bool = True,
    pallete: dict[str, object] | None = None,
    x_font: int = 32,
    y_font: int = 32,
    x_tick_font: int = 18,
    y_tick_font: int = 18,
    title_font=32,
    gene_font=22,
    plot: bool = True,
) -> None:
    """
    Run differential expression and plot |score| vs log2FC for each comparison.

    Parameters
    ----------
    groupby : str
        Column in adata.obs to compare (must be categorical or convertible).
    comparisons : "all_vs_ref" or list of (target, reference) tuples
        - "all_vs_ref": compare every other category vs `reference`.
        - list of tuples: explicit pairwise comparisons.
    reference : str
        Reference level; required when comparisons == "all_vs_ref".
    groups_keep : list[str], optional
        Restrict to a subset of categories within `groupby`.
    cell_key : str, optional
        Column in adata.obs defining a cell-type subgroup to filter by.
    cells_keep : list[str], optional
        Values in `cell_key` to keep.
    extra_filter : dict[str, list[str]], optional
        Additional {obs_column: allowed_values} filters applied before DE.
    method : str
        scanpy.tl.rank_genes_groups method (e.g., "wilcoxon").
    label_mode : {"all", "whitelist"}
        If "whitelist", annotate only genes in `gene_list`.
    gene_list : list[str], optional
        Genes allowed for labeling when label_mode == "whitelist".
    n_labels : int
        Number of top genes (by |lfc|) to annotate.
    soft_logp_cap, cap_logp : float
        Caps for -log10(padj). Values above soft_logp_cap are log2-compressed.
    cap_lfc_left, cap_lfc_right : float, optional
        Clip log2FC to [cap_lfc_left, cap_lfc_right] in the scatter plot.
    figsize : tuple
        Matplotlib figure size.
    show : bool
        Display plots immediately.
    return_tables : bool
        If True, return a dict of DataFrames with statistics per comparison.
    pallete : dict, optional
        Custom color map {category: color}.
    plot : bool
        If False, skip plotting and only return tables.

    Returns
    -------
    dict or None
        {(target, ref): DataFrame} if return_tables else None.
    """
    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} not found in adata.obs")
    if comparisons == "all_vs_ref" and not reference:
        raise ValueError("comparisons='all_vs_ref' requires `reference` to be provided.")
    if label_mode not in {"all", "whitelist"}:
        raise ValueError("label_mode must be 'all' or 'whitelist'.")
    if label_mode == "whitelist" and not gene_list:
        raise ValueError("Provide `gene_list` when label_mode == 'whitelist'.")

    obs = adata.obs.copy()
    if not pd.api.types.is_categorical_dtype(obs[groupby]):
        obs[groupby] = pd.Categorical(obs[groupby])

    if groups_keep is not None:
        obs = obs[obs[groupby].isin(set(groups_keep))]
    if cell_key and cells_keep:
        if cell_key not in obs:
            raise KeyError(f"{cell_key!r} not found in adata.obs")
        obs = obs[obs[cell_key].isin(cells_keep)]
    if extra_filter:
        for key, allowed in extra_filter.items():
            if key not in obs:
                raise KeyError(f"{key!r} not found in adata.obs")
            obs = obs[obs[key].isin(allowed)]

    adata_filt = adata[obs.index].copy()
    cats = list(pd.Categorical(adata_filt.obs[groupby]).categories)

    if comparisons == "all_vs_ref":
        if reference not in cats:
            raise ValueError(f"reference {reference!r} not found in {groupby} categories: {cats}")
        pairs = [(c, reference) for c in cats if c != reference]
    else:
        pairs = comparisons
        for t, r in pairs:
            if t not in cats or r not in cats:
                raise ValueError(f"Pair ({t},{r}) uses unknown category; available: {cats}")

    tab10 = plt.cm.tab10.colors
    colors = pallete or {}
    color_map = {lvl: colors.get(lvl, tab10[i % len(tab10)]) for i, lvl in enumerate(cats)}
    results = {}

    for target, ref in pairs:
        sub = adata_filt[adata_filt.obs[groupby].isin([target, ref])].copy()
        if pd.api.types.is_categorical_dtype(sub.obs[groupby]):
            sub.obs[groupby] = sub.obs[groupby].cat.remove_unused_categories()

        sc.tl.rank_genes_groups(sub, groupby=groupby, method=method, reference=ref, pts=True)

        res    = sub.uns["rank_genes_groups"]
        names  = list(res["names"][target])
        lfcs   = np.asarray(res["logfoldchanges"][target], dtype=float)
        scores = np.asarray(res["scores"][target], dtype=float)
        padj   = np.asarray(res["pvals_adj"][target], dtype=float)

        caps = [cap_lfc_left, cap_lfc_right]
        for i, cap in enumerate(caps):
            if cap is None:
                caps[i] = min(lfcs) if i == 0 else max(lfcs)

        lfcs_clip = np.clip(lfcs, caps[0], caps[1])
        neglogp = -np.log10(np.clip(padj, 1e-300, None))
        over = neglogp > soft_logp_cap
        neglogp[over] = soft_logp_cap + np.log2((neglogp[over] - soft_logp_cap))
        neglogp = np.minimum(neglogp, cap_logp)

        df = pd.DataFrame({
            "gene":        names,
            "log2fc":      lfcs,
            "log2fc_clip": lfcs_clip,
            "score":       scores,
            "padj":        padj,
            "-log10_padj": neglogp,
        })

        if return_tables:
            results[(target, ref)] = df.copy()

        if not plot:
            continue

        genes_oi = df[df["gene"].isin(gene_list)] if label_mode == "whitelist" else df
        label_df = genes_oi.reindex(
            genes_oi["log2fc"].abs().sort_values(ascending=False).index
        ).head(n_labels)

        plt.figure(figsize=figsize)
        plt.scatter(df["log2fc_clip"], abs(df["score"]), alpha=0.35, color='grey')
        plt.axvline(0, color="grey", linestyle="--", linewidth=1)
        plt.scatter(
            label_df["log2fc_clip"], abs(label_df["score"]),
            s=50,
            c=[color_map[target] if s > 0 else color_map[ref] for s in label_df["score"]],
            alpha=0.9,
        )
        for _, row in label_df.iterrows():
            plt.text(row["log2fc_clip"], abs(row["score"]), row["gene"],
                     fontsize=gene_font, ha="center", va="bottom")

        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=x_tick_font)
        ax.tick_params(axis='y', labelsize=y_tick_font)
        for label in ax.yaxis.get_ticklabels():
            if label.get_text() == str(soft_logp_cap):
                label.set_fontweight("bold")

        plt.xlabel("log₂ Fold Change", fontsize=x_font)
        plt.ylabel("|score|", fontsize=y_font)
        if show_title:
            plt.title(f"{groupby}: {target} vs {ref}", fontsize=title_font)
        plt.tight_layout()
        if show:
            plt.show()

        if return_tables:
            results[(target, ref)] = df.reindex(
                df["score"].abs().sort_values(ascending=False).index
            )

    return results if return_tables else None


def feats_bar(
    df,
    num_feats: int = 10,
    fig_size=None,
    title=None,
    variable: str = "score",
    orientation: str = 'vertical',
    bold_genes: list = None,
    starred_genes: list = None,
    bold_colors=None,
    colors=((85/255, 160/255, 251/255), (249/255, 100/255, 149/255)),
    hatches=["", ""],
) -> None:
    """
    Bar chart of top DE genes ranked by a chosen score variable.

    Parameters
    ----------
    df : pd.DataFrame
        DE results with 'gene' and `variable` columns.
    num_feats : int
        Number of genes to display.
    variable : str
        Column to rank and plot (e.g., 'score', 'log2fc').
    orientation : {'vertical', 'horizontal'}
        'vertical' draws horizontal bars (gene on y-axis).
        'horizontal' draws vertical bars with up/down genes side by side.
    bold_genes : list, optional
        Gene names to bold and outline.
    starred_genes : list, optional
        Gene names to mark with an asterisk.
    bold_colors : list of 2 colors, optional
        Override colors for bold genes: [negative_color, positive_color].
    colors : tuple of 2 colors
        Default fill colors for [negative, positive] values.
    hatches : list of 2 str
        Hatch patterns for [negative, positive] bars.
    """
    neg_clr = colors[0]
    pos_clr = colors[1]

    def _pick_hatch(value):
        return hatches[1] if value > 0 else hatches[0]

    if orientation == 'vertical':
        if fig_size is None:
            plt.figure(figsize=(8, 5))
        else:
            plt.figure(figsize=fig_size)

        df = df.reindex(df[variable].abs().sort_values(ascending=False).index).head(num_feats)
        plt.barh(
            df["gene"],
            df[variable],
            color=df[variable].apply(lambda x: pos_clr if x > 0 else neg_clr),
            hatch=[_pick_hatch(v) for v in df[variable]],
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.ylabel("Gene")
        plt.title(title if title is not None else f"Differential expression ({variable})")

        ax = plt.gca()
        for label in ax.get_yticklabels():
            if bold_genes and label.get_text() in bold_genes:
                label.set_fontweight('bold')

        for bar, gene in zip(ax.containers[0], df["gene"]):
            if bold_genes and gene in bold_genes:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)

        plt.show()

    if orientation == 'horizontal':
        pos = df[df[variable] > 0].reindex(
            df[df[variable] > 0][variable].abs().sort_values(ascending=False).index
        )
        neg = df[df[variable] < 0].reindex(
            df[df[variable] < 0][variable].abs().sort_values(ascending=False).index
        )

        pos_n = (num_feats + 1) // 2
        neg_n = num_feats - pos_n
        pos = pos.head(pos_n)
        neg = neg.head(neg_n).reindex(
            neg.head(neg_n)[variable].sort_values(ascending=False).index
        )
        plot_df = pd.concat([neg, pos], axis=0)

        if fig_size is None:
            plt.figure(figsize=(4 + (num_feats / 2.5), 5))
        else:
            plt.figure(figsize=fig_size)

        plt.bar(
            plot_df["gene"],
            plot_df[variable],
            color=plot_df[variable].apply(lambda x: pos_clr if x > 0 else neg_clr),
            hatch=plot_df[variable].apply(lambda x: hatches[0] if x > 0 else hatches[1]),
        )
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(rotation=45, ha='right')
        plt.title(title if title is not None else f"Differential expression ({variable})")

        plotted_genes = plot_df["gene"]
        plotted_vals  = plot_df[variable]
        ax = plt.gca()
        ax.set_ylabel(variable)

        if bold_genes:
            for label in ax.get_xticklabels():
                if label.get_text() in bold_genes:
                    label.set_fontweight('bold')

            effective_bold_colors = bold_colors if bold_colors is not None else colors
            for bar, gene, val in zip(ax.containers[0], plotted_genes, plotted_vals):
                if gene in bold_genes:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)
                    bar.set_facecolor(
                        effective_bold_colors[1] if val > 0 else effective_bold_colors[0]
                    )

        for bar, gene, val in zip(ax.containers[0], plotted_genes, plotted_vals):
            if starred_genes and gene in starred_genes:
                y_lims  = ax.get_ylim()
                offset  = (y_lims[1] - y_lims[0]) * 0.01
                plt.text(
                    x=bar.get_x() + bar.get_width() / 2,
                    y=val + (offset if val > 0 else -offset),
                    s="*",
                    ha='center',
                    va='bottom' if val > 0 else 'top',
                    fontsize=16,
                    color='black',
                    fontweight='bold',
                )

        plt.tight_layout()
        plt.show()


def umap(
    adata: "sc.AnnData",
    *,
    cluster_key: str = "cell_type",
    group_by1: str = None,
    group_by1_list="All",
    group_by2: str = None,
    group_by2_list="All",
    group_by3: str = None,
    group_by3_list="All",
    n_top_clusters: int = 20,
    n_neighbors: int = 25,
    n_pcs_: int = 10,
    random_state: int = 0,
    legend_fontsize: float = 10,
    show: bool = True,
) -> List["matplotlib.figure.Figure"]:
    """
    Compute a UMAP once on a filtered subset and plot per level for up to three
    grouping columns, all sharing the same cluster color palette.

    The UMAP is computed on a copy of adata; the input object is not mutated.
    Computing once and reusing the embedding ensures visually consistent plots
    across groups.

    Parameters
    ----------
    adata : AnnData
        Input object. Should be normalized and log-transformed.
    cluster_key : str
        .obs column with cluster/cell-type labels (used for point coloring).
    group_by1, group_by2, group_by3 : str, optional
        .obs columns to facet by. One UMAP is produced per level.
    group_by1_list, group_by2_list, group_by3_list : list or "All"
        Which levels of each grouping column to plot.
    n_top_clusters : int
        Restrict to the top-K most frequent clusters before computing UMAP.
    n_neighbors : int
        Neighborhood size for sc.pp.neighbors.
    n_pcs_ : int
        Number of PCs used for the neighbor graph.
    random_state : int
        Random seed for reproducibility.
    legend_fontsize : float
        Font size for the cluster legend.
    show : bool
        Whether to display figures immediately (passed to sc.pl.umap).

    Returns
    -------
    list[matplotlib.figure.Figure]
        One figure per requested group level, in order.
    """
    required = [cluster_key]
    group_specs = [
        (group_by1, group_by1_list),
        (group_by2, group_by2_list),
        (group_by3, group_by3_list),
    ]
    for g, _ in group_specs:
        if g is not None:
            required.append(g)

    for col in required:
        if col not in adata.obs.columns:
            raise KeyError(f"Column '{col}' not found in adata.obs")

    full = adata.copy()
    if full.n_obs == 0:
        raise ValueError("No cells left after applying group filters.")

    top_vals = (
        full.obs[cluster_key].value_counts(dropna=False).nlargest(n_top_clusters).index
    )
    full = full[full.obs[cluster_key].isin(top_vals)].copy()
    if full.n_obs == 0:
        raise ValueError(
            f"No cells left after restricting to top {n_top_clusters} of '{cluster_key}'."
        )

    sc.pp.neighbors(full, n_neighbors=n_neighbors, n_pcs=n_pcs_, random_state=random_state)
    sc.tl.umap(full, random_state=random_state)

    cluster_order = sorted(full.obs[cluster_key].dropna().astype(str).unique(), key=str)
    default_tab20 = plt.cm.tab20.colors
    palette_dict = {
        cat: default_tab20[(i + 4) % len(default_tab20)]
        for i, cat in enumerate(cluster_order)
    }

    full.obs[cluster_key] = pd.Categorical(full.obs[cluster_key], categories=cluster_order)
    full.uns[f"{cluster_key}_colors"] = [palette_dict[cat] for cat in cluster_order]

    def levels_for(g: str | None, glist) -> list[str]:
        if g is None:
            return []
        if isinstance(glist, str) and glist.lower() == "all":
            col = full.obs[g]
            return list(col.cat.categories) if hasattr(col, "cat") else list(col.unique())
        return list(glist)

    figs = []
    for g, glist in group_specs:
        if g is None:
            continue
        for level in levels_for(g, glist):
            sub = full[full.obs[g] == level]
            if sub.n_obs == 0:
                print(f"Skipping plot for {g}: {level} (0 cells)")
                continue

            sub.obs[cluster_key] = pd.Categorical(
                sub.obs[cluster_key], categories=cluster_order
            )
            fig = sc.pl.umap(
                sub,
                color=cluster_key,
                palette=palette_dict,
                title=f"{g}: {level} (n={sub.n_obs})",
                legend_fontsize=legend_fontsize,
                return_fig=True,
                show=show,
            )
            figs.append(fig)

    return figs
