#!/usr/bin/env python3
"""
visualize_g1_g2_comparison.py
==============================
G1 (UP) vs G2 (DOWN) Composition-Engineered Dataset Comparison

  G1 (UP)  : Stroke-FMT cortex-enriched (+1 SD), Healthy-FMT depleted
  G2 (DOWN): Healthy-FMT cortex-enriched (+1 SD), Stroke-FMT depleted

Loads specified cell types × models × annotations using lmm_comparison_utils,
generates Figures A–E for every combination, organized by cell type.

Output: ./submission/g1g2/{cell_type}/{model}_{annotation}/figX_*.png
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path
from adjustText import adjust_text
from matplotlib.patheffects import withStroke

warnings.filterwarnings("ignore")

AB_Y_DEFAULT = 8
AB_Y_SMALL   = 6.5
AB_Y_SMALL_CTS = {"Astrocytes", "Microglia"}

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR  = SCRIPT_DIR.parent                     # …/Manuscript
LMM_BASE    = PARENT_DIR / "LMM_extract"/"Seurat_&_Dream_updated_2_5"
DATA_PATH   = LMM_BASE / "Global_CT_Analysis"
LOCAL_PATH  = LMM_BASE / "Local_Regional_Analysis"
PB_PATH     = LMM_BASE / "Pseudobulk_Validation"
OUT_BASE    = SCRIPT_DIR / "S&Du_(search)" / "g1g2"

# ── Target Filters ───────────────────────────────────────────────────────────
# Define which models and cell types to process.
# Leave as empty lists [] to process EVERYTHING found in the directories.
TARGET_MODELS = ['dream'] # e.g., ['seurat', 'dream', 'deseq2']
TARGET_CELL_TYPES = ['Astrocytes', 'Microglia','WHOLE'] # e.g., ['WHOLE', 'Astrocytes']

# P-value selection
USE_ADJUSTED_PVAL = False  # True uses 'adj.P.Val', False uses unadjusted 'P.Value'
# ─────────────────────────────────────────────────────────────────────────────

# Dynamic P-value column configuration
PVAL_BASE_COL = "adj.P.Val" if USE_ADJUSTED_PVAL else "P.Value"
PVAL_DISP_NAME = "adj.P" if USE_ADJUSTED_PVAL else "P-value"

# Add parent dir to sys.path so we can import lmm_comparison_utils
sys.path.insert(0, str(PARENT_DIR))
import lmm_comparison_utils as lmm
from plot_utils import draw_proportional_venn

SIG = 0.05

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         15,
    "axes.titlesize":    17,
    "axes.titleweight":  "bold",
    "axes.labelsize":    15,
    "legend.fontsize":   13,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

C = {
    "g1":   "#CC4400",   # dark orange-red  — G1 (UP, Stroke cortex-enriched)
    "g2":   "#006699",   # dark teal        — G2 (DOWN, Healthy cortex-enriched)
    "both": "#00AA44",   # green            — Sig in Both
    "flip": "#9900CC",   # violet           — direction flip
    "up":   "#EE1111",   # red              — upregulated in volcanos
    "down": "#0033CC",   # navy             — downregulated in volcanos
}
GREY_COL   = "#808080"
GREY_ALPHA = 0.50
GREY_SIZE  = 18

# ── Load data using notebook pattern ─────────────────────────────────────────
print("=" * 70)
print("LOADING FILTERED G1 (UP) / G2 (DOWN) PAIRS")
print("=" * 70)

data_by_celltype = {}

def load_up_down_pair(up_path, down_path, model_type):
    std_model = "pseudobulk" if model_type == "deseq2" else model_type
    df_up  = lmm.standardize_columns(lmm.load_de_file(up_path), std_model)
    df_down = lmm.standardize_columns(lmm.load_de_file(down_path), std_model)
    return pd.merge(df_up, df_down, on="Gene", suffixes=("_up", "_down"), how="inner")

# Collect UP/DOWN files from all directories
up_files, down_files = {}, {}

for search_dir in [DATA_PATH, LOCAL_PATH]:
    if not search_dir.exists():
        print(f"  Warning: {search_dir} not found, skipping")
        continue
    for fp in search_dir.glob("*.csv"):
        parsed = lmm.parse_seurat_dream_filename(fp.name)
        if parsed is None or parsed["annotation"] is None:
            continue
        ct    = lmm.simplify_celltype_name(parsed["cell_type"])
        model = "seurat" if parsed["model"] == "seaurat" else parsed["model"]
        ann   = parsed["annotation"]

        # Apply Filters
        if TARGET_CELL_TYPES and ct not in TARGET_CELL_TYPES:
            continue
        if TARGET_MODELS and model not in TARGET_MODELS:
            continue

        key   = (model, ct, ann)
        if parsed["direction"] == "UP":
            up_files[key] = fp
        else:
            down_files[key] = fp

# Pseudobulk directory
if PB_PATH.exists():
    for fp in PB_PATH.glob("*.csv"):
        parsed = lmm.parse_seurat_dream_filename(fp.name)
        if parsed is None or parsed["model"] != "pseudobulk":
            continue
        ct  = lmm.simplify_celltype_name(parsed["cell_type"])

        # Apply Filters
        if TARGET_CELL_TYPES and ct not in TARGET_CELL_TYPES:
            continue
        if TARGET_MODELS and "deseq2" not in TARGET_MODELS:
            continue

        key = ("deseq2", ct, "pb")
        if parsed["direction"] == "UP":
            up_files[key] = fp
        else:
            down_files[key] = fp

# Merge pairs
all_keys = sorted(set(up_files.keys()) & set(down_files.keys()))
for key in all_keys:
    model, ct, ann = key
    try:
        merged = load_up_down_pair(up_files[key], down_files[key], model)
        data_by_celltype.setdefault(ct, {}).setdefault(model, {})[ann] = merged
        print(f"  {ct}/{model}/{ann}: {len(merged)} genes")
    except Exception as e:
        print(f"  FAIL {ct}/{model}/{ann}: {e}")

print(f"\nLoaded {len(all_keys)} combinations across {len(data_by_celltype)} cell types\n")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_adjust(texts, ax, **kwargs):
    """Run adjust_text and expand axis limits to include all label positions.

    Any legend attached to `ax` is automatically added to the objects
    adjust_text must avoid, so gene labels can't land underneath it.
    """
    leg = ax.get_legend()
    if leg is not None:
        objs = list(kwargs.get("objects") or [])
        objs.append(leg)
        kwargs["objects"] = objs
    adjust_text(texts, ax=ax, **kwargs)
    # Expand axis limits to encompass any labels that were pushed outward
    xl = list(ax.get_xlim())
    yl = list(ax.get_ylim())
    for t in texts:
        x, y = t.get_position()
        xl[0] = min(xl[0], x)
        xl[1] = max(xl[1], x)
        yl[0] = min(yl[0], y)
        yl[1] = max(yl[1], y)
    xpad = (xl[1] - xl[0]) * 0.03
    ypad = (yl[1] - yl[0]) * 0.03
    ax.set_xlim(xl[0] - xpad, xl[1] + xpad)
    ax.set_ylim(yl[0] - ypad, yl[1] + ypad)


def _save_venn_genes(outdir, prefix, gene_sets):
    """Save Venn gene lists to CSVs under {outdir}/venn_genes/."""
    venn_dir = os.path.join(outdir, "venn_genes")
    os.makedirs(venn_dir, exist_ok=True)
    for label, genes in gene_sets.items():
        pd.Series(sorted(genes), name="Gene").to_csv(
            os.path.join(venn_dir, f"{prefix}_{label}.csv"), index=False)


def label_all(ax, sub, xcol, ycol, gene_col="Gene", fontsize=13, col="#222"):
    """Return text objects for all rows in sub-dataframe for adjustText."""
    texts = []
    for _, row in sub.iterrows():
        t = ax.text(row[xcol], row[ycol], row[gene_col],
                    fontsize=fontsize, color=col, fontweight="bold", zorder=9,
                    clip_on=True)
        texts.append(t)
    return texts


def label_genes(ax, xs, ys, genes, n=10, criterion=None, fontsize=13, color="#222"):
    """Return text objects for top n genes by |criterion| for adjustText."""
    if criterion is None:
        criterion = pd.Series(np.ones(len(xs)), index=xs.index)
    idx = criterion.abs().nlargest(n).index
    texts = []
    for i in idx:
        texts.append(ax.text(xs[i], ys[i], genes[i],
                             fontsize=fontsize, color=color, zorder=8,
                             clip_on=True))
    return texts


def colored_sym_lim(xs, ys, pad=0.30, fallback=1.0):
    if len(xs) == 0: return fallback
    return max(float(np.abs(xs).max()), float(np.abs(ys).max())) * (1 + pad)

def colored_lim_rect(xs, ys, pad=0.30, fallback=1.0):
    if len(xs) == 0: return (0, fallback), (0, fallback)
    xlo, xhi = float(xs.min()), float(xs.max())
    ylo, yhi = float(ys.min()), float(ys.max())
    xp = max((xhi - xlo) * pad, fallback * 0.1)
    yp = max((yhi - ylo) * pad, fallback * 0.1)
    return (xlo - xp, xhi + xp), (ylo - yp, yhi + yp)

def scatter_grey_sym(ax, x, y, lim, label=None):
    xc = np.clip(np.asarray(x, float), -lim, lim)
    yc = np.clip(np.asarray(y, float), -lim, lim)
    ax.scatter(xc, yc, color=GREY_COL, alpha=GREY_ALPHA, s=GREY_SIZE,
               edgecolors="none", rasterized=True, zorder=1, label=label)

def scatter_grey_rect(ax, x, y, xlim, ylim, label=None):
    xc = np.clip(np.asarray(x, float), xlim[0], xlim[1])
    yc = np.clip(np.asarray(y, float), ylim[0], ylim[1])
    ax.scatter(xc, yc, color=GREY_COL, alpha=GREY_ALPHA, s=GREY_SIZE,
               edgecolors="none", rasterized=True, zorder=1, label=label)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_df(merged, logfc_thresh=None):
    """Add computed columns to a merged UP/DOWN dataframe."""
    df = merged.copy()

    p_up_col = f"{PVAL_BASE_COL}_up"
    p_down_col = f"{PVAL_BASE_COL}_down"

    df["sig_up"]      = df[p_up_col]   < SIG
    df["sig_down"]    = df[p_down_col]  < SIG
    if logfc_thresh is not None:
        df["sig_up"]   = df["sig_up"]   & (df["logFC_up"].abs()   >= logfc_thresh)
        df["sig_down"] = df["sig_down"] & (df["logFC_down"].abs() >= logfc_thresh)
    df["sign_flip"]   = np.sign(df["logFC_up"]) != np.sign(df["logFC_down"])
    df["delta_logFC"] = df["logFC_down"] - df["logFC_up"]
    df["mean_logFC"]  = (df["logFC_up"] + df["logFC_down"]) / 2
    df["nla_up"]      = -np.log10(df[p_up_col].clip(1e-300))
    df["nla_down"]    = -np.log10(df[p_down_col].clip(1e-300))

    def _cat(row):
        if row.sig_up and row.sig_down: return "Sig in Both"
        if row.sig_up:                  return "G1 Only"
        if row.sig_down:                return "G2 Only"
        return "Neither"

    df["sig_cat"] = df.apply(_cat, axis=1)
    return df


def _plot_figAB_combined(df, outdir, filename, ba_ylim=None, ba_xlim=None,
                          nla_lim=None, logfc_thresh=None, ab_y_axis=AB_Y_DEFAULT):
    """4-panel figAB_combined for any prepared dataframe. Self-contained."""
    cats = df["sig_cat"].value_counts().reindex(
        ["Sig in Both", "G1 Only", "G2 Only", "Neither"], fill_value=0)
    n_g1 = int(cats["G1 Only"])
    n_g2 = int(cats["G2 Only"])
    n_ab = int(cats["Sig in Both"])

    magnitude      = (df["logFC_up"].abs() + df["logFC_down"].abs()) / 2
    top10_mask     = magnitude >= magnitude.quantile(0.90)
    top10_df_ba    = df[top10_mask]
    abs_bias_all   = np.abs(df["delta_logFC"]).mean()
    abs_bias_top10 = np.abs(top10_df_ba["delta_logFC"]).mean() if len(top10_df_ba) > 1 else np.nan
    mean_abs_logfc_all   = magnitude.mean()
    mean_abs_logfc_top10 = magnitude[top10_mask].mean() if top10_mask.sum() > 0 else np.nan
    mab_pct_all   = abs_bias_all / mean_abs_logfc_all * 100 if mean_abs_logfc_all > 0 else np.nan
    mab_pct_top10 = (abs_bias_top10 / mean_abs_logfc_top10 * 100
                     if (not np.isnan(abs_bias_top10) and mean_abs_logfc_top10 > 0) else np.nan)
    top10_mab_str     = f"{abs_bias_top10:.3f}" if not np.isnan(abs_bias_top10) else "n/a"
    top10_mab_pct_str = f"{mab_pct_top10:.1f}%" if not np.isnan(mab_pct_top10) else "n/a"
    r_nla_sp, _ = stats.spearmanr(df["nla_up"], df["nla_down"])
    if top10_mask.sum() > 1:
        r_nla_sp_top10, _ = stats.spearmanr(df.loc[top10_mask, "nla_up"],
                                             df.loc[top10_mask, "nla_down"])
    else:
        r_nla_sp_top10 = np.nan

    sig_pool     = df[df["sig_cat"] != "Neither"]
    sig_genes_df = sig_pool.reindex(
        sig_pool["delta_logFC"].abs().nlargest(25).index
    ).sort_values("delta_logFC")
    top25_genes  = set(sig_genes_df["Gene"])
    colored = df["sig_cat"] != "Neither"
    grey    = ~colored
#__________________________________________________________________________________________________________________________________________________________________
    fig, axes = plt.subplots(1, 4, figsize=(24, ab_y_axis),
                             gridspec_kw={"width_ratios": [0.9, 1.4, 1.0, 1.4]})

    # Panel 1: Venn
    ax = axes[0]
    draw_proportional_venn(ax, n_g1, n_g2, n_ab, C["g1"], C["g2"], C["both"],
                           title=f"Significant Genes ({PVAL_DISP_NAME} < {SIG})",
                           logfc_thresh=logfc_thresh)
    _save_venn_genes(outdir, f"{os.path.splitext(filename)[0]}", {
        "G1_Only": df.loc[df["sig_cat"] == "G1 Only",     "Gene"].tolist(),
        "G2_Only": df.loc[df["sig_cat"] == "G2 Only",     "Gene"].tolist(),
        "Both":    df.loc[df["sig_cat"] == "Sig in Both", "Gene"].tolist(),
    })

    # Panel 2: Bland-Altman
    ax = axes[1]
    mu_ba = df["delta_logFC"].mean()
    sd_ba = df["delta_logFC"].std()
    lo_ba, hi_ba = mu_ba - 1.96 * sd_ba, mu_ba + 1.96 * sd_ba
    colored_mask_BA = df["sig_cat"] != "Neither"
    y_margin = float(df.loc[colored_mask_BA, "delta_logFC"].abs().max()) * 1.15 \
               if colored_mask_BA.sum() else 0.5
    x_margin = float(df.loc[colored_mask_BA, "mean_logFC"].abs().max()) * 1.5 \
               if colored_mask_BA.sum() else 0.5
    if ba_xlim is not None:
        xlim_BA = (-ba_xlim, ba_xlim)
    else:
        xlim_BA = (-x_margin, x_margin)
    if ba_ylim is not None:
        ylim_BA = (-ba_ylim, ba_ylim)
    else:
        ylim_BA = (-y_margin, y_margin)
    scatter_grey_rect(ax, df.loc[~colored_mask_BA, "mean_logFC"],
                      df.loc[~colored_mask_BA, "delta_logFC"], xlim_BA, ylim_BA,
                      label=f"Neither (n={(~colored_mask_BA).sum()}, clipped)")
    labeled_genes_BA = df["Gene"].isin(top25_genes)
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        mask = df["sig_cat"] == cat
        if mask.sum():
            ax.scatter(df.loc[mask & ~labeled_genes_BA, "mean_logFC"],
                       df.loc[mask & ~labeled_genes_BA, "delta_logFC"],
                       color=col, alpha=GREY_ALPHA, s=GREY_SIZE, edgecolors="none",
                       zorder=4, label=f"{cat} (n={mask.sum()})")
            if (mask & labeled_genes_BA).sum():
                ax.scatter(df.loc[mask & labeled_genes_BA, "mean_logFC"],
                           df.loc[mask & labeled_genes_BA, "delta_logFC"],
                           color=col, alpha=0.92, s=24, edgecolors="none", zorder=5)
    ax.axhline(hi_ba, color="firebrick", lw=1.2, ls="--", label="95% limits of agreement")
    ax.axhline(lo_ba, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="#555", lw=0.8, ls=":", alpha=0.7)
    ax.annotate("G1 > G2", xy=(xlim_BA[1] * 0.55, lo_ba * 0.6),
                fontsize=11, color=C["g1"], style="italic")
    ax.annotate("G2 > G1", xy=(xlim_BA[1] * 0.55, hi_ba * 0.55),
                fontsize=11, color=C["g2"], style="italic")
    ax.set_xlim(*xlim_BA); ax.set_ylim(*ylim_BA)
    ax.set_xlabel("Mean logFC  [(G1 + G2) / 2]")
    ax.set_ylabel("\u0394 logFC  [G2 \u2212 G1]\n(positive = G2 has larger effect)")
    ax.set_title(f"Bland-Altman\n"
                 f"All Genes: MAB={abs_bias_all:.3f} (% of |logFC|: {mab_pct_all:.1f}%)\n"
                 f"Top 10%: MAB={top10_mab_str} (% of |logFC|: {top10_mab_pct_str})",
                 fontsize=14)
    ax.legend(fontsize=10.5, framealpha=0.9, loc="upper right")
    texts_BA = []
    sub_BA_top25 = df[colored_mask_BA & df["Gene"].isin(top25_genes)].copy()
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        texts_BA.extend(label_all(ax, sub_BA_top25[sub_BA_top25["sig_cat"] == cat],
                                  "mean_logFC", "delta_logFC", col=col))
    if texts_BA:
        safe_adjust(texts_BA, ax, force_text=(0.3, 1.5), force_static=(0.2, 0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))

    # Panel 3: Dumbbell
    ax = axes[2]
    sig_r     = sig_genes_df.reset_index(drop=True)
    ys_db     = list(range(len(sig_r)))
    sig_col_map = {"G1 Only": C["g1"], "G2 Only": C["g2"], "Sig in Both": C["both"]}
    for yi, (_, row) in zip(ys_db, sig_r.iterrows()):
        ax.plot([row["logFC_up"], row["logFC_down"]], [yi, yi],
                color=sig_col_map.get(row["sig_cat"], "#888"), lw=1.4, zorder=2, alpha=0.55)
    for cat, col in [("Sig in Both", C["both"]), ("G1 Only", C["g1"]), ("G2 Only", C["g2"])]:
        mask = sig_r["sig_cat"] == cat
        if not mask.any():
            continue
        ys_cat = sig_r.index[mask].tolist()
        ax.scatter(sig_r.loc[mask, "logFC_up"], ys_cat,
                   facecolors="none", edgecolors=col, s=55, marker="o",
                   linewidths=1.4, zorder=4, label=cat)
        ax.scatter(sig_r.loc[mask, "logFC_down"], ys_cat,
                   color=col, s=55, marker="o", zorder=4, edgecolors="none")
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_yticks(ys_db)
    ax.set_yticklabels(sig_r["Gene"].values, fontsize=11, fontweight="bold")
    ax.set_xlabel("logFC")
    _mab = sig_r["delta_logFC"].abs().mean()
    _avg_mag = ((sig_r["logFC_up"].abs() + sig_r["logFC_down"].abs()) / 2).mean()
    _mab_pct = _mab / _avg_mag * 100 if _avg_mag > 0 else np.nan
    ax.set_title(f"G1 vs G2 logFC\n(Significant genes only, n={len(sig_r)})\n"
                 f"MAB={_mab:.3f} (% of |logFC|: {_mab_pct:.1f}%)", fontsize=14)
    leg_sig = [mpatches.Patch(color=sig_col_map[c], label=c)
               for c in ["Sig in Both", "G1 Only", "G2 Only"]
               if (sig_r["sig_cat"] == c).any()]
    leg_shape = [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                        markeredgecolor="#555", markeredgewidth=1.4,
                        markersize=7, label="\u25cb G1"),
                 Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                        markersize=7, label="\u25cf G2")]
    ax.legend(handles=leg_sig + leg_shape, fontsize=10.5, framealpha=0.9)

    # Panel 4: -log10(p) scatter
    ax = axes[3]
    cx2 = df.loc[colored, "nla_up"]; cy2 = df.loc[colored, "nla_down"]
    if nla_lim is not None:
        lm = nla_lim
    else:
        lm = max(float(cx2.max()) if len(cx2) > 0 else 0.5,
                 float(cy2.max()) if len(cy2) > 0 else 0.5) * 1.12
    scatter_grey_rect(ax, df.loc[grey, "nla_up"], df.loc[grey, "nla_down"],
                      (0, lm), (0, lm), label=f"Not significant (n={grey.sum()})")
    for cat, col, lbl in [("G1 Only", C["g1"], "Sig: G1 only"),
                           ("G2 Only", C["g2"], "Sig: G2 only"),
                           ("Sig in Both", C["both"], "Sig: Both")]:
        m = df["sig_cat"] == cat
        if m.sum():
            ax.scatter(df.loc[m, "nla_up"], df.loc[m, "nla_down"],
                       color=col, alpha=1.0, s=24, edgecolors="none",
                       zorder=5, label=f"{lbl} (n={m.sum()})")
    ax.plot([0, lm], [0, lm], "k--", lw=0.9, alpha=0.5, label="y = x")
    sl = -np.log10(SIG)
    ax.axhline(sl, color=C["g2"], lw=1, ls=":", alpha=0.8)
    ax.axvline(sl, color=C["g1"], lw=1, ls=":", alpha=0.8)
    ax.set_xlim(0, lm); ax.set_ylim(0, lm)
    ax.set_xlabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})  [G1]")
    ax.set_ylabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})  [G2]")
    _r_nla_top10_str = f"{r_nla_sp_top10:.3f}" if not np.isnan(r_nla_sp_top10) else "n/a"
    ax.set_title(f"Significance Concordance\n"
                 f"Spearman r = {r_nla_sp:.3f} (all)  |  {_r_nla_top10_str} (top 10%)",
                 fontsize=14)
    ax.legend(markerscale=1.1, fontsize=10.5, framealpha=0.88, loc="upper right")
    thresh_nla = 2.3
    texts_nla = []
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        sub = df[(df["sig_cat"] == cat) &
                 ((df["nla_up"] > thresh_nla) | (df["nla_down"] > thresh_nla) |
                  df["Gene"].isin(top25_genes))]
        texts_nla.extend(label_all(ax, sub, "nla_up", "nla_down", col=col))
    if texts_nla:
        safe_adjust(texts_nla, ax, force_text=(0.3, 1.5), force_static=(0.2, 0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_figures(df, outdir, tag, ba_ylim=None, ba_xlim=None, nla_lim=None, merged=None, ab_y_axis=AB_Y_DEFAULT):
    """Generate Figures A–E for a single dataset and save to outdir."""
    os.makedirs(outdir, exist_ok=True)

    cats = df["sig_cat"].value_counts().reindex(
        ["Sig in Both", "G1 Only", "G2 Only", "Neither"], fill_value=0)
    n_g1   = int(cats["G1 Only"])
    n_g2   = int(cats["G2 Only"])
    n_ab   = int(cats["Sig in Both"])
    n_flip = int(df["sign_flip"].sum())

    r_lfc_sp, _ = stats.spearmanr(df["logFC_up"], df["logFC_down"])
    r_lfc_pe, _ = stats.pearsonr(df["logFC_up"],  df["logFC_down"])
    r_nla_sp, _ = stats.spearmanr(df["nla_up"],   df["nla_down"])

    # Top-10% by mean |logFC| magnitude — used in BA title stats
    magnitude    = (df["logFC_up"].abs() + df["logFC_down"].abs()) / 2
    top10_mask   = magnitude >= magnitude.quantile(0.90)
    top10_df_ba  = df[top10_mask]
    abs_bias_all = np.abs(df["delta_logFC"]).mean()
    if len(top10_df_ba) > 1:
        r_lfc_pe_top10, _ = stats.pearsonr(top10_df_ba["logFC_up"],
                                           top10_df_ba["logFC_down"])
        abs_bias_top10 = np.abs(top10_df_ba["delta_logFC"]).mean()
        r_nla_sp_top10, _ = stats.spearmanr(top10_df_ba["nla_up"],
                                             top10_df_ba["nla_down"])
    else:
        r_lfc_pe_top10, abs_bias_top10, r_nla_sp_top10 = np.nan, np.nan, np.nan

    # MAB as % of mean absolute logFC (mean of (|logFC_up|+|logFC_down|)/2)
    mean_abs_logfc_all   = magnitude.mean()
    mean_abs_logfc_top10 = magnitude[top10_mask].mean() if top10_mask.sum() > 0 else np.nan
    mab_pct_all  = abs_bias_all / mean_abs_logfc_all * 100 if mean_abs_logfc_all > 0 else np.nan
    mab_pct_top10 = (abs_bias_top10 / mean_abs_logfc_top10 * 100
                     if (not np.isnan(abs_bias_top10) and mean_abs_logfc_top10 > 0) else np.nan)

    colored = df["sig_cat"] != "Neither"
    grey    = ~colored

    # ── FIGURE A — Venn + logFC scatter + adjP scatter ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5),
                              gridspec_kw={"width_ratios": [1.1, 1.6, 1.6]})

    # A1: Venn
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(plt.Circle((3.2, 3.5), 2.5, color=C["g1"], alpha=0.35))
    ax.add_patch(plt.Circle((6.8, 3.5), 2.5, color=C["g2"], alpha=0.35))
    ax.text(2.1, 3.5, str(n_g1), ha="center", va="center",
            fontsize=38, fontweight="bold", color=C["g1"])
    ax.text(7.9, 3.5, str(n_g2), ha="center", va="center",
            fontsize=38, fontweight="bold", color=C["g2"])
    overlap_col = C["both"] if n_ab > 0 else "#888"
    t_overlap = ax.text(5.0, 3.5, str(n_ab), ha="center", va="center",
                        fontsize=38, fontweight="bold", color=overlap_col)
    if n_ab > 0:
        t_overlap.set_path_effects([withStroke(linewidth=3, foreground="white")])
    ax.text(2.0, 0.6, "G1\n(Stroke Cortex\u2191)", ha="center", fontsize=16,
            fontweight="bold", color=C["g1"])
    ax.text(8.0, 0.6, "G2\n(Healthy Cortex\u2191)", ha="center", fontsize=16,
            fontweight="bold", color=C["g2"])
    ax.set_title(f"Significant Genes ({PVAL_DISP_NAME} < {SIG})", fontsize=17, pad=8)
    _save_venn_genes(outdir, "figA_unadj", {
        "G1_Only": df.loc[df["sig_cat"] == "G1 Only",     "Gene"].tolist(),
        "G2_Only": df.loc[df["sig_cat"] == "G2 Only",     "Gene"].tolist(),
        "Both":    df.loc[df["sig_cat"] == "Sig in Both", "Gene"].tolist(),
    })

    # A2: Bland-Altman (x = logFC_up, y = delta_logFC, y-axis shared per cell type)
    ax = axes[1]
    mu_ba = df["delta_logFC"].mean()
    sd_ba = df["delta_logFC"].std()
    lo_ba, hi_ba = mu_ba - 1.96 * sd_ba, mu_ba + 1.96 * sd_ba

    x_margin_A2 = float(df.loc[colored, "mean_logFC"].abs().max()) * 1.5 \
                  if colored.sum() else 0.5
    if ba_xlim is not None:
        xlim_A2 = (-ba_xlim, ba_xlim)
    else:
        xlim_A2 = (-x_margin_A2, x_margin_A2)
    if ba_ylim is not None:
        ylim_A2 = (-ba_ylim, ba_ylim)
    else:
        y_margin_A2 = float(df.loc[colored, "delta_logFC"].abs().max()) * 1.20 \
                      if colored.sum() else 0.5
        ylim_A2 = (-y_margin_A2, y_margin_A2)

    scatter_grey_rect(ax, df.loc[grey, "mean_logFC"], df.loc[grey, "delta_logFC"],
                      xlim_A2, ylim_A2,
                      label=f"Neither (n={grey.sum()}, clipped)")
    for cat, col, lbl in [("G1 Only", C["g1"], "G1 only"),
                           ("G2 Only", C["g2"], "G2 only"),
                           ("Sig in Both", C["both"], "Both")]:
        m = df["sig_cat"] == cat
        if m.sum():
            ax.scatter(df.loc[m, "mean_logFC"], df.loc[m, "delta_logFC"],
                       color=col, alpha=0.92, s=24, edgecolors="none",
                       zorder=5, label=f"{lbl} (n={m.sum()})")
    ax.axhline(hi_ba, color="firebrick", lw=1.2, ls="--", label="95% LoA")
    ax.axhline(lo_ba, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="#555", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(*xlim_A2); ax.set_ylim(*ylim_A2)
    ax.set_xlabel("Mean logFC  [(G1 + G2) / 2]")
    ax.set_ylabel("\u0394 logFC  [G2 \u2212 G1]")
    top10_mab_str     = f"{abs_bias_top10:.3f}" if not np.isnan(abs_bias_top10) else "n/a"
    top10_mab_pct_str = f"{mab_pct_top10:.1f}%" if not np.isnan(mab_pct_top10) else "n/a"
    ax.set_title(f"Bland-Altman\n"
                 f"All Genes: MAB={abs_bias_all:.3f} (% of |logFC|: {mab_pct_all:.1f}%)\n"
                 f"Top 10%: MAB={top10_mab_str} (% of |logFC|: {top10_mab_pct_str})",
                 fontsize=14)
    ax.legend(markerscale=1.1, fontsize=10.5, framealpha=0.88, loc="upper right")
    texts_A2 = []
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        sub = df[(df["sig_cat"] == cat) & (df["mean_logFC"].abs() > 0.2)]
        texts_A2.extend(label_all(ax, sub, "mean_logFC", "delta_logFC", col=col))
    if texts_A2:
        safe_adjust(texts_A2, ax, force_text=(0.3, 1.5), force_static=(0.2, 0.8),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#555"))

    # A3: adjP scatter (both axes shared per cell type via nla_lim)
    ax = axes[2]
    cx2 = df.loc[colored, "nla_up"]; cy2 = df.loc[colored, "nla_down"]
    if nla_lim is not None:
        lm = nla_lim
    else:
        lm = max(float(cx2.max()) if len(cx2) > 0 else 0.5,
                 float(cy2.max()) if len(cy2) > 0 else 0.5) * 1.12
    xl2 = (0, lm); yl2 = (0, lm)
    scatter_grey_rect(ax, df.loc[grey, "nla_up"], df.loc[grey, "nla_down"],
                      xl2, yl2, label=f"Not significant (n={grey.sum()})")
    for cat, col, lbl in [("G1 Only", C["g1"], "Sig: G1 only"),
                           ("G2 Only", C["g2"], "Sig: G2 only"),
                           ("Sig in Both", C["both"], "Sig: Both")]:
        m = df["sig_cat"] == cat
        if m.sum():
            ax.scatter(df.loc[m, "nla_up"], df.loc[m, "nla_down"],
                       color=col, alpha=1.0, s=24, edgecolors="none",
                       zorder=5, label=f"{lbl} (n={m.sum()})")
    ax.plot([0, lm], [0, lm], "k--", lw=0.9, alpha=0.5, label="y = x")
    sl = -np.log10(SIG)
    ax.axhline(sl, color=C["g2"], lw=1, ls=":", alpha=0.8)
    ax.axvline(sl, color=C["g1"], lw=1, ls=":", alpha=0.8)
    ax.set_xlim(*xl2); ax.set_ylim(*yl2)
    ax.set_xlabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})  [G1]")
    ax.set_ylabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})  [G2]")
    _r_nla_top10_str = f"{r_nla_sp_top10:.3f}" if not np.isnan(r_nla_sp_top10) else "n/a"
    ax.set_title(f"Significance Concordance\n"
                 f"Spearman r = {r_nla_sp:.3f} (all)  |  {_r_nla_top10_str} (top 10%)",
                 fontsize=14)
    ax.legend(markerscale=1.1, fontsize=10.5, framealpha=0.88, loc="upper right")
    thresh_A3 = 2.3
    texts_A3 = []
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        sub = df[(df["sig_cat"] == cat) &
                 ((df["nla_up"] > thresh_A3) | (df["nla_down"] > thresh_A3))]
        texts_A3.extend(label_all(ax, sub, "nla_up", "nla_down", col=col))
    if texts_A3:
        safe_adjust(texts_A3, ax,
                        force_text=(0.3, 1.5), force_static=(0.2, 0.8),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figA_divergence.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── FIGURE A (adj-p Venn) — Venn diagram using adj.P.Val ────────────────
    adj_up_col   = "adj.P.Val_up"
    adj_down_col = "adj.P.Val_down"
    if adj_up_col in df.columns and adj_down_col in df.columns:
        sig_up_adj   = df[adj_up_col]   < SIG
        sig_down_adj = df[adj_down_col] < SIG
        n_g1_adj = int((sig_up_adj & ~sig_down_adj).sum())
        n_g2_adj = int((sig_down_adj & ~sig_up_adj).sum())
        n_ab_adj = int((sig_up_adj & sig_down_adj).sum())

        fig_v, ax_v = plt.subplots(1, 1, figsize=(5.5, 5))
        draw_proportional_venn(ax_v, n_g1_adj, n_g2_adj, n_ab_adj,
                               C["g1"], C["g2"], C["both"],
                               title=f"Significant Genes (adj.P < {SIG})")
        _save_venn_genes(outdir, "figA_adjp", {
            "G1_Only": df.loc[sig_up_adj & ~sig_down_adj, "Gene"].tolist(),
            "G2_Only": df.loc[sig_down_adj & ~sig_up_adj, "Gene"].tolist(),
            "Both":    df.loc[sig_up_adj & sig_down_adj,  "Gene"].tolist(),
        })

        fig_v.tight_layout()
        fig_v.savefig(os.path.join(outdir, "figA_venn_adjp.png"), dpi=160, bbox_inches="tight")
        plt.close(fig_v)

    # ── FIGURE B — Bland-Altman (left) + Dumbbell (right) ───────────────────
    plt.rcParams.update({
        "font.size": 28, "axes.titlesize": 31, "axes.labelsize": 28,
        "legend.fontsize": 22, "xtick.labelsize": 25, "ytick.labelsize": 25,
    })
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              gridspec_kw={"width_ratios": [1.1, 1.0]})

    # Top 25 significant genes by |delta_logFC| for B2 dumbbell
    sig_pool     = df[df["sig_cat"] != "Neither"]
    sig_genes_df = sig_pool.reindex(
        sig_pool["delta_logFC"].abs().nlargest(25).index
    ).sort_values("delta_logFC")
    top25_genes  = set(sig_genes_df["Gene"])

    # B1: Bland-Altman — x-axis = logFC_up
    ax = axes[0]
    mu = df["delta_logFC"].mean()
    sd = df["delta_logFC"].std()
    lo_ba, hi_ba = mu - 1.96 * sd, mu + 1.96 * sd

    colored_mask_B1 = df["sig_cat"] != "Neither"
    y_margin = float(df.loc[colored_mask_B1, "delta_logFC"].abs().max()) * 1.15 \
               if colored_mask_B1.sum() else 0.5
    x_margin = float(df.loc[colored_mask_B1, "mean_logFC"].abs().max()) * 1.5 \
               if colored_mask_B1.sum() else 0.5
    if ba_xlim is not None:
        xlim_B1 = (-ba_xlim, ba_xlim)
    else:
        xlim_B1 = (-x_margin, x_margin)
    if ba_ylim is not None:
        ylim_B1 = (-ba_ylim, ba_ylim)
    else:
        ylim_B1 = (-y_margin, y_margin)

    scatter_grey_rect(ax,
                      df.loc[~colored_mask_B1, "mean_logFC"],
                      df.loc[~colored_mask_B1, "delta_logFC"],
                      xlim_B1, ylim_B1,
                      label=f"Neither (n={(~colored_mask_B1).sum()}, clipped)")

    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]),
                     ("Sig in Both", C["both"])]:
        mask = df["sig_cat"] == cat
        if mask.sum():
            ax.scatter(df.loc[mask, "mean_logFC"], df.loc[mask, "delta_logFC"],
                       color=col, alpha=0.92, s=24, edgecolors="none",
                       zorder=5, label=f"{cat} (n={mask.sum()})")

    ax.axhline(hi_ba, color="firebrick", lw=1.2, ls="--", label="95% limits of agreement")
    ax.axhline(lo_ba, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="#555", lw=0.8, ls=":", alpha=0.7)

    ax.annotate("G1 > G2", xy=(x_margin * 0.55, lo_ba * 0.6),
                fontsize=22, color=C["g1"], style="italic")
    ax.annotate("G2 > G1", xy=(x_margin * 0.55, hi_ba * 0.55),
                fontsize=22, color=C["g2"], style="italic")

    ax.set_xlim(*xlim_B1)
    ax.set_ylim(*ylim_B1)
    ax.set_xlabel("Mean logFC  [(G1 + G2) / 2]")
    ax.set_ylabel("\u0394 logFC  [G2 \u2212 G1]\n(positive = G2 has larger effect)")
    ax.set_title(f"Bland-Altman\n"
                 f"All Genes: MAB={abs_bias_all:.3f} (% of |logFC|: {mab_pct_all:.1f}%)\n"
                 f"Top 10%: MAB={top10_mab_str} (% of |logFC|: {top10_mab_pct_str})",
                 fontsize=25)
    ax.legend(fontsize=22, framealpha=0.9, loc="upper right")

    texts_B1 = []
    sub_B1_top25 = df[colored_mask_B1 & df["Gene"].isin(top25_genes)].copy()
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]),
                     ("Sig in Both", C["both"])]:
        texts_B1.extend(label_all(ax, sub_B1_top25[sub_B1_top25["sig_cat"] == cat],
                                   "mean_logFC", "delta_logFC", col=col, fontsize=22))
    if texts_B1:
        safe_adjust(texts_B1, ax,
                        force_text=(0.3, 1.5), force_static=(0.2, 0.8),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))

    # B2: Dumbbell — significant genes only, colored by sig_cat
    ax = axes[1]
    sig_r  = sig_genes_df.reset_index(drop=True)
    ys_db  = list(range(len(sig_r)))

    sig_col_map = {
        "G1 Only":     C["g1"],
        "G2 Only":     C["g2"],
        "Sig in Both": C["both"],
    }

    # Connecting lines colored by sig_cat
    for yi, (_, row) in zip(ys_db, sig_r.iterrows()):
        ax.plot([row["logFC_up"], row["logFC_down"]], [yi, yi],
                color=sig_col_map.get(row["sig_cat"], "#888"),
                lw=1.4, zorder=2, alpha=0.55)

    # Dots grouped by sig_cat
    for cat, col in [("Sig in Both", C["both"]), ("G1 Only", C["g1"]),
                     ("G2 Only", C["g2"])]:
        mask = sig_r["sig_cat"] == cat
        if not mask.any():
            continue
        ys_cat = sig_r.index[mask].tolist()
        # G1 — hollow circle
        ax.scatter(sig_r.loc[mask, "logFC_up"], ys_cat,
                   facecolors="none", edgecolors=col, s=55, marker="o",
                   linewidths=1.4, zorder=4, label=cat)
        # G2 — filled circle (no extra legend entry)
        ax.scatter(sig_r.loc[mask, "logFC_down"], ys_cat,
                   color=col, s=55, marker="o", zorder=4, edgecolors="none")

    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_yticks(ys_db)
    ax.set_yticklabels(sig_r["Gene"].values, fontsize=24, fontweight="bold")
    ax.set_xlabel("logFC")
    _mab = sig_r["delta_logFC"].abs().mean()
    _avg_mag = ((sig_r["logFC_up"].abs() + sig_r["logFC_down"].abs()) / 2).mean()
    _mab_pct = _mab / _avg_mag * 100 if _avg_mag > 0 else np.nan
    ax.set_title(f"G1 vs G2 logFC\n(Significant genes only, n={len(sig_r)})\n"
                 f"MAB={_mab:.3f} (% of |logFC|: {_mab_pct:.1f}%)",
                 fontsize=25)

    leg_sig = [mpatches.Patch(color=sig_col_map[c], label=c)
               for c in ["Sig in Both", "G1 Only", "G2 Only"]
               if (sig_r["sig_cat"] == c).any()]
    leg_shape = [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                        markeredgecolor="#555", markeredgewidth=1.4,
                        markersize=7, label="\u25cb G1"),
                 Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                        markersize=7, label="\u25cf G2")]
    ax.legend(handles=leg_sig + leg_shape, fontsize=22, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figB_effect_size_shift.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
        "legend.fontsize": 13, "xtick.labelsize": 15, "ytick.labelsize": 15,
    })

    # ── FIGURES A+B COMBINED — Venn | Bland-Altman | Top-25 Dumbbell | -log10(p) scatter ──
    fig, axes = plt.subplots(1, 4, figsize=(24, ab_y_axis),
                             gridspec_kw={"width_ratios": [0.9, 1.4, 1.0, 1.4]})

    # Panel 1: Venn
    ax = axes[0]
    draw_proportional_venn(ax, n_g1, n_g2, n_ab, C["g1"], C["g2"], C["both"],
                           title=f"Significant Genes ({PVAL_DISP_NAME} < {SIG})")
    _save_venn_genes(outdir, "figAB_combined_unadj", {
        "G1_Only": df.loc[df["sig_cat"] == "G1 Only",     "Gene"].tolist(),
        "G2_Only": df.loc[df["sig_cat"] == "G2 Only",     "Gene"].tolist(),
        "Both":    df.loc[df["sig_cat"] == "Sig in Both", "Gene"].tolist(),
    })

    # Panel 2: Bland-Altman (with top25 labels)
    ax = axes[1]
    mu_ba_AB = df["delta_logFC"].mean()
    sd_ba_AB = df["delta_logFC"].std()
    lo_ba_AB, hi_ba_AB = mu_ba_AB - 1.96 * sd_ba_AB, mu_ba_AB + 1.96 * sd_ba_AB
    colored_mask_AB = df["sig_cat"] != "Neither"
    y_margin_AB = float(df.loc[colored_mask_AB, "delta_logFC"].abs().max()) * 1.15 \
                  if colored_mask_AB.sum() else 0.5
    x_margin_AB = float(df.loc[colored_mask_AB, "mean_logFC"].abs().max()) * 1.5 \
                  if colored_mask_AB.sum() else 0.5
    if ba_xlim is not None:
        xlim_AB = (-ba_xlim, ba_xlim)
    else:
        xlim_AB = (-x_margin_AB, x_margin_AB)
    if ba_ylim is not None:
        ylim_AB = (-ba_ylim, ba_ylim)
    else:
        ylim_AB = (-y_margin_AB, y_margin_AB)
    scatter_grey_rect(ax, df.loc[~colored_mask_AB, "mean_logFC"],
                      df.loc[~colored_mask_AB, "delta_logFC"], xlim_AB, ylim_AB,
                      label=f"Neither (n={(~colored_mask_AB).sum()}, clipped)")
    labeled_genes_AB = df["Gene"].isin(top25_genes)
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        mask = df["sig_cat"] == cat
        if mask.sum():
            ax.scatter(df.loc[mask & ~labeled_genes_AB, "mean_logFC"],
                       df.loc[mask & ~labeled_genes_AB, "delta_logFC"],
                       color=col, alpha=GREY_ALPHA, s=GREY_SIZE, edgecolors="none",
                       zorder=4, label=f"{cat} (n={mask.sum()})")
            if (mask & labeled_genes_AB).sum():
                ax.scatter(df.loc[mask & labeled_genes_AB, "mean_logFC"],
                           df.loc[mask & labeled_genes_AB, "delta_logFC"],
                           color=col, alpha=0.92, s=24, edgecolors="none", zorder=5)
    ax.axhline(hi_ba_AB, color="firebrick", lw=1.2, ls="--", label="95% limits of agreement")
    ax.axhline(lo_ba_AB, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="#555", lw=0.8, ls=":", alpha=0.7)
    ax.annotate("G1 > G2", xy=(x_margin_AB * 0.55, lo_ba_AB * 0.6),
                fontsize=11, color=C["g1"], style="italic")
    ax.annotate("G2 > G1", xy=(x_margin_AB * 0.55, hi_ba_AB * 0.55),
                fontsize=11, color=C["g2"], style="italic")
    ax.set_xlim(*xlim_AB); ax.set_ylim(*ylim_AB)
    ax.set_xlabel("Mean logFC  [(G1 + G2) / 2]")
    ax.set_ylabel("\u0394 logFC  [G2 \u2212 G1]\n(positive = G2 has larger effect)")
    ax.set_title(f"Bland-Altman\n"
                 f"All Genes: MAB={abs_bias_all:.3f} (% of |logFC|: {mab_pct_all:.1f}%)\n"
                 f"Top 10%: MAB={top10_mab_str} (% of |logFC|: {top10_mab_pct_str})",
                 fontsize=14)
    ax.legend(fontsize=10.5, framealpha=0.9, loc="upper right")
    texts_AB_ba = []
    sub_AB_top25 = df[colored_mask_AB & df["Gene"].isin(top25_genes)].copy()
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        texts_AB_ba.extend(label_all(ax, sub_AB_top25[sub_AB_top25["sig_cat"] == cat],
                                     "mean_logFC", "delta_logFC", col=col))
    if texts_AB_ba:
        safe_adjust(texts_AB_ba, ax, force_text=(0.3, 1.5), force_static=(0.2, 0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))

    # Panel 3: Dumbbell
    ax = axes[2]
    sig_r_AB  = sig_genes_df.reset_index(drop=True)
    ys_db_AB  = list(range(len(sig_r_AB)))
    sig_col_map_AB = {"G1 Only": C["g1"], "G2 Only": C["g2"], "Sig in Both": C["both"]}
    for yi, (_, row) in zip(ys_db_AB, sig_r_AB.iterrows()):
        ax.plot([row["logFC_up"], row["logFC_down"]], [yi, yi],
                color=sig_col_map_AB.get(row["sig_cat"], "#888"), lw=1.4, zorder=2, alpha=0.55)
    for cat, col in [("Sig in Both", C["both"]), ("G1 Only", C["g1"]), ("G2 Only", C["g2"])]:
        mask = sig_r_AB["sig_cat"] == cat
        if not mask.any(): continue
        ys_cat = sig_r_AB.index[mask].tolist()
        ax.scatter(sig_r_AB.loc[mask, "logFC_up"], ys_cat,
                   facecolors="none", edgecolors=col, s=55, marker="o",
                   linewidths=1.4, zorder=4, label=cat)
        ax.scatter(sig_r_AB.loc[mask, "logFC_down"], ys_cat,
                   color=col, s=55, marker="o", zorder=4, edgecolors="none")
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_yticks(ys_db_AB)
    ax.set_yticklabels(sig_r_AB["Gene"].values, fontsize=11, fontweight="bold")
    ax.set_xlabel("logFC")
    _mab_AB = sig_r_AB["delta_logFC"].abs().mean()
    _avg_mag_AB = ((sig_r_AB["logFC_up"].abs() + sig_r_AB["logFC_down"].abs()) / 2).mean()
    _mab_pct_AB = _mab_AB / _avg_mag_AB * 100 if _avg_mag_AB > 0 else np.nan
    ax.set_title(f"G1 vs G2 logFC\n(Significant genes only, n={len(sig_r_AB)})\n"
                 f"MAB={_mab_AB:.3f} (% of |logFC|: {_mab_pct_AB:.1f}%)",
                 fontsize=14)
    leg_sig_AB = [mpatches.Patch(color=sig_col_map_AB[c], label=c)
                  for c in ["Sig in Both", "G1 Only", "G2 Only"]
                  if (sig_r_AB["sig_cat"] == c).any()]
    leg_shape_AB = [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                           markeredgecolor="#555", markeredgewidth=1.4,
                           markersize=7, label="\u25cb G1"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                           markersize=7, label="\u25cf G2")]
    ax.legend(handles=leg_sig_AB + leg_shape_AB, fontsize=10.5, framealpha=0.9)

    # Panel 4: -log10(p) scatter
    ax = axes[3]
    colored_AB = df["sig_cat"] != "Neither"
    grey_AB    = ~colored_AB
    cx2_AB = df.loc[colored_AB, "nla_up"]; cy2_AB = df.loc[colored_AB, "nla_down"]
    if nla_lim is not None:
        lm_AB = nla_lim
    else:
        lm_AB = max(float(cx2_AB.max()) if len(cx2_AB) > 0 else 0.5,
                    float(cy2_AB.max()) if len(cy2_AB) > 0 else 0.5) * 1.12
    scatter_grey_rect(ax, df.loc[grey_AB, "nla_up"], df.loc[grey_AB, "nla_down"],
                      (0, lm_AB), (0, lm_AB), label=f"Not significant (n={grey_AB.sum()})")
    for cat, col, lbl in [("G1 Only", C["g1"], "Sig: G1 only"),
                           ("G2 Only", C["g2"], "Sig: G2 only"),
                           ("Sig in Both", C["both"], "Sig: Both")]:
        m = df["sig_cat"] == cat
        if m.sum():
            ax.scatter(df.loc[m, "nla_up"], df.loc[m, "nla_down"],
                       color=col, alpha=1.0, s=24, edgecolors="none",
                       zorder=5, label=f"{lbl} (n={m.sum()})")
    ax.plot([0, lm_AB], [0, lm_AB], "k--", lw=0.9, alpha=0.5, label="y = x")
    sl_AB = -np.log10(SIG)
    ax.axhline(sl_AB, color=C["g2"], lw=1, ls=":", alpha=0.8)
    ax.axvline(sl_AB, color=C["g1"], lw=1, ls=":", alpha=0.8)
    ax.set_xlim(0, lm_AB); ax.set_ylim(0, lm_AB)
    ax.set_xlabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})  [G1]")
    ax.set_ylabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})  [G2]")
    _r_nla_top10_str = f"{r_nla_sp_top10:.3f}" if not np.isnan(r_nla_sp_top10) else "n/a"
    ax.set_title(f"Significance Concordance\n"
                 f"Spearman r = {r_nla_sp:.3f} (all)  |  {_r_nla_top10_str} (top 10%)",
                 fontsize=14)
    ax.legend(markerscale=1.1, fontsize=10.5, framealpha=0.88, loc="upper right")
    thresh_AB = 2.3
    texts_AB_nla = []
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        sub = df[(df["sig_cat"] == cat) &
                 ((df["nla_up"] > thresh_AB) | (df["nla_down"] > thresh_AB) |
                  df["Gene"].isin(top25_genes))]
        texts_AB_nla.extend(label_all(ax, sub, "nla_up", "nla_down", col=col))
    if texts_AB_nla:
        safe_adjust(texts_AB_nla, ax, force_text=(0.3, 1.5), force_static=(0.2, 0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figAB_combined.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── Additional figAB with logFC threshold filters ─────────────────────────
    if merged is not None:
        for _thresh, _suffix in [(0.10, "logfc010")]:
            _df = prepare_df(merged, logfc_thresh=_thresh)
            _plot_figAB_combined(_df, outdir, f"figAB_combined_{_suffix}.png",
                                 ba_ylim=ba_ylim, ba_xlim=ba_xlim,
                                 nla_lim=nla_lim, logfc_thresh=_thresh,
                                 ab_y_axis=ab_y_axis)
    fig, axes_v = plt.subplots(1, 2, figsize=(16, ab_y_axis))

    def _volcano(ax, lfc_col, padj_col, own_sig, other_sig, own_col, other_col,
                 title, other_label):
        lfc = df[lfc_col]; nlp = -np.log10(df[padj_col].clip(1e-300))
        ns = ~own_sig; up = own_sig & (lfc > 0); dn = own_sig & (lfc < 0)
        cv = own_sig | other_sig
        if cv.sum() > 0:
            xh = max(float(lfc[cv].abs().max()) * 1.5, 0.15)
            yt = float(nlp[cv].max()) * 1.45
        else:
            xh = max(float(lfc.abs().max()) * 1.1, 0.15)
            yt = max(float(nlp.max()) * 1.1, 1.0)
        scatter_grey_rect(ax, lfc[ns], nlp[ns], (-xh, xh), (0, yt),
                          label="Not significant")
        ax.scatter(lfc[up], nlp[up], color=C["up"], alpha=0.88, s=60,
                   edgecolors="none", label=f"Upregulated (n={up.sum()})", zorder=4)
        ax.scatter(lfc[dn], nlp[dn], color=C["down"], alpha=0.88, s=60,
                   edgecolors="none", label=f"Downregulated (n={dn.sum()})", zorder=4)
        oo = other_sig & ~own_sig
        ax.scatter(lfc[oo], nlp[oo], color=other_col, alpha=0.92, s=120, marker="D",
                   edgecolors="none",
                   label=f"Sig in {other_label} (n={oo.sum()})", zorder=5)
        ax.axhline(-np.log10(SIG), color="#666", lw=0.9, ls="--", alpha=0.7,
                   label=f"{PVAL_DISP_NAME} = {SIG}")
        ax.axvline(0, color="#aaa", lw=0.6, ls=":")
        texts = []
        top_own = nlp[own_sig].nlargest(min(6, own_sig.sum())).index if own_sig.sum() else []
        for i in top_own:
            texts.append(ax.text(lfc[i], nlp[i], df.loc[i, "Gene"],
                                 fontsize=13, fontweight="bold", zorder=7))
        for i in df.index[oo]:
            texts.append(ax.text(lfc[i], nlp[i], df.loc[i, "Gene"],
                                 fontsize=13, style="italic", color=other_col, zorder=7))
        if texts:
            safe_adjust(texts, ax, force_text=(0.3, 1.5), force_static=(0.2, 0.8),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#888"))
        ax.set_xlim(-xh, xh); ax.set_ylim(0, yt)
        ax.set_xlabel("logFC")
        ax.set_ylabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})")
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=11, framealpha=0.9, loc="upper right")

    _volcano(axes_v[0], "logFC_up", f"{PVAL_BASE_COL}_up",
             df["sig_up"], df["sig_down"], C["g1"], C["g2"],
             f"G1  (Stroke Cortex\u2191)\n{n_g1} significant", "G2")
    _volcano(axes_v[1], "logFC_down", f"{PVAL_BASE_COL}_down",
             df["sig_down"], df["sig_up"], C["g2"], C["g1"],
             f"G2  (Healthy Cortex\u2191)\n{n_g2} significant", "G1")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figC_volcanos.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── FIGURE E — Summary dashboard ────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs_e = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)

    # E1: Key metrics text
    ax1 = fig.add_subplot(gs_e[0])
    ax1.axis("off")
    metrics_txt = [
        ("Shared genes analysed",        f"{len(df):,}"),
        ("", ""),
        ("Sig G1 only",                  f"{n_g1} genes"),
        ("Sig G2 only",                  f"{n_g2} genes"),
        ("Sig in BOTH",                  f"{n_ab} genes"),
        ("", ""),
        ("Direction flips",              f"{n_flip}/{len(df)} ({100*n_flip/len(df):.1f}%)"),
        ("logFC Pearson r",              f"{r_lfc_pe:.3f}"),
        ("logFC Spearman r",             f"{r_lfc_sp:.3f}"),
        (f"{PVAL_DISP_NAME} Spearman r", f"{r_nla_sp:.3f}"),
        ("", ""),
        ("Mean \u0394 logFC",            f"{df['delta_logFC'].mean():+.4f}"),
        ("SD  \u0394 logFC",             f"{df['delta_logFC'].std():.4f}"),
    ]
    y_start = 0.98
    for key, val in metrics_txt:
        if not key:
            y_start -= 0.03; continue
        ax1.text(0.02, y_start, key + ":", transform=ax1.transAxes,
                 fontsize=11, fontweight="bold", va="top")
        ax1.text(0.55, y_start, val, transform=ax1.transAxes,
                 fontsize=11, va="top", color="#333")
        y_start -= 0.06
    ax1.set_title("Key Metrics", fontsize=14, pad=8)

    # E2: Correlation bar chart
    ax2 = fig.add_subplot(gs_e[1])
    metrics_bar = ["logFC\nPearson", "logFC\nSpearman", f"{PVAL_DISP_NAME}\nSpearman"]
    vals_bar    = [r_lfc_pe, r_lfc_sp, r_nla_sp]
    bar_c = ["#4c956c" if v >= 0.7 else "#e9c46a" if v >= 0.5 else "#e76f51"
             for v in vals_bar]
    bars2 = ax2.bar(metrics_bar, vals_bar, color=bar_c, edgecolor="white",
                    alpha=0.88, width=0.55)
    for bar, val in zip(bars2, vals_bar):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                 ha="center", fontsize=13, fontweight="bold")
    ax2.axhline(0.7, color="#4c956c", lw=1, ls="--", alpha=0.7, label="r=0.70 (good)")
    ax2.axhline(0.5, color="#e9c46a", lw=1, ls="--", alpha=0.7, label="r=0.50 (fair)")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Correlation (r)")
    ax2.set_title("G1 vs G2 Concordance\nby Metric", fontsize=14)
    ax2.legend(fontsize=10.5)

    # E3: Stacked sig-category bar
    ax3 = fig.add_subplot(gs_e[2])
    cat_vals = [int(cats[c]) for c in ["Sig in Both", "G1 Only", "G2 Only", "Neither"]]
    cat_cols = [C["both"], C["g1"], C["g2"], GREY_COL]
    cat_labs = ["Sig in Both", "G1 Only", "G2 Only", "Neither"]
    bottom = 0
    for val, col, lab in zip(cat_vals, cat_cols, cat_labs):
        ax3.bar(0, val, bottom=bottom, color=col, width=0.55,
                edgecolor="white", alpha=0.88, label=f"{lab} (n={val})")
        if val > 20:
            ax3.text(0, bottom + val / 2, str(val), ha="center", va="center",
                     fontsize=14, fontweight="bold", color="white")
        bottom += val
    ax3.set_xlim(-0.6, 0.6); ax3.set_xticks([])
    ax3.set_ylabel("Number of genes")
    ax3.set_title(f"Significance Category\n(n={len(df)} shared genes)", fontsize=14)
    ax3.legend(loc="upper right", fontsize=10.5, framealpha=0.9)

    # E4: Direction-consistency pie
    ax4 = fig.add_subplot(gs_e[3])
    ax4.pie([n_flip, len(df) - n_flip],
            labels=[f"Flipped\n(n={n_flip})", f"Consistent\n(n={len(df)-n_flip})"],
            colors=[C["flip"], "#d8e2dc"],
            autopct="%1.1f%%", startangle=90, pctdistance=0.75,
            textprops={"fontsize": 13},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax4.set_title(f"Direction Consistency\n({100*n_flip/len(df):.1f}% flip sign)", fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figE_summary_dashboard.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_genes":         len(df),
        "n_sig_G1_only":   n_g1,
        "n_sig_G2_only":   n_g2,
        "n_sig_both":      n_ab,
        "n_flip":          n_flip,
        "pct_flip":        round(100 * n_flip / len(df), 2),
        # Bland-Altman statistics (Fig A2)
        "pearson_r_all":   round(r_lfc_pe, 4),
        "mab_all":         round(abs_bias_all, 4),
        "pearson_r_top10": round(r_lfc_pe_top10, 4) if not np.isnan(r_lfc_pe_top10) else np.nan,
        "mab_top10":       round(abs_bias_top10, 4)  if not np.isnan(abs_bias_top10)  else np.nan,
        # Spearman r of -log10(p) values
        "spearman_r_neglogp": round(r_nla_sp, 4),
        "spearman_r_neglogp_top10": round(r_nla_sp_top10, 4) if not np.isnan(r_nla_sp_top10) else np.nan,
        # Spearman r of logFC (kept for reference)
        "spearman_r_lfc":  round(r_lfc_sp, 4),
        # MAB as % of mean |logFC|
        "mab_pct_all":     round(mab_pct_all, 2) if not np.isnan(mab_pct_all) else np.nan,
        "mab_pct_top10":   round(mab_pct_top10, 2) if not np.isnan(mab_pct_top10) else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

DISPLAY_NAMES = {
    "WHOLE": "WholeBrain",
    "Astrocytes": "Astrocytes",
    "Microglia": "Microglia",
    "Astrocytes_cortex_hippo": "Astrocytes_Cortex_Hippo",
}

# ── Pre-compute shared axis limits per cell type ─────────────────────────────
# BA x-axis & y-axis: from BLIND annotation only, applied to all variants
# NLA axes: max -log10(p) across all sig genes for this cell type
print("Pre-computing per-cell-type axis limits…")
celltype_ylims = {}
for ct in data_by_celltype:
    max_ba_y = 0.5
    max_ba_x = 0.5
    all_nla_vals = []
    for model in data_by_celltype[ct]:
        for ann, merged in data_by_celltype[ct][model].items():
            df_tmp = prepare_df(merged)
            cm = df_tmp["sig_cat"] != "Neither"
            if cm.sum() > 0:
                # BA x-axis and y-axis: use ALL annotations so every colored point fits
                max_ba_y = max(max_ba_y,
                               float(df_tmp.loc[cm, "delta_logFC"].abs().max()))
                max_ba_x = max(max_ba_x,
                               float(df_tmp.loc[cm, "logFC_up"].abs().max()))
                all_nla_vals.extend(df_tmp.loc[cm, "nla_up"].tolist())
                all_nla_vals.extend(df_tmp.loc[cm, "nla_down"].tolist())
    if all_nla_vals:
        nla_arr = np.array(all_nla_vals)
        nla_limit = float(nla_arr.max()) * 1.15
    else:
        nla_limit = 0.5
    celltype_ylims[ct] = {
        "ba_ylim": max_ba_y * 1.20,
        "ba_xlim": max_ba_x * 1.50,
        "nla_lim": max(nla_limit, 0.5),
    }
    print(f"  {ct}: ba_ylim=±{celltype_ylims[ct]['ba_ylim']:.3f}, "
          f"ba_xlim=±{celltype_ylims[ct]['ba_xlim']:.3f}, "
          f"nla_lim={celltype_ylims[ct]['nla_lim']:.2f}")

summary_rows = []
total = sum(len(anns) for ct in data_by_celltype for m, anns in data_by_celltype[ct].items())
count = 0

for ct in sorted(data_by_celltype.keys()):
    ct_display = DISPLAY_NAMES.get(ct, ct)
    ylims = celltype_ylims.get(ct, {})
    ct_ab_y = AB_Y_SMALL if ct in AB_Y_SMALL_CTS else AB_Y_DEFAULT
    for model in sorted(data_by_celltype[ct].keys()):
        for ann in sorted(data_by_celltype[ct][model].keys()):
            count += 1
            tag = f"{ct_display}/{model}_{ann}"
            outdir = str(OUT_BASE / ct_display / f"{model}_{ann}")

            print(f"[{count}/{total}] {tag}…")
            merged = data_by_celltype[ct][model][ann]
            df_prepped = prepare_df(merged, logfc_thresh=0.1)
            row = make_figures(df_prepped, outdir, tag,
                               ba_ylim=ylims.get("ba_ylim"),
                               ba_xlim=ylims.get("ba_xlim"),
                               nla_lim=ylims.get("nla_lim"),
                               merged=merged,
                               ab_y_axis=ct_ab_y)
            row.update({"cell_type": ct_display, "model": model, "annotation": ann})
            summary_rows.append(row)
            print(f"         {row['n_genes']} genes | G1={row['n_sig_G1_only']} "
                  f"G2={row['n_sig_G2_only']} Both={row['n_sig_both']} | "
                  f"flips={row['pct_flip']:.1f}% | "
                  f"r_pe={row['pearson_r_all']:.3f} mab={row['mab_all']:.4f} | "
                  f"r_nla_sp={row['spearman_r_neglogp']:.3f}")

# Save summary table
summary_df = pd.DataFrame(summary_rows)
OUT_BASE.mkdir(parents=True, exist_ok=True)
summary_path = str(OUT_BASE / "summary_all_comparisons.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary table saved to: {summary_path}")

if summary_df.empty:
    print("No combinations loaded — skipping condensed table and figures.")
    print(f"All figures saved under: {OUT_BASE}")
    print("Done.")
    sys.exit(0)

print(f"All figures saved under: {OUT_BASE}")
print("Done.")
