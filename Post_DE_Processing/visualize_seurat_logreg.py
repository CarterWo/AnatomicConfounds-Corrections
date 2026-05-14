#!/usr/bin/env python3
"""
visualize_seurat_logreg.py
==========================
G1 (UP) vs G2 (DOWN) Comparison — Seurat Logistic Regression

  G1 (UP)  : Stroke-FMT cortex-enriched (+1 SD), Healthy-FMT depleted
  G2 (DOWN): Healthy-FMT cortex-enriched (+1 SD), Stroke-FMT depleted

Uses Seurat logistic regression (test.use="LR") model outputs.
Generates Figures A–E for every combination, organized by cell type.

Output: ./submission/seurat_logreg/{cell_type}/{model}_{annotation}/figX_*.png
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

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR  = SCRIPT_DIR.parent                     # …/Manuscript
LMM_BASE    = PARENT_DIR / "LMM_extract" / "Seurat_&_Dream_updated_2_5"
DATA_PATH   = LMM_BASE / "Global_CT_Analysis"
LOCAL_PATH  = LMM_BASE / "Local_Regional_Analysis"
PB_PATH     = LMM_BASE / "Pseudobulk_Validation"
OUT_BASE    = SCRIPT_DIR / "S&Du_(search)" / "g1g2"

# ── Target Filters ───────────────────────────────────────────────────────────
# Define which models and cell types to process.
# Leave as empty lists [] to process EVERYTHING found in the directories.
TARGET_MODELS = ['seurat']  # Seurat logistic regression
TARGET_CELL_TYPES = ['Astrocytes', 'Microglia','WHOLE'] # e.g., ['WHOLE', 'Astrocytes']

# P-value selection
USE_ADJUSTED_PVAL = True  # True uses 'adj.P.Val', False uses unadjusted 'P.Value'
# ─────────────────────────────────────────────────────────────────────────────

# Dynamic P-value column configuration
PVAL_BASE_COL = "adj.P.Val" if USE_ADJUSTED_PVAL else "P.Value"
PVAL_DISP_NAME = "adj.P" if USE_ADJUSTED_PVAL else "P-value"

# Add parent dir to sys.path so we can import lmm_comparison_utils
sys.path.insert(0, str(PARENT_DIR))
import lmm_comparison_utils as lmm
from plot_utils import draw_proportional_venn

SIG = 0.05
LFC_THRESH = 0.5  # minimum |logFC| to call a gene significant

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         16,
    "axes.titlesize":    18,
    "axes.titleweight":  "bold",
    "axes.labelsize":    16,
    "legend.fontsize":   14,
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

def _save_venn_genes(outdir, prefix, gene_sets):
    """Save Venn gene lists to CSVs under {outdir}/venn_genes/."""
    venn_dir = os.path.join(outdir, "venn_genes")
    os.makedirs(venn_dir, exist_ok=True)
    for label, genes in gene_sets.items():
        pd.Series(sorted(genes), name="Gene").to_csv(
            os.path.join(venn_dir, f"{prefix}_{label}.csv"), index=False)


def label_all(ax, sub, xcol, ycol, gene_col="Gene", fontsize=14, col="#222"):
    """Return text objects for all rows in sub-dataframe for adjustText."""
    texts = []
    for _, row in sub.iterrows():
        texts.append(ax.text(row[xcol], row[ycol], row[gene_col],
                             fontsize=fontsize, color=col, fontweight="bold", zorder=9))
    return texts


def label_genes(ax, xs, ys, genes, n=10, criterion=None, fontsize=14, color="#222"):
    """Return text objects for top n genes by |criterion| for adjustText."""
    if criterion is None:
        criterion = pd.Series(np.ones(len(xs)), index=xs.index)
    idx = criterion.abs().nlargest(n).index
    texts = []
    for i in idx:
        texts.append(ax.text(xs[i], ys[i], genes[i],
                             fontsize=fontsize, color=color, zorder=8))
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

def prepare_df(merged):
    """Add computed columns to a merged UP/DOWN dataframe."""
    df = merged.copy()

    p_up_col = f"{PVAL_BASE_COL}_up"
    p_down_col = f"{PVAL_BASE_COL}_down"

    df["sig_up"]      = (df[p_up_col]   < SIG) & (df["logFC_up"].abs()   >= LFC_THRESH)
    df["sig_down"]    = (df[p_down_col]  < SIG) & (df["logFC_down"].abs() >= LFC_THRESH)
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


def make_figures(df, outdir, tag, ba_ylim=None, nla_lim=None):
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
    draw_proportional_venn(ax, n_g1, n_g2, n_ab, C["g1"], C["g2"], C["both"],
                           title=f"Significant Genes ({PVAL_DISP_NAME} < {SIG})")
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

    x_margin_A2 = float(df.loc[colored, "logFC_up"].abs().max()) * 1.5 \
                  if colored.sum() else 0.5
    xlim_A2 = (-x_margin_A2, x_margin_A2)
    # Use shared y-limit if provided, otherwise fall back to local
    if ba_ylim is not None:
        ylim_A2 = (-ba_ylim, ba_ylim)
    else:
        y_margin_A2 = float(df.loc[colored, "delta_logFC"].abs().max()) * 1.20 \
                      if colored.sum() else 0.5
        ylim_A2 = (-y_margin_A2, y_margin_A2)

    scatter_grey_rect(ax, df.loc[grey, "logFC_up"], df.loc[grey, "delta_logFC"],
                      xlim_A2, ylim_A2,
                      label=f"Neither (n={grey.sum()}, clipped)")
    for cat, col, lbl in [("G1 Only", C["g1"], "G1 only"),
                           ("G2 Only", C["g2"], "G2 only"),
                           ("Sig in Both", C["both"], "Both")]:
        m = df["sig_cat"] == cat
        if m.sum():
            ax.scatter(df.loc[m, "logFC_up"], df.loc[m, "delta_logFC"],
                       color=col, alpha=0.92, s=24, edgecolors="none",
                       zorder=5, label=f"{lbl} (n={m.sum()})")
    ax.axhline(hi_ba, color="firebrick", lw=1.2, ls="--", label="95% LoA")
    ax.axhline(lo_ba, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="#555", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(*xlim_A2); ax.set_ylim(*ylim_A2)
    ax.set_xlabel("logFC  [G1]")
    ax.set_ylabel("\u0394 logFC  [G2 \u2212 G1]")
    top10_mab_str     = f"{abs_bias_top10:.3f}" if not np.isnan(abs_bias_top10) else "n/a"
    top10_mab_pct_str = f"{mab_pct_top10:.1f}%" if not np.isnan(mab_pct_top10) else "n/a"
    ax.set_title(f"Bland-Altman\n"
                 f"All Genes: MAB={abs_bias_all:.3f} (% of logFC: {mab_pct_all:.1f}%)\n"
                 f"Top 10%: MAB={top10_mab_str} (% of logFC: {top10_mab_pct_str})",
                 fontsize=15)
    ax.legend(markerscale=1.1, fontsize=11.2, framealpha=0.88, loc="upper right")
    texts_A2 = []
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        sub = df[(df["sig_cat"] == cat) & (df["logFC_up"].abs() > 0.2)]
        texts_A2.extend(label_all(ax, sub, "logFC_up", "delta_logFC", col=col))
    if texts_A2:
        adjust_text(texts_A2, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color="#bbb"),
                        force_text=(0.3, 4.0), force_points=(0.2, 1.8), lim=1000)

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
                 fontsize=15)
    ax.legend(markerscale=1.1, fontsize=11.2, framealpha=0.88, loc="upper right")
    TOP_N_A3 = 10  # cap labels per category to avoid OOM with large datasets
    texts_A3 = []
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]), ("Sig in Both", C["both"])]:
        sub = df[df["sig_cat"] == cat].copy()
        sub["_max_nla"] = sub[["nla_up", "nla_down"]].max(axis=1)
        if len(sub) > TOP_N_A3:
            sub = sub.nlargest(TOP_N_A3, "_max_nla")
        texts_A3.extend(label_all(ax, sub, "nla_up", "nla_down", col=col))
    if texts_A3:
        adjust_text(texts_A3, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"),
                        force_text=(0.3, 4.0), force_points=(0.2, 1.8), lim=1000)

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
        "font.size": 30, "axes.titlesize": 33, "axes.labelsize": 30,
        "legend.fontsize": 24, "xtick.labelsize": 27, "ytick.labelsize": 27,
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
    x_margin = float(df.loc[colored_mask_B1, "logFC_up"].abs().max()) * 1.5 \
               if colored_mask_B1.sum() else 0.5
    xlim_B1 = (-x_margin, x_margin)
    ylim_B1 = (-y_margin, y_margin)

    scatter_grey_rect(ax,
                      df.loc[~colored_mask_B1, "logFC_up"],
                      df.loc[~colored_mask_B1, "delta_logFC"],
                      xlim_B1, ylim_B1,
                      label=f"Neither (n={(~colored_mask_B1).sum()}, clipped)")

    labeled_genes_B1 = df["Gene"].isin(top25_genes)
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]),
                     ("Sig in Both", C["both"])]:
        mask = df["sig_cat"] == cat
        if mask.sum():
            # Non-labeled: faded to grey alpha/size so labels stand out
            mask_unlabeled = mask & ~labeled_genes_B1
            ax.scatter(df.loc[mask_unlabeled, "logFC_up"], df.loc[mask_unlabeled, "delta_logFC"],
                       color=col, alpha=GREY_ALPHA, s=GREY_SIZE, edgecolors="none",
                       zorder=4, label=f"{cat} (n={mask.sum()})")
            # Labeled genes: full opacity on top
            mask_labeled = mask & labeled_genes_B1
            if mask_labeled.sum():
                ax.scatter(df.loc[mask_labeled, "logFC_up"], df.loc[mask_labeled, "delta_logFC"],
                           color=col, alpha=0.92, s=24, edgecolors="none", zorder=5)

    ax.axhline(hi_ba, color="firebrick", lw=1.2, ls="--", label="95% limits of agreement")
    ax.axhline(lo_ba, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="#555", lw=0.8, ls=":", alpha=0.7)

    ax.annotate("G1 > G2\n(larger effect)", xy=(x_margin * 0.55, lo_ba * 0.6),
                fontsize=24, color=C["g1"], style="italic")
    ax.annotate("G2 > G1\n(larger effect)", xy=(x_margin * 0.55, hi_ba * 0.55),
                fontsize=24, color=C["g2"], style="italic")

    ax.set_xlim(*xlim_B1)
    ax.set_ylim(*ylim_B1)
    ax.set_xlabel("logFC  [G1]")
    ax.set_ylabel("\u0394 logFC  [G2 \u2212 G1]\n(positive = G2 has larger effect)")
    ax.set_title(f"Bland-Altman\n"
                 f"All Genes: MAB={abs_bias_all:.3f} (% of logFC: {mab_pct_all:.1f}%)\n"
                 f"Top 10%: MAB={top10_mab_str} (% of logFC: {top10_mab_pct_str})",
                 fontsize=27)
    ax.legend(fontsize=24, framealpha=0.9, loc="upper right")

    texts_B1 = []
    sub_B1_top25 = df[colored_mask_B1 & df["Gene"].isin(top25_genes)].copy()
    for cat, col in [("G1 Only", C["g1"]), ("G2 Only", C["g2"]),
                     ("Sig in Both", C["both"])]:
        texts_B1.extend(label_all(ax, sub_B1_top25[sub_B1_top25["sig_cat"] == cat],
                                   "logFC_up", "delta_logFC", col=col, fontsize=24))
    if texts_B1:
        adjust_text(texts_B1, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"),
                        force_text=(0.3, 4.0), force_points=(0.2, 1.8), lim=1000)

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
    ax.set_yticklabels(sig_r["Gene"].values, fontsize=26, fontweight="bold")
    ax.set_xlabel("logFC")
    ax.set_title(f"G1 vs G2 logFC\n(Significant genes only, n={len(sig_r)})", fontsize=27)

    leg_sig = [mpatches.Patch(color=sig_col_map[c], label=c)
               for c in ["Sig in Both", "G1 Only", "G2 Only"]
               if (sig_r["sig_cat"] == c).any()]
    leg_shape = [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                        markeredgecolor="#555", markeredgewidth=1.4,
                        markersize=7, label="\u25cb G1"),
                 Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                        markersize=7, label="\u25cf G2")]
    ax.legend(handles=leg_sig + leg_shape, fontsize=24, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figB_effect_size_shift.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
        "legend.fontsize": 14, "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    # ── FIGURE C — Volcano plots ─────────────────────────────────────────────
    fig, axes_v = plt.subplots(1, 2, figsize=(16, 6.5))

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
                                 fontsize=14, fontweight="bold", zorder=7))
        for i in df.index[oo]:
            texts.append(ax.text(lfc[i], nlp[i], df.loc[i, "Gene"],
                                 fontsize=14, style="italic", color=other_col, zorder=7))
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaa"),
                        force_text=(0.3, 4.0), force_points=(0.2, 1.8), lim=1000)
        ax.set_xlim(-xh, xh); ax.set_ylim(0, yt)
        ax.set_xlabel("logFC")
        ax.set_ylabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})")
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12, framealpha=0.9, loc="upper right")

    _volcano(axes_v[0], "logFC_up", f"{PVAL_BASE_COL}_up",
             df["sig_up"], df["sig_down"], C["g1"], C["g2"],
             f"G1  (Stroke Cortex\u2191)\n{n_g1} significant", "G2")
    _volcano(axes_v[1], "logFC_down", f"{PVAL_BASE_COL}_down",
             df["sig_down"], df["sig_up"], C["g2"], C["g1"],
             f"G2  (Healthy Cortex\u2191)\n{n_g2} significant", "G1")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figC_volcanos.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── FIGURE D — Direction flips ───────────────────────────────────────────
    fig, axes_d = plt.subplots(1, 3, figsize=(20, 6))
    flipped = df[df["sign_flip"]]

    # D1: logFC scatter with flips
    ax = axes_d[0]
    cm_d = df["sign_flip"] | (df["sig_cat"] != "Neither")
    ld = colored_sym_lim(df.loc[cm_d, "logFC_up"], df.loc[cm_d, "logFC_down"], pad=0.30)
    scatter_grey_sym(ax, df.loc[~cm_d, "logFC_up"], df.loc[~cm_d, "logFC_down"],
                     ld, label=f"Consistent, not sig (n={(~cm_d).sum()}, clipped)")
    ax.scatter(flipped["logFC_up"], flipped["logFC_down"],
               color=C["flip"], alpha=0.70, s=22, marker="^",
               edgecolors="none",
               label=f"Sign reversal (n={n_flip})", zorder=4)
    ax.plot([-ld, ld], [-ld, ld], "k--", lw=0.8, alpha=0.45)
    ax.axhline(0, color="#aaa", lw=0.5, ls=":"); ax.axvline(0, color="#aaa", lw=0.5, ls=":")
    ax.fill_between([-ld, 0], [0]*2, [ld]*2, color="#fff0ee", alpha=0.4, zorder=0)
    ax.fill_between([0, ld], [-ld]*2, [0]*2, color="#fff0ee", alpha=0.4, zorder=0)
    ax.set_xlim(-ld, ld); ax.set_ylim(-ld, ld)
    ax.set_xlabel("logFC  [G1]"); ax.set_ylabel("logFC  [G2]")
    ax.set_title("Sign Reversals on logFC Scatter\n(shaded = discordant quadrants)",
                 fontsize=15)
    ax.legend(markerscale=1.2, fontsize=12, framealpha=0.88, loc="upper left")
    if len(flipped) > 0:
        texts_D1 = label_genes(ax, flipped["logFC_up"], flipped["logFC_down"],
                               flipped["Gene"], n=min(8, len(flipped)),
                               criterion=flipped["logFC_up"].abs(), fontsize=9.8)
        if texts_D1:
            adjust_text(texts_D1, ax=axes_d[0],
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaa"),
                        force_text=(0.3, 4.0), force_points=(0.2, 1.8), lim=1000)

    # D2: delta vs G1
    ax = axes_d[1]
    cm_d2 = df["sign_flip"] | (df["sig_cat"] != "Neither")
    col_x_D2 = df.loc[cm_d2, "logFC_up"]
    col_y_D2 = df.loc[cm_d2, "delta_logFC"]
    lim_D2_x = float(col_x_D2.abs().max()) * 1.30 if cm_d2.sum() else 0.5
    lim_D2_y = float(col_y_D2.abs().max()) * 1.30 if cm_d2.sum() else 0.5
    xlim_D2 = (-lim_D2_x, lim_D2_x)
    ylim_D2 = (-lim_D2_y, lim_D2_y)
    scatter_grey_rect(ax, df.loc[~cm_d2, "logFC_up"], df.loc[~cm_d2, "delta_logFC"],
                      xlim_D2, ylim_D2,
                      label=f"No flip, not sig (n={(~cm_d2).sum()}, clipped)")
    ax.scatter(flipped["logFC_up"], flipped["delta_logFC"],
               color=C["flip"], alpha=0.70, s=22, marker="^",
               edgecolors="none",
               label=f"Sign reversal (n={n_flip})", zorder=4)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.axvline(0, color="#aaa", lw=0.6, ls=":")
    ax.set_xlim(*xlim_D2); ax.set_ylim(*ylim_D2)
    ax.set_xlabel("logFC  [G1]  (starting direction)")
    ax.set_ylabel("\u0394 logFC  [G2 \u2212 G1]")
    ax.set_title("Effect Size Shift by Starting Direction\n(reversals cross the dashed zero line)",
                 fontsize=15)
    ax.legend(fontsize=12, framealpha=0.88)
    if len(flipped) > 0:
        texts_D2 = label_genes(ax, flipped["logFC_up"], flipped["delta_logFC"],
                               flipped["Gene"], n=min(8, len(flipped)),
                               criterion=flipped["delta_logFC"].abs(), fontsize=9.8)
        if texts_D2:
            adjust_text(texts_D2, ax=axes_d[1],
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaa"),
                        force_text=(0.3, 4.0), force_points=(0.2, 1.8), lim=1000)

    # D3: Pie
    ax = axes_d[2]
    uu = ((df["logFC_up"] > 0) & (df["logFC_down"] > 0) & ~df["sign_flip"]).sum()
    dd = ((df["logFC_up"] < 0) & (df["logFC_down"] < 0) & ~df["sign_flip"]).sum()
    ud = ((df["logFC_up"] > 0) & (df["logFC_down"] < 0)).sum()
    du = ((df["logFC_up"] < 0) & (df["logFC_down"] > 0)).sum()
    _, _, autot = ax.pie(
        [uu, dd, ud, du],
        labels=[f"Both up\n(n={uu})", f"Both down\n(n={dd})",
                f"Reversal\n(n={ud})", f"Reversal\n(n={du})"],
        colors=["#c8e6c9", "#bbdefb", C["flip"], "#ff8a65"],
        explode=[0, 0, 0.07, 0.07],
        autopct="%1.0f%%", startangle=90, pctdistance=0.75,
        textprops={"fontsize": 14}, labeldistance=1.10)
    for at in autot:
        at.set_fontsize(14); at.set_fontweight("bold")
    ax.set_title(f"Direction Agreement\n{100*(ud+du)/len(df):.1f}% reverse sign", fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figD_direction_flips.png"), dpi=160, bbox_inches="tight")
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
                 fontsize=12.8, fontweight="bold", va="top")
        ax1.text(0.55, y_start, val, transform=ax1.transAxes,
                 fontsize=12.8, va="top", color="#333")
        y_start -= 0.06
    ax1.set_title("Key Metrics", fontsize=15, pad=8)

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
                 ha="center", fontsize=14, fontweight="bold")
    ax2.axhline(0.7, color="#4c956c", lw=1, ls="--", alpha=0.7, label="r=0.70 (good)")
    ax2.axhline(0.5, color="#e9c46a", lw=1, ls="--", alpha=0.7, label="r=0.50 (fair)")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Correlation (r)")
    ax2.set_title("G1 vs G2 Concordance\nby Metric", fontsize=15)
    ax2.legend(fontsize=11.2)

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
                     fontsize=15, fontweight="bold", color="white")
        bottom += val
    ax3.set_xlim(-0.6, 0.6); ax3.set_xticks([])
    ax3.set_ylabel("Number of genes")
    ax3.set_title(f"Significance Category\n(n={len(df)} shared genes)", fontsize=15)
    ax3.legend(loc="upper right", fontsize=11.2, framealpha=0.9)

    # E4: Direction-consistency pie
    ax4 = fig.add_subplot(gs_e[3])
    ax4.pie([n_flip, len(df) - n_flip],
            labels=[f"Flipped\n(n={n_flip})", f"Consistent\n(n={len(df)-n_flip})"],
            colors=[C["flip"], "#d8e2dc"],
            autopct="%1.1f%%", startangle=90, pctdistance=0.75,
            textprops={"fontsize": 14},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax4.set_title(f"Direction Consistency\n({100*n_flip/len(df):.1f}% flip sign)", fontsize=15)

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
        # Spearman r of logFC
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

# ── Pre-compute shared y-axis limits per cell type ───────────────────────────
# BA y-axis  : max |delta_logFC| across all sig genes for this cell type
# NLA axes   : max -log10(p) across all sig genes for this cell type
# These are passed to make_figures so all annotation variants share the same scale.
print("Pre-computing per-cell-type axis limits…")
celltype_ylims = {}
for ct in data_by_celltype:
    max_ba_y = 0.5
    max_nla  = 0.5
    for model in data_by_celltype[ct]:
        for ann, merged in data_by_celltype[ct][model].items():
            df_tmp = prepare_df(merged)
            cm = df_tmp["sig_cat"] != "Neither"
            if cm.sum() > 0:
                max_ba_y = max(max_ba_y,
                               float(df_tmp.loc[cm, "delta_logFC"].abs().max()))
                max_nla  = max(max_nla,
                               float(df_tmp.loc[cm, "nla_up"].max()),
                               float(df_tmp.loc[cm, "nla_down"].max()))
    celltype_ylims[ct] = {
        "ba_ylim": max_ba_y * 1.20,
        "nla_lim": max_nla  * 1.12,
    }
    print(f"  {ct}: ba_ylim=±{celltype_ylims[ct]['ba_ylim']:.3f}, "
          f"nla_lim={celltype_ylims[ct]['nla_lim']:.2f}")

summary_rows = []
total = sum(len(anns) for ct in data_by_celltype for m, anns in data_by_celltype[ct].items())
count = 0

for ct in sorted(data_by_celltype.keys()):
    ct_display = DISPLAY_NAMES.get(ct, ct)
    ylims = celltype_ylims.get(ct, {})
    for model in sorted(data_by_celltype[ct].keys()):
        for ann in sorted(data_by_celltype[ct][model].keys()):
            count += 1
            tag = f"{ct_display}/{model}_{ann}"
            outdir = str(OUT_BASE / ct_display / f"{model}_{ann}")

            print(f"[{count}/{total}] {tag}…")
            merged = data_by_celltype[ct][model][ann]
            df_prepped = prepare_df(merged)
            row = make_figures(df_prepped, outdir, tag,
                               ba_ylim=ylims.get("ba_ylim"),
                               nla_lim=ylims.get("nla_lim"))
            row.update({"cell_type": ct_display, "model": model, "annotation": ann})
            summary_rows.append(row)
            print(f"         {row['n_genes']} genes | G1={row['n_sig_G1_only']} "
                  f"G2={row['n_sig_G2_only']} Both={row['n_sig_both']} | "
                  f"flips={row['pct_flip']:.1f}% | "
                  f"r_pe={row['pearson_r_all']:.3f} mab={row['mab_all']:.4f} | "
                  f"r_nla_sp={row['spearman_r_neglogp']:.3f}")

# Save summary table
summary_df = pd.DataFrame(summary_rows)
summary_path = str(OUT_BASE / "summary_all_comparisons_LR.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary table saved to: {summary_path}")

# ── Condensed manuscript-style table ─────────────────────────────────────────
MODEL_DISPLAY = {"seurat": "Logistic Regression (Seurat)"}
condensed = summary_df.rename(columns={
    "cell_type":                "Cell Type",
    "n_sig_G1_only":            "n-sig\n(G1)",
    "n_sig_G2_only":            "n-sig\n(G2)",
    "n_sig_both":               "n-sig\n(Both)",
    "spearman_r_neglogp":       "-log10(p) r\n(All)",
    "spearman_r_neglogp_top10": "-log10(p) r\n(Top %10)",
    "mab_all":                  "MAB\n(All)",
    "mab_top10":                "MAB\n(Top %10)",
    "mab_pct_all":              "% of logFC\n(All)",
    "mab_pct_top10":            "% of logFC\n(Top %10)",
})
ANNOTATION_DISPLAY = {
    "blind":  "Region-Blind",
    "napari": "Region-Aware (Napari)",
    "quint":  "Region-Aware (QuInt)",
}
base_model = condensed["model"].map(MODEL_DISPLAY).fillna(condensed["model"])
ann_label = condensed["annotation"].map(ANNOTATION_DISPLAY).fillna(condensed["annotation"])
condensed["Model"] = base_model + " — " + ann_label
keep_cols = [
    "Model", "Cell Type",
    "n-sig\n(G1)", "n-sig\n(G2)", "n-sig\n(Both)",
    "-log10(p) r\n(All)", "-log10(p) r\n(Top %10)",
    "MAB\n(All)", "MAB\n(Top %10)",
    "% of logFC\n(All)", "% of logFC\n(Top %10)",
]
condensed = condensed[keep_cols]
condensed_path = str(OUT_BASE / "summary_condensed_LR.csv")
condensed.to_csv(condensed_path, index=False)
print(f"Condensed table saved to: {condensed_path}")

print(f"All figures saved under: {OUT_BASE}")
print("Done.")
