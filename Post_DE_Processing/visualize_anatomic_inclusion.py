#!/usr/bin/env python3
"""
visualize_model_comparison.py
=============================
Region-Blind vs Region-Aware DREAM LMM — Focused Difference Visualizations

  BLIND : data/blind/Jan_26BASE_WHOLE_dream_blind.csv
  AWARE : data/aware/Jan_26BASE_WHOLE_dream_quint.csv

Produces 5 focused figures saved to ./figures/
"""

import os, sys, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import gaussian_kde
from adjustText import adjust_text
from plot_utils import draw_proportional_venn
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Multi-comparison dispatcher ────────────────────────────────────────────────
# When invoked directly (no COMPARISON_OVERRIDE set), re-exec this script once
# per comparison listed below in a fresh subprocess. Each run writes to its own
# output subfolder. Running in separate processes avoids matplotlib/pandas
# global-state leakage between variants.
COMPARISONS_TO_RUN = ["blind_vs_quint", "blind_vs_napari"]
if __name__ == "__main__" and "COMPARISON_OVERRIDE" not in os.environ:
    import subprocess
    for _comp in COMPARISONS_TO_RUN:
        print(f"\n{'=' * 80}\n  Running comparison: {_comp}\n{'=' * 80}")
        env = os.environ.copy()
        env["COMPARISON_OVERRIDE"] = _comp
        ret = subprocess.run([sys.executable, __file__], env=env)
        if ret.returncode != 0:
            sys.exit(ret.returncode)
    sys.exit(0)

# ── Data-source switch ─────────────────────────────────────────────────────────
# True  → use data_no_O/ (OB-excluded cohort; single-file format, no UP/DOWN split)
# False → use data/      (original cohort; blind/ and aware/ subdirs)
USE_NO_O_DATA = True

# ── Comparison selection ───────────────────────────────────────────────────────
# "blind_vs_quint"  : region-blind  vs quint atlas-aware
# "blind_vs_napari" : region-blind  vs napari manually-drawn
# "napari_vs_quint" : napari-aware  vs quint atlas-aware
COMPARISON = os.environ.get("COMPARISON_OVERRIDE", "blind_vs_quint")

_LABEL_MAP = {
    "blind_vs_quint":  ("Blind",  "Quint"),
    "blind_vs_napari": ("Blind",  "Napari"),
    "napari_vs_quint": ("Napari", "Quint"),
}
G1_LABEL, G2_LABEL = _LABEL_MAP[COMPARISON]

if USE_NO_O_DATA:
    _NO_O_DIR    = os.path.join(SCRIPT_DIR, "data_no_O")
    _NO_O_GLOBAL = os.path.join(_NO_O_DIR, "Global_CT_Analysis")
    _NO_O_LOCAL  = os.path.join(_NO_O_DIR, "Local_Regional_Analysis")

    if COMPARISON == "blind_vs_quint":
        BLIND_CSV = os.path.join(_NO_O_LOCAL,  "Jan_26BASE_WHOLE_dream_blind.csv")
        AWARE_CSV = os.path.join(_NO_O_GLOBAL, "Jan_26BASE_WHOLE_dream_quint.csv")
        OUTDIR    = os.path.join(SCRIPT_DIR, "S&Du_(search)", "anatomic", "blind_vs_quint")
    elif COMPARISON == "blind_vs_napari":
        BLIND_CSV = os.path.join(_NO_O_LOCAL,  "Jan_26BASE_WHOLE_dream_blind.csv")
        AWARE_CSV = os.path.join(_NO_O_GLOBAL, "Jan_26BASE_WHOLE_dream_napari.csv")
        OUTDIR    = os.path.join(SCRIPT_DIR, "S&Du_(search)", "anatomic", "blind_vs_napari")
    elif COMPARISON == "napari_vs_quint":
        BLIND_CSV = os.path.join(_NO_O_GLOBAL, "Jan_26BASE_WHOLE_dream_napari.csv")
        AWARE_CSV = os.path.join(_NO_O_GLOBAL, "Jan_26BASE_WHOLE_dream_quint.csv")
        OUTDIR    = os.path.join(SCRIPT_DIR, "S&Du_(search)", "anatomic", "napari_vs_quint")
else:
    BLIND_CSV = os.path.join(SCRIPT_DIR, "data", "blind",  "Jan_26BASE_WHOLE_dream_blind.csv")
    AWARE_CSV = os.path.join(SCRIPT_DIR, "data", "aware",  "Jan_26BASE_WHOLE_dream_quint.csv")
    OUTDIR    = os.path.join(SCRIPT_DIR, "S&Du_(search)", "anatomic", f"{COMPARISON}_original")

os.makedirs(OUTDIR, exist_ok=True)

SIG = 0.05
LOGFC_THRESH = 0.1

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    13,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

C = {
    "blind":  "#FF4500",   # vivid OrangeRed   — Region-Blind
    "aware":  "#0080FF",   # vivid DodgerBlue  — Region-Aware
    "both":   "#00BB55",   # vivid Green       — Sig in Both
    "flip":   "#9900CC",   # vivid Violet      — direction flip / 3rd category
    "up":     "#EE1111",   # vivid Red         — upregulated in volcanos
    "down":   "#0033CC",   # vivid Navy        — downregulated in volcanos
}

# Grey point style — medium-dark for clear visibility on white backgrounds
GREY_COL   = "#808080"
GREY_ALPHA = 0.50
GREY_SIZE  = 18


USE_ADJUSTED_PVAL = False  # True uses 'adj.P.Val', False uses unadjusted 'P.Value'
PVAL_BASE_COL = "adj.P.Val" if USE_ADJUSTED_PVAL else "P.Value"
PVAL_DISP_NAME = "adj.P" if USE_ADJUSTED_PVAL else "P-value"
# ── Load & Merge ───────────────────────────────────────────────────────────────
print("Loading data…")
blind_raw = pd.read_csv(BLIND_CSV)
aware_raw = pd.read_csv(AWARE_CSV)

blind = blind_raw.rename(columns={c: c+"_blind" for c in blind_raw.columns if c != "Gene"})
aware = aware_raw.rename(columns={c: c+"_aware" for c in aware_raw.columns if c != "Gene"})
df = blind.merge(aware, on="Gene", how="inner")

only_blind = set(blind_raw.Gene) - set(aware_raw.Gene)
only_aware = set(aware_raw.Gene) - set(blind_raw.Gene)

df["sig_blind"]   = (df[f"{PVAL_BASE_COL}_blind"] < SIG) & (df["logFC_blind"].abs() >= LOGFC_THRESH)
df["sig_aware"]   = (df[f"{PVAL_BASE_COL}_aware"] < SIG) & (df["logFC_aware"].abs() >= LOGFC_THRESH)
df["sign_flip"]   = np.sign(df["logFC_blind"]) != np.sign(df["logFC_aware"])
df["delta_logFC"] = df["logFC_aware"] - df["logFC_blind"]
df["mean_logFC"]  = (df["logFC_aware"] + df["logFC_blind"]) / 2
df["nlp_blind"]   = -np.log10(df[f"{PVAL_BASE_COL}_blind"].clip(1e-300))
df["nlp_aware"]   = -np.log10(df[f"{PVAL_BASE_COL}_aware"].clip(1e-300))
df["nla_blind"]   = -np.log10(df[f"{PVAL_BASE_COL}_blind"].clip(1e-300))
df["nla_aware"]   = -np.log10(df[f"{PVAL_BASE_COL}_aware"].clip(1e-300))

def sig_cat(row):
    if row.sig_blind and row.sig_aware: return "Sig in Both"
    if row.sig_blind:                   return "Blind Only"
    if row.sig_aware:                   return "Aware Only"
    return "Neither"

df["sig_cat"] = df.apply(sig_cat, axis=1)

cats = df["sig_cat"].value_counts().reindex(
    ["Sig in Both", "Blind Only", "Aware Only", "Neither"], fill_value=0)

n_flip = int(df["sign_flip"].sum())
r_lfc_sp, _ = stats.spearmanr(df["logFC_blind"], df["logFC_aware"])
r_lfc_pe, _ = stats.pearsonr(df["logFC_blind"],  df["logFC_aware"])
r_nla_sp, _ = stats.spearmanr(df["nla_blind"],   df["nla_aware"])
r_z_sp,   _ = stats.spearmanr(df["z.std_blind"], df["z.std_aware"])

# Top 10% correlations (by max -log10 p across both conditions)
_max_nla = df[["nla_blind", "nla_aware"]].max(axis=1)
_top10_idx = _max_nla.nlargest(max(1, len(df) // 10)).index
r_nla_sp_top10 = stats.spearmanr(df.loc[_top10_idx, "nla_blind"],
                                  df.loc[_top10_idx, "nla_aware"])[0] \
                 if len(_top10_idx) >= 3 else np.nan
_max_lfc = df[["logFC_blind", "logFC_aware"]].abs().max(axis=1)
_top10_lfc_idx = _max_lfc.nlargest(max(1, len(df) // 10)).index
r_lfc_sp_top10 = stats.spearmanr(df.loc[_top10_lfc_idx, "logFC_blind"],
                                  df.loc[_top10_lfc_idx, "logFC_aware"])[0] \
                 if len(_top10_lfc_idx) >= 3 else np.nan
r_lfc_pe_top10 = stats.pearsonr(df.loc[_top10_lfc_idx, "logFC_blind"],
                                 df.loc[_top10_lfc_idx, "logFC_aware"])[0] \
                 if len(_top10_lfc_idx) >= 3 else np.nan

# MAB (Mean Absolute Bias) statistics
abs_bias_all = df["delta_logFC"].abs().mean()
avg_mag_all  = ((df["logFC_blind"].abs() + df["logFC_aware"].abs()) / 2).mean()
mab_pct_all  = abs_bias_all / avg_mag_all * 100 if avg_mag_all > 0 else np.nan
_top10_mab_idx = _max_lfc.nlargest(max(1, len(df) // 10)).index
top10_mab     = df.loc[_top10_mab_idx, "delta_logFC"].abs().mean()
top10_avg_mag = ((df.loc[_top10_mab_idx, "logFC_blind"].abs() +
                  df.loc[_top10_mab_idx, "logFC_aware"].abs()) / 2).mean()
top10_mab_pct = top10_mab / top10_avg_mag * 100 if top10_avg_mag > 0 else np.nan
top10_mab_str     = f"{top10_mab:.3f}" if not np.isnan(top10_mab) else "n/a"
top10_mab_pct_str = f"{top10_mab_pct:.1f}%" if not np.isnan(top10_mab_pct) else "n/a"

print(f"  Shared genes: {len(df)}")
print(f"  Sig categories:\n{cats.to_string()}")
print(f"  Sign flips: {n_flip}/{len(df)} ({100*n_flip/len(df):.1f}%)")
print(f"  logFC Spearman r={r_lfc_sp:.3f}, Pearson r={r_lfc_pe:.3f}")
print(f"  -log10(adjP) Spearman r={r_nla_sp:.3f}")

# ── Effect-change statistics for all significant genes (unadjusted P) ──────────
sig_stats = df[df["sig_cat"] != "Neither"][
    ["Gene", "sig_cat", "logFC_blind", "logFC_aware", "delta_logFC", "mean_logFC"]
].copy()
sig_stats["pct_of_mean"] = (
    sig_stats["delta_logFC"] / sig_stats["mean_logFC"] * 100
).replace([np.inf, -np.inf], np.nan).round(2)
sig_stats = sig_stats.sort_values("delta_logFC", key=abs, ascending=False)

W = 20
print(f"\n{'-'*85}")
print(f"Effect-change statistics - significant genes (unadjusted P < {SIG})")
print(f"{'-'*85}")
print(f"{'Gene':<{W}} {'Category':<14} {'logFC_blind':>11} {'logFC_aware':>11} {'dlogFC':>9} {'% of mean':>10}")
print(f"{'-'*85}")
for _, row in sig_stats.iterrows():
    pct_str = f"{row['pct_of_mean']:>9.1f}%" if pd.notna(row["pct_of_mean"]) else f"{'NA':>10}"
    print(f"{row['Gene']:<{W}} {row['sig_cat']:<14} "
          f"{row['logFC_blind']:>11.4f} {row['logFC_aware']:>11.4f} "
          f"{row['delta_logFC']:>9.4f} {pct_str}")
mean_delta = sig_stats["delta_logFC"].mean()
mean_pct   = sig_stats["pct_of_mean"].mean()
print(f"{'-'*85}")
print(f"{'MEAN (all sig genes)':<{W}} {'':14} {'':>11} {'':>11} "
      f"{mean_delta:>9.4f} {mean_pct:>9.1f}%")
print(f"{'-'*85}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def label_genes(ax, xs, ys, genes, n=10, criterion=None, fontsize=10, color="#222"):
    """Return text objects for top n genes by |criterion| for adjustText."""
    if criterion is None:
        criterion = pd.Series(np.ones(len(xs)), index=xs.index)
    idx = criterion.abs().nlargest(n).index
    texts = []
    for i in idx:
        texts.append(ax.text(xs[i], ys[i], genes[i],
                             fontsize=fontsize, color=color, zorder=8))
    return texts


def _save_venn_genes(outdir, prefix, gene_sets):
    """Save Venn gene lists to CSVs under {outdir}/venn_genes/."""
    venn_dir = os.path.join(outdir, "venn_genes")
    os.makedirs(venn_dir, exist_ok=True)
    for label, genes in gene_sets.items():
        pd.Series(sorted(genes), name="Gene").to_csv(
            os.path.join(venn_dir, f"{prefix}_{label}.csv"), index=False)


def colored_sym_lim(xs_col, ys_col, pad=0.30, fallback=1.0):
    """
    Symmetric axis limit based only on colored (non-grey) points.
    Returns a single half-range r so axes go [-r, r] x [-r, r].
    pad: fractional padding beyond the outermost colored point.
    """
    if len(xs_col) == 0:
        return fallback
    xr = float(np.abs(xs_col).max())
    yr = float(np.abs(ys_col).max())
    return max(xr, yr) * (1 + pad)


def colored_lim_rect(xs_col, ys_col, pad=0.30, fallback=1.0):
    """
    Non-symmetric limits (for -log10 adjP plots, Bland-Altman).
    Returns ((xlo, xhi), (ylo, yhi)).
    """
    if len(xs_col) == 0:
        return (0, fallback), (0, fallback)
    xlo, xhi = float(xs_col.min()), float(xs_col.max())
    ylo, yhi = float(ys_col.min()), float(ys_col.max())
    xpad = max((xhi - xlo) * pad, fallback * 0.1)
    ypad = max((yhi - ylo) * pad, fallback * 0.1)
    return (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)


def scatter_grey_sym(ax, x, y, lim, label=None):
    """
    Plot grey background points, clipping to [-lim, lim] on both axes.
    Points beyond the range are compressed to the edge.
    """
    xc = np.clip(np.asarray(x, float), -lim, lim)
    yc = np.clip(np.asarray(y, float), -lim, lim)
    ax.scatter(xc, yc, color=GREY_COL, alpha=GREY_ALPHA, s=GREY_SIZE,
               edgecolors="none", rasterized=True, zorder=1, label=label)


def scatter_grey_rect(ax, x, y, xlim, ylim, label=None):
    """
    Plot grey background points, clipping to given (xlo,xhi), (ylo,yhi).
    """
    xc = np.clip(np.asarray(x, float), xlim[0], xlim[1])
    yc = np.clip(np.asarray(y, float), ylim[0], ylim[1])
    ax.scatter(xc, yc, color=GREY_COL, alpha=GREY_ALPHA, s=GREY_SIZE,
               edgecolors="none", rasterized=True, zorder=1, label=label)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — "The Divergence": 3-panel (Venn + two scatter plots)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure A: The Divergence…")

# shared sig-gene masks used across A1/A2/A3
colored_mask_A3 = df["sig_cat"] != "Neither"
grey_mask_A3    = ~colored_mask_A3
n_b  = int(cats["Blind Only"])
n_a  = int(cats["Aware Only"])
n_ab = int(cats["Sig in Both"])

# helper: return text objects for all rows in a sub-dataframe for adjustText
def label_all(ax, sub, xcol, ycol, gene_col="Gene", fontsize=11.2, col="#222"):
    texts = []
    for _, row in sub.iterrows():
        texts.append(ax.text(row[xcol], row[ycol], row[gene_col],
                             fontsize=fontsize, color=col, fontweight="bold", zorder=9))
    return texts

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5),
                          gridspec_kw={"width_ratios": [1.1, 1.6, 1.6]})

# ── A1: Venn ──────────────────────────────────────────────────────────────────
ax = axes[0]
draw_proportional_venn(ax, n_b, n_a, n_ab, C["blind"], C["aware"], C["both"],
                       title=f"Significant Genes ({PVAL_DISP_NAME} < {SIG} & |logFC| > {LOGFC_THRESH})")
_save_venn_genes(OUTDIR, "figA_unadj", {
    "Blind_Only": df.loc[df["sig_cat"] == "Blind Only", "Gene"].tolist(),
    "Aware_Only":  df.loc[df["sig_cat"] == "Aware Only",  "Gene"].tolist(),
    "Both":        df.loc[df["sig_cat"] == "Sig in Both", "Gene"].tolist(),
})

# ── A2: logFC scatter ─────────────────────────────────────────────────────────
ax = axes[1]
col_x_A3 = df.loc[colored_mask_A3, "logFC_blind"]
col_y_A3 = df.loc[colored_mask_A3, "logFC_aware"]
lim_A3   = colored_sym_lim(col_x_A3, col_y_A3, pad=0.40)

scatter_grey_sym(ax, df.loc[grey_mask_A3, "logFC_blind"],
                 df.loc[grey_mask_A3, "logFC_aware"], lim_A3,
                 label=f"Not significant (n={grey_mask_A3.sum()})")

# Fully solid colored dots — slightly smaller
for cat, col, lbl in [("Blind Only", C["blind"], f"Sig: {G1_LABEL} only"),
                       ("Aware Only", C["aware"], f"Sig: {G2_LABEL} only"),
                       ("Sig in Both", C["both"], "Sig: Both")]:
    mask = df["sig_cat"] == cat
    if mask.sum():
        ax.scatter(df.loc[mask, "logFC_blind"], df.loc[mask, "logFC_aware"],
                   color=col, alpha=1.0, s=24, edgecolors="none",
                   zorder=5, label=f"{lbl} (n={mask.sum()})")

ax.plot([-0.6, 0.9], [-0.6, 0.9], "k--", lw=0.9, alpha=0.5,
        label="y = x")
ax.axhline(0, color="#aaa", lw=0.5, ls=":")
ax.axvline(0, color="#aaa", lw=0.5, ls=":")
ax.set_xlim(-0.6, 0.9)
ax.set_ylim(-0.6, 0.9)
ax.set_xlabel(f"logFC  [{G1_LABEL}]")
ax.set_ylabel(f"logFC  [{G2_LABEL}]")
ax.set_title(f"Effect Size Concordance\n"
             f"All genes: MAB={abs_bias_all:.3f} (% of |logFC|: {mab_pct_all:.1f}%)\n"
             f"Top 10%:  MAB={top10_mab_str} (% of |logFC|: {top10_mab_pct_str})",
             fontsize=12)
ax.legend(markerscale=1.1, fontsize=11.2, framealpha=0.88, loc="upper left")

# Label all significant genes
texts_A2 = []
for cat, col in [("Blind Only", C["blind"]), ("Aware Only", C["aware"]),
                 ("Sig in Both", C["both"])]:
    sub = df[df["sig_cat"] == cat]
    texts_A2.extend(label_all(ax, sub, "logFC_blind", "logFC_aware", col=col))
if texts_A2:
    adjust_text(texts_A2, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))


# ── A3: -log10(adjP) scatter ──────────────────────────────────────────────────
ax = axes[2]
col_x_A4 = df.loc[colored_mask_A3, "nla_blind"]
col_y_A4 = df.loc[colored_mask_A3, "nla_aware"]
xlim_A4, ylim_A4 = colored_lim_rect(col_x_A4, col_y_A4, pad=0.35, fallback=0.5)
xlim_A4 = (max(0, xlim_A4[0]), float(col_x_A4.max()) * 1.08)
ylim_A4 = (max(0, ylim_A4[0]), ylim_A4[1])

scatter_grey_rect(ax, df.loc[grey_mask_A3, "nla_blind"],
                  df.loc[grey_mask_A3, "nla_aware"],
                  xlim_A4, ylim_A4,
                  label=f"Not significant (n={grey_mask_A3.sum()})")

for cat, col, lbl in [("Blind Only", C["blind"], f"Sig: {G1_LABEL} only"),
                       ("Aware Only", C["aware"], f"Sig: {G2_LABEL} only"),
                       ("Sig in Both", C["both"], "Sig: Both")]:
    mask = df["sig_cat"] == cat
    if mask.sum():
        ax.scatter(df.loc[mask, "nla_blind"], df.loc[mask, "nla_aware"],
                   color=col, alpha=1.0, s=24, edgecolors="none",
                   zorder=5, label=f"{lbl} (n={mask.sum()})")

lim_A4_max = max(xlim_A4[1], ylim_A4[1])
ax.plot([0, lim_A4_max], [0, lim_A4_max], "k--", lw=0.9, alpha=0.5, label="y = x")

sig_line = -np.log10(SIG)
ax.axhline(sig_line, color=C["aware"], lw=1, ls=":", alpha=0.8)
ax.axvline(sig_line, color=C["blind"], lw=1, ls=":", alpha=0.8)
ax.set_xlim(*xlim_A4)
ax.set_ylim(*ylim_A4)
ax.set_xlabel(f"−log₁₀({PVAL_DISP_NAME})  [{G1_LABEL}]")
ax.set_ylabel(f"−log₁₀({PVAL_DISP_NAME})  [{G2_LABEL}]")
_r_nla_top10_str = f"{r_nla_sp_top10:.3f}" if not np.isnan(r_nla_sp_top10) else "n/a"
ax.set_title(f"Significance Concordance\n"
             f"Spearman r = {r_nla_sp:.3f} (all)  |  {_r_nla_top10_str} (top 10%)",
             fontsize=12)
ax.legend(markerscale=1.1, fontsize=11.2, framealpha=0.88, loc="upper right")

# Label all colored (significant) genes
texts_A3 = []
for cat, col in [("Blind Only", C["blind"]), ("Aware Only", C["aware"]),
                 ("Sig in Both", C["both"])]:
    sub = df[df["sig_cat"] == cat]
    texts_A3.extend(label_all(ax, sub, "nla_blind", "nla_aware", col=col))
if texts_A3:
    adjust_text(texts_A3, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))

fig.tight_layout()
out = os.path.join(OUTDIR, "figA_divergence.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

# ── Venn gene table ───────────────────────────────────────────────────────────
_venn_mask = df["sig_cat"] != "Neither"
_venn_cols = {
    "Gene":          df.loc[_venn_mask, "Gene"],
    "Venn_group":    df.loc[_venn_mask, "sig_cat"],
    "logFC_blind":   df.loc[_venn_mask, "logFC_blind"],
    "P.Value_blind": df.loc[_venn_mask, "P.Value_blind"],
    "adj.P.Val_blind": df.loc[_venn_mask, "adj.P.Val_blind"],
    "logFC_aware":   df.loc[_venn_mask, "logFC_aware"],
    "P.Value_aware": df.loc[_venn_mask, "P.Value_aware"],
    "adj.P.Val_aware": df.loc[_venn_mask, "adj.P.Val_aware"],
}
venn_table = pd.DataFrame(_venn_cols).sort_values(
    ["Venn_group", "P.Value_blind"]).reset_index(drop=True)
_venn_table_path = os.path.join(OUTDIR, "venn_gene_table.csv")
venn_table.to_csv(_venn_table_path, index=False)
print(f"  Venn gene table saved: {_venn_table_path}  ({len(venn_table)} genes)")

# ── Standalone adj-p Venn ────────────────────────────────────────────────────
adj_blind_col = "adj.P.Val_blind"
adj_aware_col = "adj.P.Val_aware"
if adj_blind_col in df.columns and adj_aware_col in df.columns:
    sig_blind_adj = df[adj_blind_col] < SIG
    sig_aware_adj = df[adj_aware_col] < SIG
    n_b_adj  = int((sig_blind_adj & ~sig_aware_adj).sum())
    n_a_adj  = int((sig_aware_adj & ~sig_blind_adj).sum())
    n_ab_adj = int((sig_blind_adj & sig_aware_adj).sum())

    fig_v, ax_v = plt.subplots(1, 1, figsize=(5.5, 5))
    draw_proportional_venn(ax_v, n_b_adj, n_a_adj, n_ab_adj,
                           C["blind"], C["aware"], C["both"],
                           title=f"Significant Genes (adj.P < {SIG})")
    _save_venn_genes(OUTDIR, "figA_adjp", {
        "Blind_Only": df.loc[sig_blind_adj & ~sig_aware_adj, "Gene"].tolist(),
        "Aware_Only":  df.loc[sig_aware_adj & ~sig_blind_adj, "Gene"].tolist(),
        "Both":        df.loc[sig_blind_adj & sig_aware_adj,  "Gene"].tolist(),
    })
    fig_v.tight_layout()
    out_v = os.path.join(OUTDIR, "figA_venn_adjp.png")
    fig_v.savefig(out_v, dpi=160, bbox_inches="tight")
    plt.close(fig_v)
    print(f"  Saved {out_v}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Effect Size Shifts: Bland-Altman (left) + Top Discordant Bar (right)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure B: Effect size shifts…")

# All significant genes for B2 dumbbell, sorted by delta_logFC
sig_pool     = df[df["sig_cat"] != "Neither"]
sig_genes_df = sig_pool.sort_values("delta_logFC").reset_index(drop=True)
all_sig_genes = set(sig_genes_df["Gene"])

# Scale figure height so dumbbell labels have room (~0.35 in per gene, min 6)
_b2_height = max(6, len(sig_genes_df) * 0.27)
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 15, "ytick.labelsize": 15,
})
fig, axes = plt.subplots(1, 2, figsize=(19, _b2_height),
                          gridspec_kw={"width_ratios": [1.1, 1.0]})

# ── B1: Bland-Altman — x-axis = logFC_blind ──────────────────────────────────
ax = axes[0]
mu = df["delta_logFC"].mean()
sd = df["delta_logFC"].std()
lo_ba, hi_ba = mu - 1.96 * sd, mu + 1.96 * sd

colored_mask_B1 = df["sig_cat"] != "Neither"
y_margin = float(df.loc[colored_mask_B1, "delta_logFC"].abs().max()) * 1.15
x_margin = float(df.loc[colored_mask_B1, "mean_logFC"].abs().max()) * 1.5 if colored_mask_B1.sum() else 0.5
xlim_B1 = (-x_margin, x_margin)
ylim_B1 = (-y_margin, y_margin)

scatter_grey_rect(ax,
                  df.loc[~colored_mask_B1, "mean_logFC"],
                  df.loc[~colored_mask_B1, "delta_logFC"],
                  xlim_B1, ylim_B1,
                  label=f"Neither (n={(~colored_mask_B1).sum()}, clipped)")

for cat, col in [("Blind Only", C["blind"]), ("Aware Only", C["aware"]),
                 ("Sig in Both", C["both"])]:
    mask = df["sig_cat"] == cat
    if mask.sum():
        ax.scatter(df.loc[mask, "mean_logFC"], df.loc[mask, "delta_logFC"],
                   color=col, alpha=0.92, s=24, edgecolors="none",
                   zorder=5, label=f"{cat} (n={mask.sum()})")

ax.axhline(hi_ba, color="firebrick", lw=1.2, ls="--", label="95% limits of agreement")
ax.axhline(lo_ba, color="firebrick", lw=1.2, ls="--")
ax.axhline(0, color="#555", lw=0.8, ls=":", alpha=0.7)

ax.annotate(f"{G1_LABEL} > {G2_LABEL}\n(larger effect)", xy=(x_margin * 0.55, lo_ba * 0.6),
             color=C["blind"], style="italic")
ax.annotate(f"{G2_LABEL} > {G1_LABEL}\n(larger effect)", xy=(x_margin * 0.55, hi_ba * 0.55),
            color=C["aware"], style="italic")

ax.set_xlim(*xlim_B1)
ax.set_ylim(*ylim_B1)
ax.set_xlabel(f"Mean logFC  [({G1_LABEL} + {G2_LABEL}) / 2]")
ax.set_ylabel(f"\u0394 logFC  [{G2_LABEL} \u2212 {G1_LABEL}]\n(negative = {G1_LABEL} has larger effect)")
ax.set_title(f"Bland-Altman\n"
             f"All Genes: MAB={abs_bias_all:.3f} (% of |logFC|: {mab_pct_all:.1f}%)\n"
             f"Top 10%: MAB={top10_mab_str} (% of |logFC|: {top10_mab_pct_str})",
             )
ax.legend(framealpha=0.9, loc="upper right")

texts_B1 = []
sub_B1_top25 = df[colored_mask_B1 & df["Gene"].isin(all_sig_genes)].copy()
for cat, col in [("Blind Only", C["blind"]), ("Aware Only", C["aware"]),
                 ("Sig in Both", C["both"])]:
    texts_B1.extend(label_all(ax, sub_B1_top25[sub_B1_top25["sig_cat"] == cat],
                               "mean_logFC", "delta_logFC", col=col, fontsize=12))
if texts_B1:
    adjust_text(texts_B1, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))

# ── B2: Dumbbell — significant genes only, colored by sig_cat ─────────────────
ax = axes[1]
sig_r  = sig_genes_df.reset_index(drop=True)
ys_db  = list(range(len(sig_r)))

sig_col_map = {
    "Blind Only":  C["blind"],
    "Aware Only":  C["aware"],
    "Sig in Both": C["both"],
}

# Connecting lines colored by sig_cat
for yi, (_, row) in zip(ys_db, sig_r.iterrows()):
    ax.plot([row["logFC_blind"], row["logFC_aware"]], [yi, yi],
            color=sig_col_map[row["sig_cat"]],
            lw=1.4, zorder=2, alpha=0.55)

# Dots grouped by sig_cat so legend entries are clean
from matplotlib.lines import Line2D
for cat, col in [("Sig in Both", C["both"]), ("Blind Only", C["blind"]),
                 ("Aware Only", C["aware"])]:
    mask = sig_r["sig_cat"] == cat
    if not mask.any():
        continue
    ys_cat = sig_r.index[mask].tolist()
    # Blind — hollow circle
    ax.scatter(sig_r.loc[mask, "logFC_blind"], ys_cat,
               facecolors="none", edgecolors=col, s=55, marker="o",
               linewidths=1.4, zorder=4, label=cat)
    # Aware — filled circle (no extra legend entry)
    ax.scatter(sig_r.loc[mask, "logFC_aware"], ys_cat,
               color=col, s=55, marker="o", zorder=4, edgecolors="none")

ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
ax.set_yticks(ys_db)
ax.set_yticklabels(sig_r["Gene"].values, fontweight="bold")
ax.set_xlabel("logFC")
_mab_sig = sig_r["delta_logFC"].abs().mean()
_avg_mag_sig = ((sig_r["logFC_blind"].abs() + sig_r["logFC_aware"].abs()) / 2).mean()
_mab_pct_sig = _mab_sig / _avg_mag_sig * 100 if _avg_mag_sig > 0 else np.nan
ax.set_title(f"{G1_LABEL} vs {G2_LABEL} logFC\n(Significant genes only, n={len(sig_r)})\n"
             f"MAB={_mab_sig:.3f} (% of |logFC|: {_mab_pct_sig:.1f}%)")

# Combined legend: sig-cat colors + shape key for model
_sig_cat_labels = {
    "Blind Only": f"{G1_LABEL} Only",
    "Aware Only": f"{G2_LABEL} Only",
    "Sig in Both": "Sig in Both",
}
leg_sig  = [mpatches.Patch(color=sig_col_map[c], label=_sig_cat_labels[c])
            for c in ["Sig in Both", "Blind Only", "Aware Only"]
            if (sig_r["sig_cat"] == c).any()]
leg_shape = [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                    markeredgecolor="#555", markeredgewidth=1.4,
                    markersize=7, label=f"o {G1_LABEL}"),
             Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                    markersize=7, label=f"* {G2_LABEL}")]
ax.legend(handles=leg_sig + leg_shape, framealpha=0.9)

fig.tight_layout()
out = os.path.join(OUTDIR, "figB_effect_size_shift.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 15, "ytick.labelsize": 12,
})
print(f"  Saved {out}")

r'''
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Side-by-Side Volcano Plots
# ══════════════════════════════════════════════════════════════════════════════
print("Figure C: Volcano plots…")

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))


def volcano(ax, lfc_col, padj_col, own_sig, other_sig, own_col, other_col,
            title, other_label):
    lfc = df[lfc_col]
    nlp = -np.log10(df[padj_col].clip(1e-300))
    ns  = ~own_sig
    up  = own_sig & (lfc > 0)
    dn  = own_sig & (lfc < 0)

    # Compute axis range from colored (sig) points + other-model diamonds
    colored_vol = own_sig | other_sig
    col_lfc_vol = lfc[colored_vol]
    col_nlp_vol = nlp[colored_vol]

    if colored_vol.sum() > 0:
        x_half = max(float(col_lfc_vol.abs().max()) * 1.5, 0.15)
        y_top  = float(col_nlp_vol.max()) * 1.45
    else:
        x_half = float(lfc.abs().max()) * 1.1
        y_top  = float(nlp.max()) * 1.1

    y_bot = 0.0

    # Grey NS points clipped to colored range
    scatter_grey_rect(ax, lfc[ns], nlp[ns],
                      (-x_half, x_half), (y_bot, y_top),
                      label=f"Not significant")

    # Own sig genes
    ax.scatter(lfc[up], nlp[up], color=C["up"], alpha=0.88, s=60,
               edgecolors="none", label=f"Upregulated (n={up.sum()})", zorder=4)
    ax.scatter(lfc[dn], nlp[dn], color=C["down"], alpha=0.88, s=60,
               edgecolors="none", label=f"Downregulated (n={dn.sum()})", zorder=4)

    # Other-model sig diamonds
    only_other = other_sig & ~own_sig
    ax.scatter(lfc[only_other], nlp[only_other],
               color=other_col, alpha=0.92, s=120, marker="D",
               edgecolors="none",
               label=f"Sig in {other_label} (n={only_other.sum()})", zorder=5)

    sig_line = -np.log10(SIG)
    ax.axhline(sig_line, color="#666", lw=0.9, ls="--", alpha=0.7,
               label=f"{PVAL_DISP_NAME} = {SIG}")
    ax.axvline(0, color="#aaa", lw=0.6, ls=":")

    texts = []
    # Labels: own sig genes
    top_own = nlp[own_sig].nlargest(min(6, own_sig.sum())).index if own_sig.sum() else []
    for i in top_own:
        texts.append(ax.text(lfc[i], nlp[i], df.loc[i, "Gene"],
                             fontsize=11.2, fontweight="bold", zorder=7))
    # Labels: other-model sig (italic)
    for i in df.index[only_other]:
        texts.append(ax.text(lfc[i], nlp[i], df.loc[i, "Gene"],
                             fontsize=10, style="italic", color=other_col, zorder=7))
    
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaa"))

    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel("logFC")
    ax.set_ylabel(f"\u2212log\u2081\u2080({PVAL_DISP_NAME})")
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, framealpha=0.9, loc="upper right")


volcano(axes[0],
        lfc_col="logFC_blind", padj_col=f"{PVAL_BASE_COL}_blind",
        own_sig=df["sig_blind"], other_sig=df["sig_aware"],
        own_col=C["blind"], other_col=C["aware"],
        title=f"{G1_LABEL} Model\n({int(cats['Blind Only'])} significant, not in {G2_LABEL})",
        other_label=G2_LABEL)

volcano(axes[1],
        lfc_col="logFC_aware", padj_col=f"{PVAL_BASE_COL}_aware",
        own_sig=df["sig_aware"], other_sig=df["sig_blind"],
        own_col=C["aware"], other_col=C["blind"],
        title=f"{G2_LABEL} Model\n({int(cats['Aware Only'])} significant, not in {G1_LABEL})",
        other_label=G1_LABEL)

fig.tight_layout()
out = os.path.join(OUTDIR, "figC_volcanos.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Direction Flips (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure D: Direction flips…")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

flipped = df[df["sign_flip"]]
no_flip = df[~df["sign_flip"]]

# ── D1: logFC scatter — colored = flipped + sig genes; grey = consistent ─────
ax = axes[0]

# Colored in D1: flipped triangles + any sig stars
colored_mask_D1 = df["sign_flip"] | (df["sig_cat"] != "Neither")
col_x_D1 = df.loc[colored_mask_D1, "logFC_blind"]
col_y_D1 = df.loc[colored_mask_D1, "logFC_aware"]
lim_D1 = colored_sym_lim(col_x_D1, col_y_D1, pad=0.30)

scatter_grey_sym(ax,
                 df.loc[~colored_mask_D1, "logFC_blind"],
                 df.loc[~colored_mask_D1, "logFC_aware"],
                 lim_D1,
                 label=f"Consistent, not sig (n={(~colored_mask_D1).sum()}, clipped)")

ax.scatter(flipped["logFC_blind"], flipped["logFC_aware"],
           color=C["flip"], alpha=0.70, s=22, marker="^",
           edgecolors="none",
           label=f"Sign reversal (n={n_flip})", zorder=4)

ax.plot([-lim_D1, lim_D1], [-lim_D1, lim_D1], "k--", lw=0.8, alpha=0.45)
ax.axhline(0, color="#aaa", lw=0.5, ls=":")
ax.axvline(0, color="#aaa", lw=0.5, ls=":")
# Light shading on discordant quadrants
ax.fill_between([-lim_D1, 0], [0] * 2, [lim_D1] * 2,
                color="#fff0ee", alpha=0.4, zorder=0)
ax.fill_between([0, lim_D1], [-lim_D1] * 2, [0] * 2,
                color="#fff0ee", alpha=0.4, zorder=0)
ax.set_xlim(-lim_D1, lim_D1)
ax.set_ylim(-lim_D1, lim_D1)
ax.set_xlabel(f"logFC  [{G1_LABEL}]")
ax.set_ylabel(f"logFC  [{G2_LABEL}]")
ax.set_title("Sign Reversals on logFC Scatter\n(shaded = discordant quadrants)",
             fontsize=15)
ax.legend(markerscale=1.2, fontsize=12, framealpha=0.88, loc="upper left")

if len(flipped) > 0:
    texts_D1 = label_genes(ax, flipped["logFC_blind"], flipped["logFC_aware"],
                           flipped["Gene"], n=min(8, len(flipped)),
                           criterion=flipped["logFC_blind"].abs(), fontsize=9.8)
    if texts_D1:
        adjust_text(texts_D1, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaa"))

# ── D2: Δ logFC vs Blind logFC — flip anatomy ─────────────────────────────────
ax = axes[1]

colored_mask_D2 = df["sign_flip"] | (df["sig_cat"] != "Neither")
col_x_D2 = df.loc[colored_mask_D2, "logFC_blind"]
col_y_D2 = df.loc[colored_mask_D2, "delta_logFC"]
lim_D2_x = float(col_x_D2.abs().max()) * 1.30 if colored_mask_D2.sum() else 0.5
lim_D2_y = float(col_y_D2.abs().max()) * 1.30 if colored_mask_D2.sum() else 0.5
xlim_D2 = (-lim_D2_x, lim_D2_x)
ylim_D2 = (-lim_D2_y, lim_D2_y)

scatter_grey_rect(ax,
                  df.loc[~colored_mask_D2, "logFC_blind"],
                  df.loc[~colored_mask_D2, "delta_logFC"],
                  xlim_D2, ylim_D2,
                  label=f"No flip, not sig (n={(~colored_mask_D2).sum()}, clipped)")

ax.scatter(flipped["logFC_blind"], flipped["delta_logFC"],
           color=C["flip"], alpha=0.70, s=22, marker="^",
           edgecolors="none",
           label=f"Sign reversal (n={n_flip})", zorder=4)

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
ax.axvline(0, color="#aaa", lw=0.6, ls=":")
ax.set_xlim(*xlim_D2)
ax.set_ylim(*ylim_D2)
ax.set_xlabel(f"logFC  [{G1_LABEL}]  (starting direction)")
ax.set_ylabel(f"\u0394 logFC  [{G2_LABEL} \u2212 {G1_LABEL}]")
ax.set_title("Effect Size Shift by Starting Direction\n(reversals cross the dashed zero line)",
             fontsize=15)
ax.legend(fontsize=12, framealpha=0.88)

if len(flipped) > 0:
    texts_D2 = label_genes(ax, flipped["logFC_blind"], flipped["delta_logFC"],
                           flipped["Gene"], n=min(8, len(flipped)),
                           criterion=flipped["delta_logFC"].abs(), fontsize=9.8)
    if texts_D2:
        adjust_text(texts_D2, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaa"))

# ── D3: Concordance pie ────────────────────────────────────────────────────────
ax = axes[2]
up_up = ((df["logFC_blind"] > 0) & (df["logFC_aware"] > 0) & ~df["sign_flip"]).sum()
dn_dn = ((df["logFC_blind"] < 0) & (df["logFC_aware"] < 0) & ~df["sign_flip"]).sum()
up_dn = ((df["logFC_blind"] > 0) & (df["logFC_aware"] < 0)).sum()
dn_up = ((df["logFC_blind"] < 0) & (df["logFC_aware"] > 0)).sum()

slices   = [up_up, dn_dn, up_dn, dn_up]
labels   = [f"Both up\n(n={up_up})",
            f"Both down\n(n={dn_dn})",
            f"Reversal\n(n={up_dn})",
            f"Reversal\n(n={dn_up})"]
pie_cols = ["#c8e6c9", "#bbdefb", C["flip"], "#ff8a65"]
explode  = [0, 0, 0.07, 0.07]
wedges, texts, autotexts = ax.pie(
    slices, labels=labels, colors=pie_cols, explode=explode,
    autopct="%1.0f%%", startangle=90, pctdistance=0.75,
    textprops={"fontsize": 14}, labeldistance=1.10)
for at in autotexts:
    at.set_fontsize(14); at.set_fontweight("bold")
ax.set_title(
    f"Direction Agreement\n"
    f"{100*(up_dn+dn_up)/len(df):.1f}% of genes reverse sign",
    fontsize=15)

fig.tight_layout()
out = os.path.join(OUTDIR, "figD_direction_flips.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

'''
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E — Summary dashboard
# ══════════════════════════════════════════════════════════════════════════════
print("Figure E: Summary dashboard…")

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)

# E1: Key metrics text
ax1 = fig.add_subplot(gs[0])
ax1.axis("off")
metrics = [
    ("Shared genes analysed",    f"{len(df):,}"),
    (f"{G1_LABEL}-exclusive top-1000", f"{len(only_blind):,}"),
    (f"{G2_LABEL}-exclusive top-1000", f"{len(only_aware):,}"),
    ("", ""),
    (f"Sig ({G1_LABEL} only)",  f"{int(cats['Blind Only'])} genes"),
    (f"Sig ({G2_LABEL} only)",  f"{int(cats['Aware Only'])} genes"),
    ("Sig in BOTH",       f"{int(cats['Sig in Both'])} genes  \u2190 ZERO"),
    ("", ""),
    ("Direction flips",   f"{n_flip}/{len(df)} ({100*n_flip/len(df):.1f}%)"),
    ("logFC Pearson r",   f"{r_lfc_pe:.3f}"),
    ("logFC Spearman r",  f"{r_lfc_sp:.3f}"),
    ("adj.P Spearman r",  f"{r_nla_sp:.3f}  (weak)"),
    ("z.std Spearman r",  f"{r_z_sp:.3f}"),
    ("", ""),
    ("Mean \u0394 logFC",  f"{df['delta_logFC'].mean():+.4f}  (blind > aware)"),
    ("SD  \u0394 logFC",   f"{df['delta_logFC'].std():.4f}"),
]
y_start = 0.98
for key, val in metrics:
    if not key:
        y_start -= 0.03
        continue
    ax1.text(0.02, y_start, key + ":", transform=ax1.transAxes,
             fontsize=12.8, fontweight="bold", va="top")
    ax1.text(0.52, y_start, val, transform=ax1.transAxes,
             fontsize=12.8, va="top", color="#333")
    y_start -= 0.06
ax1.set_title("Key Metrics", fontsize=15, pad=8)

# E2: Correlation bar
ax2 = fig.add_subplot(gs[1])
metrics_bar = ["logFC\nPearson", "logFC\nSpearman", "adj.P\nSpearman", "z.std\nSpearman"]
vals_bar    = [r_lfc_pe, r_lfc_sp, r_nla_sp, r_z_sp]
bar_c       = ["#4c956c" if v >= 0.7 else "#e9c46a" if v >= 0.5 else "#e76f51"
               for v in vals_bar]
bars2 = ax2.bar(metrics_bar, vals_bar, color=bar_c, edgecolor="white",
                alpha=0.88, width=0.55)
for bar, val in zip(bars2, vals_bar):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
             ha="center", fontsize=14, fontweight="bold")
ax2.axhline(0.7, color="#4c956c", lw=1, ls="--", alpha=0.7, label="r=0.70 (good)")
ax2.axhline(0.5, color="#e9c46a", lw=1, ls="--", alpha=0.7, label="r=0.50 (fair)")
ax2.set_ylim(0, 1.1)
ax2.set_ylabel("Spearman / Pearson r")
ax2.set_title("Cross-Model Correlation\nby Metric", fontsize=15)
ax2.legend(fontsize=11.2)

# E3: Stacked bar
ax3 = fig.add_subplot(gs[2])
cat_vals = [int(cats[c]) for c in ["Sig in Both", "Blind Only", "Aware Only", "Neither"]]
cat_cols = [C["both"], C["blind"], C["aware"], GREY_COL]
cat_labs = ["Sig in Both", "Blind Only", "Aware Only", "Neither"]
bottom = 0
for val, col, lab in zip(cat_vals, cat_cols, cat_labs):
    ax3.bar(0, val, bottom=bottom, color=col, width=0.55,
            edgecolor="white", alpha=0.88, label=f"{lab} (n={val})")
    if val > 20:
        ax3.text(0, bottom + val / 2, str(val), ha="center", va="center",
                 fontsize=15, fontweight="bold", color="white")
    bottom += val
ax3.set_xlim(-0.6, 0.6)
ax3.set_xticks([])
ax3.set_ylabel("Number of genes")
ax3.set_title(f"Significance Category\n(n={len(df)} shared genes)", fontsize=15)
ax3.legend(loc="upper right", fontsize=11.2, framealpha=0.9)

# E4: Flip pie
ax4 = fig.add_subplot(gs[3])
ax4.pie([n_flip, len(df) - n_flip],
        labels=[f"Flipped\n(n={n_flip})", f"Consistent\n(n={len(df)-n_flip})"],
        colors=[C["flip"], "#d8e2dc"],
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        textprops={"fontsize": 14},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax4.set_title(f"Direction Consistency\n({100*n_flip/len(df):.1f}% flip sign)", fontsize=15)

fig.tight_layout()
out = os.path.join(OUTDIR, "figE_summary_dashboard.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")


# ── Export summary CSV (same schema as other visualize_*.py scripts) ─────────
wholebrain_row = {
    "n_genes":                  len(df),
    "n_sig_G1_only":            int(cats.get("Blind Only", 0)),
    "n_sig_G2_only":            int(cats.get("Aware Only", 0)),
    "n_sig_both":               int(cats.get("Sig in Both", 0)),
    "n_flip":                   n_flip,
    "pct_flip":                 round(100 * n_flip / len(df), 2),
    "pearson_r_all":            round(r_lfc_pe, 4),
    "mab_all":                  round(abs_bias_all, 4),
    "pearson_r_top10":          round(r_lfc_pe_top10, 4) if not np.isnan(r_lfc_pe_top10) else np.nan,
    "mab_top10":                round(top10_mab, 4) if not np.isnan(top10_mab) else np.nan,
    "spearman_r_neglogp":       round(r_nla_sp, 4),
    "spearman_r_neglogp_top10": round(r_nla_sp_top10, 4) if not np.isnan(r_nla_sp_top10) else np.nan,
    "spearman_r_lfc":           round(r_lfc_sp, 4),
    "mab_pct_all":              round(mab_pct_all, 2) if not np.isnan(mab_pct_all) else np.nan,
    "mab_pct_top10":            round(top10_mab_pct, 2) if not np.isnan(top10_mab_pct) else np.nan,
    "cell_type":                "WholeBrain",
    "model":                    "dream",
    "annotation":               "blind_vs_quint",
}

# ── Cell-type-specific blind vs quint stats ───────────────────────────────────
from pathlib import Path as _Path

if USE_NO_O_DATA:
    _NO_O_PATH   = _Path(SCRIPT_DIR) / "data_no_O"
    _GLOBAL_PATH = _NO_O_PATH / "Global_CT_Analysis"
    _LOCAL_PATH  = _NO_O_PATH / "Local_Regional_Analysis"

    # Derive G1/G2 annotation tags from COMPARISON ("blind", "napari", or "quint")
    _g1_ann, _g2_ann = COMPARISON.split("_vs_")

    def _ct_file(stem, ann):
        """Return Path for a CT file given stem and annotation tag."""
        if ann == "blind":
            return _LOCAL_PATH  / f"Jan_26BASE_{stem}_dream_blind.csv"
        return     _GLOBAL_PATH / f"Jan_26BASE_{stem}_dream_{ann}.csv"

    _CT_STEMS = {
        "Astrocytes":    "Astrocytes_Astrocytes",
        "Microglia":     "Microglia_Microglia",
        "Astrocytes_CxHp": "Astrocytes.cortex.hippocampus_Astrocytes_cortex_hippocampus",
    }
    _CT_PAIRS = {
        ct: {"blind_single": _ct_file(stem, _g1_ann),
             "quint_single": _ct_file(stem, _g2_ann)}
        for ct, stem in _CT_STEMS.items()
    }
    _USE_SINGLE_FILES = True
else:
    _LMM_BASE  = _Path(SCRIPT_DIR).parent / "LMM_extract" / "Seurat_&_Dream_updated_2_5"
    _DATA_PATH  = _LMM_BASE / "Global_CT_Analysis"
    _LOCAL_PATH = _LMM_BASE / "Local_Regional_Analysis"

    _CT_PAIRS = {
        "Astrocytes": {
            "blind_up":   _LOCAL_PATH / "Jan_26UP_Astrocytes_dream_blind.csv",
            "blind_down": _LOCAL_PATH / "Jan_26DOWN_Astrocytes_dream_blind.csv",
            "quint_up":   _DATA_PATH  / "Jan_26UP_Astrocytes_dream_quint.csv",
            "quint_down": _DATA_PATH  / "Jan_26DOWN_Astrocytes_dream_quint.csv",
        },
        "Microglia": {
            "blind_up":   _LOCAL_PATH / "Jan_26UP_Microglia_dream_blind.csv",
            "blind_down": _LOCAL_PATH / "Jan_26DOWN_Microglia_dream_blind.csv",
            "quint_up":   _DATA_PATH  / "Jan_26UP_Microglia_dream_quint.csv",
            "quint_down": _DATA_PATH  / "Jan_26DOWN_Microglia_dream_quint.csv",
        },
    }
    _USE_SINGLE_FILES = False


def _long_path(p):
    r"""Return a string path with Windows extended-length prefix when needed.

    Windows caps normal paths at 260 chars; OneDrive + deep subdirs blow past
    that. The \\?\ prefix opts the call into extended-length handling.
    """
    s = str(p)
    if os.name == "nt" and len(s) >= 260 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(s)
    return s


def _load_ct_data(paths, use_single):
    """Load blind and quint dataframes for one cell type."""
    if use_single:
        blind_raw = pd.read_csv(_long_path(paths["blind_single"]))
        quint_raw = pd.read_csv(_long_path(paths["quint_single"]))
    else:
        blind_raw = _load_and_concat(paths["blind_up"], paths["blind_down"])
        quint_raw = _load_and_concat(paths["quint_up"], paths["quint_down"])
    return blind_raw, quint_raw


def _load_and_concat(up_path, down_path):
    """Load UP and DOWN direction files, concatenate, resolve duplicate genes by |logFC|."""
    combined = pd.concat([pd.read_csv(_long_path(up_path)),
                          pd.read_csv(_long_path(down_path))], ignore_index=True)
    combined = (combined
                .assign(_abs_lfc=combined["logFC"].abs())
                .sort_values("_abs_lfc", ascending=False)
                .drop_duplicates("Gene")
                .drop(columns="_abs_lfc")
                .reset_index(drop=True))
    return combined


def _compute_bvq_stats(blind_raw, quint_raw):
    """Compute blind-vs-quint summary stats from two gene-level raw dataframes."""
    b = blind_raw.rename(columns={c: c + "_blind" for c in blind_raw.columns if c != "Gene"})
    q = quint_raw.rename(columns={c: c + "_aware" for c in quint_raw.columns if c != "Gene"})
    d = b.merge(q, on="Gene", how="inner")

    pv_b = f"{PVAL_BASE_COL}_blind"
    pv_a = f"{PVAL_BASE_COL}_aware"

    d["sig_blind"]   = (d[pv_b] < SIG) & (d["logFC_blind"].abs() >= LOGFC_THRESH)
    d["sig_aware"]   = (d[pv_a] < SIG) & (d["logFC_aware"].abs() >= LOGFC_THRESH)
    d["sign_flip"]   = np.sign(d["logFC_blind"]) != np.sign(d["logFC_aware"])
    d["delta_logFC"] = d["logFC_aware"] - d["logFC_blind"]
    d["nlp_blind"]   = -np.log10(d[pv_b].clip(1e-300))
    d["nlp_aware"]   = -np.log10(d[pv_a].clip(1e-300))

    def _sc(row):
        if row.sig_blind and row.sig_aware: return "Sig in Both"
        if row.sig_blind:                   return "Blind Only"
        if row.sig_aware:                   return "Aware Only"
        return "Neither"

    d["sig_cat"] = d.apply(_sc, axis=1)
    c = d["sig_cat"].value_counts().reindex(
        ["Sig in Both", "Blind Only", "Aware Only", "Neither"], fill_value=0)
    nf = int(d["sign_flip"].sum())

    r_lfc_sp, _ = stats.spearmanr(d["logFC_blind"], d["logFC_aware"])
    r_lfc_pe, _ = stats.pearsonr(d["logFC_blind"],  d["logFC_aware"])
    r_nla_sp, _ = stats.spearmanr(d["nlp_blind"],   d["nlp_aware"])

    _max_nla = d[["nlp_blind", "nlp_aware"]].max(axis=1)
    _t10_nla = _max_nla.nlargest(max(1, len(d) // 10)).index
    r_nla_sp_top10 = stats.spearmanr(d.loc[_t10_nla, "nlp_blind"],
                                      d.loc[_t10_nla, "nlp_aware"])[0] \
                     if len(_t10_nla) >= 3 else np.nan

    _max_lfc = d[["logFC_blind", "logFC_aware"]].abs().max(axis=1)
    _t10_lfc = _max_lfc.nlargest(max(1, len(d) // 10)).index
    r_lfc_pe_top10 = stats.pearsonr(d.loc[_t10_lfc, "logFC_blind"],
                                     d.loc[_t10_lfc, "logFC_aware"])[0] \
                     if len(_t10_lfc) >= 3 else np.nan

    ab_all = d["delta_logFC"].abs().mean()
    am_all = ((d["logFC_blind"].abs() + d["logFC_aware"].abs()) / 2).mean()
    mp_all = ab_all / am_all * 100 if am_all > 0 else np.nan

    ab_t10 = d.loc[_t10_lfc, "delta_logFC"].abs().mean()
    am_t10 = ((d.loc[_t10_lfc, "logFC_blind"].abs() +
               d.loc[_t10_lfc, "logFC_aware"].abs()) / 2).mean()
    mp_t10 = ab_t10 / am_t10 * 100 if am_t10 > 0 else np.nan

    return {
        "n_genes":                  len(d),
        "n_sig_G1_only":            int(c.get("Blind Only", 0)),
        "n_sig_G2_only":            int(c.get("Aware Only", 0)),
        "n_sig_both":               int(c.get("Sig in Both", 0)),
        "n_flip":                   nf,
        "pct_flip":                 round(100 * nf / len(d), 2),
        "pearson_r_all":            round(r_lfc_pe, 4),
        "mab_all":                  round(ab_all, 4),
        "pearson_r_top10":          round(r_lfc_pe_top10, 4) if not np.isnan(r_lfc_pe_top10) else np.nan,
        "mab_top10":                round(ab_t10, 4) if not np.isnan(ab_t10) else np.nan,
        "spearman_r_neglogp":       round(r_nla_sp, 4),
        "spearman_r_neglogp_top10": round(r_nla_sp_top10, 4) if not np.isnan(r_nla_sp_top10) else np.nan,
        "spearman_r_lfc":           round(r_lfc_sp, 4),
        "mab_pct_all":              round(mp_all, 2) if not np.isnan(mp_all) else np.nan,
        "mab_pct_top10":            round(mp_t10, 2) if not np.isnan(mp_t10) else np.nan,
        "model":                    "dream",
        "annotation":               "blind_vs_quint",
    }


print("\nComputing cell-type-specific blind vs quint stats…")
ct_summary_rows = []
for _ct, _paths in _CT_PAIRS.items():
    try:
        _blind_raw, _quint_raw = _load_ct_data(_paths, _USE_SINGLE_FILES)
        _row = _compute_bvq_stats(_blind_raw, _quint_raw)
        _row["cell_type"] = _ct
        ct_summary_rows.append(_row)
        print(f"  {_ct}: {_row['n_genes']} shared genes, "
              f"Pearson r={_row['pearson_r_all']:.4f}, "
              f"pct_flip={_row['pct_flip']:.1f}%")
    except Exception as _e:
        print(f"  WARNING: {_ct} failed — {_e}")

all_summary_rows = [wholebrain_row] + ct_summary_rows
summary_df = pd.DataFrame(all_summary_rows)
summary_path = os.path.join(OUTDIR, "summary_all_comparisons_AI.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary table saved to: {summary_path}  ({len(all_summary_rows)} rows)")

print("\nAll figures saved to:", OUTDIR)
print("Done.")
