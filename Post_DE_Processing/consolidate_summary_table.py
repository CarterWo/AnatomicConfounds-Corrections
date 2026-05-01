#!/usr/bin/env python3
"""
consolidate_summary_table.py
============================
Reads the summary_all_comparisons.csv files produced by each visualize_*.py
script and combines them into an Excel workbook with two sheets:

Sheet 1 — "G1_vs_G2":
  Gene-list comparisons (G1=UP, G2=DOWN) across models and annotations.

Sheet 2 — "Annotation_Comparisons":
  Within-model comparisons of annotation methods (blind vs napari,
  blind vs quint, napari vs quint).  Columns are renamed to reflect
  the two annotation methods rather than G1/G2.

Output: figures/consolidated_summary.xlsx
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_BASE = Path(r"C:\Users\woods\OneDrive - University of Missouri\General - Lin Brain Lab - Ogrp\Carter Woods\CosMx_Processing\Manuscript\LMM_Model_Comparison\S&Du_(search)")

# ── Source CSVs ──────────────────────────────────────────────────────────────
# G1 vs G2 sources
G1G2_SOURCES = [
    FIG_BASE / 'g1g2'              / 'summary_all_comparisons_dLMM.csv',
    FIG_BASE / 'g1g2'   / 'summary_all_comparisons_WD.csv',
    FIG_BASE / 'g1g2'     / 'summary_all_comparisons_LR.csv',
]

# Annotation-comparison sources
ANNOT_SOURCES = [
    FIG_BASE / 'anatomic'   /'blind_vs_quint'      / 'summary_all_comparisons_AI.csv',
    FIG_BASE / 'anatomic'   /'blind_vs_napari'  / 'summary_all_comparisons_AI.csv',
    
]

# ── Model-name mapping ──────────────────────────────────────────────────────
MODEL_MAP = {
    "dream":     "LMM (DREAM)",
    "wilcoxon":  "Wilcoxon",
    "deseq2":    "DESeq2",
    "seurat":    "Logistic Regression",
}

# ── Helper: load & concatenate a list of source CSVs ────────────────────────
def _load_sources(sources):
    frames = []
    for src in sources:
        if not src.exists():
            print(f"  WARNING: {src} not found — skipping")
            continue
        df = pd.read_csv(src)
        df["source_file"] = src.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Load data ───────────────────────────────────────────────────────────────
g1g2_df  = _load_sources(G1G2_SOURCES)
annot_df = _load_sources(ANNOT_SOURCES)

if g1g2_df.empty and annot_df.empty:
    raise FileNotFoundError("No summary CSVs found. Run the visualize_*.py scripts first.")

# ── Standardise shared columns ──────────────────────────────────────────────
for df in (g1g2_df, annot_df):
    if df.empty:
        continue
    df["overarching_model"] = df["model"].map(MODEL_MAP).fillna(df["model"])
    df["anatomic_term"] = df["annotation"]

# ── Sheet 1: G1 vs G2 ──────────────────────────────────────────────────────
id_cols = ["overarching_model", "anatomic_term", "cell_type"]
stat_cols = [
    "n_genes",
    "n_sig_G1_only", "n_sig_G2_only", "n_sig_both",
    "pearson_r_all",   "pearson_r_top10",
    "spearman_r_lfc",
    "spearman_r_neglogp", "spearman_r_neglogp_top10",
    "mab_all",         "mab_top10",
    "mab_pct_all",     "mab_pct_top10",
]
meta_cols = ["model", "annotation", "source_file"]

if not g1g2_df.empty:
    sheet_g1g2 = g1g2_df[id_cols + stat_cols + meta_cols].copy()
    sheet_g1g2 = sheet_g1g2.sort_values(id_cols).reset_index(drop=True)
else:
    sheet_g1g2 = pd.DataFrame()

# ── Sheet 2: Annotation comparisons ────────────────────────────────────────
# Derive comparison label from the source directory name
# (e.g. "main_unadj_no_O" → blind_vs_quint, "blind_vs_napari_no_O" → blind_vs_napari)
ANNOT_COMPARISON_MAP = {

    "blind_vs_napari":   "blind_vs_napari",
    "blind_vs_quint":   "blind_vs_quint",
}

if not annot_df.empty:
    annot_df["comparison"] = annot_df["source_file"].map(ANNOT_COMPARISON_MAP).fillna(annot_df["annotation"])

    # Derive method_A / method_B from the comparison field
    split = annot_df["comparison"].str.split("_vs_", n=1, expand=True)
    annot_df["method_A"] = split[0]
    annot_df["method_B"] = split[1]

    # Rename G1/G2 columns to annotation-method-specific names
    annot_stat_cols = [
        "n_genes",
        "n_sig_method_A_only", "n_sig_method_B_only", "n_sig_both",
        "pearson_r_all",   "pearson_r_top10",
        "spearman_r_lfc",
        "spearman_r_neglogp", "spearman_r_neglogp_top10",
        "mab_all",         "mab_top10",
        "mab_pct_all",     "mab_pct_top10",
    ]
    annot_id_cols = ["overarching_model", "comparison", "method_A", "method_B", "cell_type"]
    annot_meta_cols = ["model", "annotation", "source_file"]

    annot_df = annot_df.rename(columns={
        "n_sig_G1_only": "n_sig_method_A_only",
        "n_sig_G2_only": "n_sig_method_B_only",
    })

    sheet_annot = annot_df[annot_id_cols + annot_stat_cols + annot_meta_cols].copy()
    sheet_annot = sheet_annot.sort_values(annot_id_cols).reset_index(drop=True)
else:
    sheet_annot = pd.DataFrame()

# ── Sheet 3: Formatted G1 vs G2 (mirrors Annotation_Comparisons style) ─────
if not g1g2_df.empty:
    fmt = g1g2_df.copy()
    # Rename WholeBrain -> All
    fmt["cell_type"] = fmt["cell_type"].replace("WholeBrain", "All")
    # Drop Pearson, n_genes, mab_all, and metadata columns; rename the rest
    fmt = fmt.rename(columns={
        "overarching_model":         "Model",
        "anatomic_term":             "Annotation",
        "n_sig_G1_only":             "N Sig. (G1)",
        "n_sig_both":                "N Sig. (G1&G2)",
        "n_sig_G2_only":             "N Sig. (G2)",
        "spearman_r_lfc":            "r (logFC)",
        "spearman_r_neglogp":        "r [-log10(p)]",
        "spearman_r_neglogp_top10":  "r [-log10(p) (T10%)]",
        "mab_all":                   "MAB (All)",
        "mab_top10":                 "MAB (T10%)",
        "mab_pct_all":               "% of logFC",
        "mab_pct_top10":             "% of logFC (T10%)",
    })
    fmt_cols = [
        "Model", "Annotation", "cell_type",
        "N Sig. (G1)", "N Sig. (G1&G2)", "N Sig. (G2)",
        "r (logFC)", "r [-log10(p)]", "r [-log10(p) (T10%)]",
        "MAB (All)", "MAB (T10%)", "% of logFC", "% of logFC (T10%)",
    ]
    sheet_g1g2_fmt = fmt[fmt_cols].copy()
    sheet_g1g2_fmt = sheet_g1g2_fmt.sort_values(["Model", "Annotation", "cell_type"]).reset_index(drop=True)
else:
    sheet_g1g2_fmt = pd.DataFrame()

# ── Sheet 4: Formatted Annotation Comparisons (mirrors user edits) ─────────
if not annot_df.empty:
    afmt = annot_df.copy()
    afmt["cell_type"] = afmt["cell_type"].replace("WholeBrain", "All")
    # Drop Astrocytes_CxHp rows
    afmt = afmt[~afmt["cell_type"].str.contains("CxHp", na=False)]
    afmt = afmt.rename(columns={
        "n_sig_method_A_only":       "N Sig. (A)",
        "n_sig_both":                "N Sig. (A&B)",
        "n_sig_method_B_only":       "N Sig. (B)",
        "spearman_r_lfc":            "r (logFC)",
        "spearman_r_neglogp":        "r [-log10(p)]",
        "spearman_r_neglogp_top10":  "r [-log10(p) (T10%)]",
        "mab_all":                   "MAB (All)",
        "mab_top10":                 "MAB (T10%)",
        "mab_pct_all":               "% of logFC",
        "mab_pct_top10":             "% of logFC (T10%)",
    })
    afmt_cols = [
        "method_A", "method_B", "cell_type",
        "N Sig. (A)", "N Sig. (A&B)", "N Sig. (B)",
        "r (logFC)", "r [-log10(p)]", "r [-log10(p) (T10%)]",
        "MAB (All)", "MAB (T10%)", "% of logFC", "% of logFC (T10%)",
    ]
    sheet_annot_fmt = afmt[afmt_cols].copy()
    sheet_annot_fmt = sheet_annot_fmt.sort_values(["method_A", "method_B", "cell_type"]).reset_index(drop=True)
else:
    sheet_annot_fmt = pd.DataFrame()

# ── Save to Excel workbook ──────────────────────────────────────────────────
out_path = FIG_BASE / "consolidated_summary.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    if not sheet_g1g2.empty:
        sheet_g1g2.to_excel(writer, sheet_name="G1_vs_G2", index=False)
    if not sheet_annot.empty:
        sheet_annot.to_excel(writer, sheet_name="Annotation_Comparisons", index=False)
    if not sheet_g1g2_fmt.empty:
        sheet_g1g2_fmt.to_excel(writer, sheet_name="G1_vs_G2_Formatted", index=False)
    if not sheet_annot_fmt.empty:
        sheet_annot_fmt.to_excel(writer, sheet_name="Annot_Comp_Formatted", index=False)

n_total = len(sheet_g1g2) + len(sheet_annot)
print(f"Consolidated {n_total} rows into sheets.")
print(f"  G1_vs_G2 (raw):            {len(sheet_g1g2)} rows")
print(f"  Annotation_Comparisons:    {len(sheet_annot)} rows")
print(f"  G1_vs_G2_Formatted:        {len(sheet_g1g2_fmt)} rows")
print(f"  Annot_Comp_Formatted:      {len(sheet_annot_fmt)} rows")
print(f"Saved to: {out_path}")
print()
if not sheet_g1g2_fmt.empty:
    print("-- G1_vs_G2_Formatted --")
    print(sheet_g1g2_fmt.to_string(index=False))
    print()
if not sheet_annot_fmt.empty:
    print("-- Annot_Comp_Formatted --")
    print(sheet_annot_fmt.to_string(index=False))
