#!/usr/bin/env python3
"""
compare_g1_g2.py

Lightweight numeric G1-vs-G2 comparison for the independent Zhuang-ABCA-1
validation, mirroring the manuscript's own composition-engineering
comparison metrics (DEG overlap / Venn counts, p-value correlation, mean
absolute bias in logFC) without the full plotting pipeline in
Post_DE_Processing/visualize_g1_g2_comparison.py (which is tied to the
real dataset's specific directory layout and lmm_comparison_utils module).

Tests whether anatomically-aware modeling reduces G1-vs-G2 discordance on
an independent MERFISH platform, the same qualitative pattern reported for
the manuscript's own real CosMx composition-engineering experiment --
Reviewer 2 Item 1's generalizability question.

Usage:
    python3 compare_g1_g2.py <de_results_root>
      de_results_root: .../independent_validation/de_results
        (containing g1/ and g2/ subfolders, each with
        Global_CT_Analysis/, Local_Regional_Analysis/, Pseudobulk_Validation/)
"""
import sys
import glob
import os
import numpy as np
import pandas as pd

SIG = 0.05

MODELS = {
    "dream_blind":    ("Local_Regional_Analysis", "INDEP_{scenario}_dream_blind.csv"),
    "dream_napari":   ("Global_CT_Analysis",       "INDEP_{scenario}_dream_napari.csv"),
    "dream_quint":    ("Global_CT_Analysis",       "INDEP_{scenario}_dream_quint.csv"),
    "seurat_blind":   ("Global_CT_Analysis",       "INDEP_{scenario}_seurat_blind.csv"),
    "seurat_napari":  ("Global_CT_Analysis",       "INDEP_{scenario}_seurat_napari.csv"),
    "seurat_quint":   ("Global_CT_Analysis",       "INDEP_{scenario}_seurat_quint.csv"),
    "wilcoxon":       ("Global_CT_Analysis",       "INDEP_{scenario}_wilcoxon.csv"),
    "deseq2_pb":      ("Pseudobulk_Validation",    "DESEQ2INDEP_{scenario}_PB.csv"),
}

def load_model(root, scenario, model):
    subdir, pattern = MODELS[model]
    path = os.path.join(root, scenario, subdir, pattern.format(scenario=scenario))
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Normalize column names across model output formats
    if "logFC" not in df.columns and "log2FoldChange" in df.columns:
        df = df.rename(columns={"log2FoldChange": "logFC"})
    if "P.Value" not in df.columns and "pvalue" in df.columns:
        df = df.rename(columns={"pvalue": "P.Value"})
    if "adj.P.Val" not in df.columns and "padj" in df.columns:
        df = df.rename(columns={"padj": "adj.P.Val"})
    if "Gene" not in df.columns:
        return None
    keep = [c for c in ["Gene", "logFC", "P.Value", "adj.P.Val"] if c in df.columns]
    return df[keep].dropna(subset=["Gene"])


def compare(root, model, use_adjusted=False):
    g1 = load_model(root, "g1", model)
    g2 = load_model(root, "g2", model)
    if g1 is None or g2 is None:
        return None
    pcol = "adj.P.Val" if use_adjusted else "P.Value"
    if pcol not in g1.columns or pcol not in g2.columns:
        return None
    merged = pd.merge(g1, g2, on="Gene", suffixes=("_g1", "_g2"), how="inner")
    if len(merged) == 0:
        return None
    sig_g1 = merged[f"{pcol}_g1"] < SIG
    sig_g2 = merged[f"{pcol}_g2"] < SIG
    n_g1_only = int((sig_g1 & ~sig_g2).sum())
    n_g2_only = int((sig_g2 & ~sig_g1).sum())
    n_both    = int((sig_g1 & sig_g2).sum())
    n_union   = n_g1_only + n_g2_only + n_both
    overlap_frac = n_both / n_union if n_union > 0 else np.nan

    valid = merged["logFC_g1"].notna() & merged["logFC_g2"].notna()
    logfc_corr = merged.loc[valid, "logFC_g1"].corr(merged.loc[valid, "logFC_g2"])
    mean_abs_diff = (merged.loc[valid, "logFC_g1"] - merged.loc[valid, "logFC_g2"]).abs().mean()
    mean_abs_mag  = ((merged.loc[valid, "logFC_g1"].abs() + merged.loc[valid, "logFC_g2"].abs()) / 2).mean()
    bias_ratio_pct = 100 * mean_abs_diff / mean_abs_mag if mean_abs_mag > 0 else np.nan

    p_g1 = -np.log10(merged[f"{pcol}_g1"].clip(lower=1e-300))
    p_g2 = -np.log10(merged[f"{pcol}_g2"].clip(lower=1e-300))
    p_corr = p_g1.corr(p_g2)

    return {
        "model": model,
        "n_genes_compared": len(merged),
        "n_g1_only": n_g1_only,
        "n_g2_only": n_g2_only,
        "n_both": n_both,
        "deg_overlap_pct": 100 * overlap_frac if not np.isnan(overlap_frac) else np.nan,
        "logfc_pearson_r": logfc_corr,
        "neglog10p_pearson_r": p_corr,
        "mean_abs_logfc_diff": mean_abs_diff,
        "bias_ratio_pct": bias_ratio_pct,
    }


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "de_results"
    rows = []
    for use_adj in [False, True]:
        for model in MODELS:
            res = compare(root, model, use_adjusted=use_adj)
            if res is not None:
                res["pval_basis"] = "adjusted" if use_adj else "raw"
                rows.append(res)
    out = pd.DataFrame(rows)
    cols = ["model", "pval_basis", "n_genes_compared", "n_g1_only", "n_g2_only", "n_both",
            "deg_overlap_pct", "logfc_pearson_r", "neglog10p_pearson_r",
            "mean_abs_logfc_diff", "bias_ratio_pct"]
    out = out[cols].sort_values(["pval_basis", "model"])
    out_path = os.path.join(root, "g1_g2_comparison_summary.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}")
