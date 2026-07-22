"""
ezy_seq.lmm
===========
Utilities for loading, comparing, and summarizing differential expression results
produced by DREAM, Seurat, or pseudobulk (DESeq2) models.

Expected file naming convention
--------------------------------
Model output files should follow this pattern:

  {UP|DOWN}_{celltype}_{model}_{annotation}.csv

  Examples:
    UP_Astrocytes_dream_napari.csv
    DOWN_Microglia_seurat_quint.csv

Pseudobulk files follow a slightly different pattern:

  DESEQ2{UP|DOWN}_{celltype}_PB.csv

  Examples:
    DESEQ2UP_WHOLE_PB.csv
    DESEQ2DOWN_Astrocytes_PB.csv

Column schemas expected per model
----------------------------------
- DREAM / Seurat : Gene, logFC, P.Value, adj.P.Val
- Pseudobulk     : Gene, log2FoldChange (→ logFC), pvalue, padj (→ adj.P.Val)

Use standardize_columns() to normalize pseudobulk column names before merging.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, fisher_exact
from typing import Dict, Tuple, Optional, List
import warnings


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_de_file(filepath: Path) -> pd.DataFrame:
    """Load a single DE results CSV file."""
    return pd.read_csv(filepath)


def load_and_merge_up_down(
    up_path: Path,
    down_path: Path,
    merge_on: str = "Gene",
) -> pd.DataFrame:
    """
    Load and inner-merge an UP/DOWN file pair on a common gene column.

    Parameters
    ----------
    up_path, down_path : Path
        Paths to the UP and DOWN DE result CSVs.
    merge_on : str
        Column to merge on (default: "Gene").

    Returns
    -------
    pd.DataFrame with suffixes _up and _down.
    """
    df_up   = load_de_file(up_path)
    df_down = load_de_file(down_path)
    return pd.merge(df_up, df_down, on=merge_on, suffixes=("_up", "_down"), how="inner")


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_correlations(
    merged_df: pd.DataFrame,
    col1: str = "logFC_up",
    col2: str = "logFC_down",
) -> Dict[str, float]:
    """
    Calculate Pearson and Spearman correlations between two columns.

    Returns
    -------
    dict with keys: pearson_r, pearson_p, spearman_r, spearman_p, n_genes.
    """
    valid_mask = ~(merged_df[col1].isna() | merged_df[col2].isna())
    x = merged_df.loc[valid_mask, col1].values
    y = merged_df.loc[valid_mask, col2].values

    pearson_r,  pearson_p  = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    return {
        "pearson_r":  pearson_r,
        "pearson_p":  pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n_genes":    len(x),
    }


def calculate_directional_concordance(
    merged_df: pd.DataFrame,
    col1: str = "logFC_up",
    col2: str = "logFC_down",
) -> Dict[str, float]:
    """
    Calculate the fraction of genes where logFC has the same sign in both datasets.

    Returns
    -------
    dict with keys: concordance_rate, both_positive, both_negative, discordant, n_genes.
    """
    valid_mask = ~(merged_df[col1].isna() | merged_df[col2].isna())
    x = merged_df.loc[valid_mask, col1].values
    y = merged_df.loc[valid_mask, col2].values

    same_direction   = np.sign(x) == np.sign(y)
    concordance_rate = np.sum(same_direction) / len(same_direction) * 100

    return {
        "concordance_rate": concordance_rate,
        "both_positive":    int(np.sum((x > 0) & (y > 0))),
        "both_negative":    int(np.sum((x < 0) & (y < 0))),
        "discordant":       int(np.sum(~same_direction)),
        "n_genes":          len(x),
    }


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def get_significant_genes(
    df: pd.DataFrame,
    pval_col: str = "adj.P.Val",
    cutoff: float = 0.05,
) -> set:
    """Return the set of significant gene names from a DE DataFrame."""
    return set(df.loc[df[pval_col] < cutoff, "Gene"].values)


def fisher_overlap_test(
    sig_up: set,
    sig_down: set,
    background_size: int = 20000,
) -> Dict[str, float]:
    """
    Fisher's exact test for enrichment of overlap between two significant gene sets.

    Parameters
    ----------
    sig_up, sig_down : set
        Sets of significant genes from each condition.
    background_size : int
        Estimated total number of tested genes (used as the 'neither' cell).

    Returns
    -------
    dict with both_significant, only_up, only_down, neither, odds_ratio, fisher_p.
    """
    both_sig  = len(sig_up & sig_down)
    only_up   = len(sig_up  - sig_down)
    only_down = len(sig_down - sig_up)
    neither   = background_size - (both_sig + only_up + only_down)

    contingency = np.array([[both_sig, only_up], [only_down, neither]])
    odds_ratio, p_value = fisher_exact(contingency, alternative="greater")

    return {
        "both_significant": both_sig,
        "only_up":          only_up,
        "only_down":        only_down,
        "neither":          neither,
        "odds_ratio":       odds_ratio,
        "fisher_p":         p_value,
    }


def jaccard_index(set1: set, set2: set) -> float:
    """Jaccard index: |intersection| / |union|."""
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


def analyze_overlap(
    merged_df: pd.DataFrame,
    pval_cutoff: float = 0.05,
    background_size: int = 20000,
) -> Dict:
    """
    Full overlap analysis for a merged UP/DOWN dataset.

    Returns
    -------
    dict with n_sig_up, n_sig_down, jaccard_index, sig_genes_up, sig_genes_down,
    and all fields from fisher_overlap_test.
    """
    sig_up   = set(merged_df.loc[merged_df["adj.P.Val_up"]   < pval_cutoff, "Gene"])
    sig_down = set(merged_df.loc[merged_df["adj.P.Val_down"] < pval_cutoff, "Gene"])

    return {
        "n_sig_up":       len(sig_up),
        "n_sig_down":     len(sig_down),
        "jaccard_index":  jaccard_index(sig_up, sig_down),
        "sig_genes_up":   sig_up,
        "sig_genes_down": sig_down,
        **fisher_overlap_test(sig_up, sig_down, background_size),
    }


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_full_comparison(
    merged_df: pd.DataFrame,
    name: str = "",
) -> Dict:
    """
    Compute correlation, directional concordance, and overlap for one comparison.

    Returns
    -------
    dict suitable for building a summary table.
    """
    results = {"name": name, "n_genes": len(merged_df)}

    corr = calculate_correlations(merged_df)
    results.update({f"corr_{k}": v for k, v in corr.items()})

    conc = calculate_directional_concordance(merged_df)
    results.update({f"conc_{k}": v for k, v in conc.items()})

    overlap = analyze_overlap(merged_df)
    overlap_summary = {k: v for k, v in overlap.items() if not k.startswith("sig_genes")}
    results.update({f"overlap_{k}": v for k, v in overlap_summary.items()})

    return results


def create_summary_table(all_merged_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a summary DataFrame with one row per comparison.

    Parameters
    ----------
    all_merged_data : dict
        {key: merged_DataFrame} from load_all_seurat_dream_data() or similar.

    Returns
    -------
    pd.DataFrame
    """
    rows = [compute_full_comparison(df, name) for name, df in all_merged_data.items()]
    summary_df = pd.DataFrame(rows)

    col_order = [
        "name", "n_genes",
        "corr_pearson_r", "corr_spearman_r",
        "conc_concordance_rate",
        "overlap_n_sig_up", "overlap_n_sig_down",
        "overlap_both_significant", "overlap_jaccard_index",
        "overlap_odds_ratio", "overlap_fisher_p",
    ]
    existing = [c for c in col_order if c in summary_df.columns]
    other    = [c for c in summary_df.columns if c not in col_order]
    return summary_df[existing + other]


# =============================================================================
# BLAND-ALTMAN
# =============================================================================

def bland_altman_stats(
    merged_df: pd.DataFrame,
    col1: str = "logFC_up",
    col2: str = "logFC_down",
) -> Dict[str, float]:
    """
    Bland-Altman (method-comparison) statistics.

    Returns
    -------
    dict with mean_diff, std_diff, upper_loa, lower_loa, diff (array), avg (array).
    """
    valid_mask = ~(merged_df[col1].isna() | merged_df[col2].isna())
    x = merged_df.loc[valid_mask, col1].values
    y = merged_df.loc[valid_mask, col2].values

    diff = x - y
    avg  = (x + y) / 2
    mean_diff = np.mean(diff)
    std_diff  = np.std(diff)

    return {
        "mean_diff": mean_diff,
        "std_diff":  std_diff,
        "upper_loa": mean_diff + 1.96 * std_diff,
        "lower_loa": mean_diff - 1.96 * std_diff,
        "diff":      diff,
        "avg":       avg,
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_comparison_name(name: str) -> Dict[str, str]:
    """
    Parse a comparison key (e.g., 'Astrocytes_region_aware') into components.

    Returns
    -------
    dict with 'cell_type' and 'model_type'.
    """
    if "cortex_hippocampus" in name:
        ct_end     = name.find("_cortex_hippocampus")
        cell_type  = name[:ct_end]
        model_type = "cortex_hippocampus_quint" if "_quint" in name else "cortex_hippocampus_region_aware"
    elif "_region_aware_quint" in name:
        ct_end     = name.find("_region_aware_quint")
        cell_type  = name[:ct_end]
        model_type = "quint"
    elif "_region_aware" in name:
        ct_end     = name.find("_region_aware")
        cell_type  = name[:ct_end]
        model_type = "region_aware"
    elif "_region_blind" in name:
        ct_end     = name.find("_region_blind")
        cell_type  = name[:ct_end]
        model_type = "region_blind"
    else:
        cell_type  = name
        model_type = "unknown"

    return {"cell_type": cell_type, "model_type": model_type}


def get_annotation_display_name(annotation_type: str) -> str:
    """Return a human-readable label for an annotation type string."""
    display_names = {
        "quint":        "QUINT Atlas Annotated",
        "region_aware": "Region Aware (Non-QUINT)",
        "region_blind": "Region Blind",
        "napari":       "Napari Annotation",
        "blind":        "Region Blind",
    }
    return display_names.get(annotation_type, annotation_type)


# =============================================================================
# SEURAT / DREAM / PSEUDOBULK FILE LOADING
# =============================================================================

def standardize_columns(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """
    Normalize column names to a common schema across model types.

    Target schema:
      Gene, logFC, P.Value, adj.P.Val

    Parameters
    ----------
    df : pd.DataFrame
    model_type : str
        One of 'dream', 'seurat', or 'pseudobulk'.

    Returns
    -------
    pd.DataFrame with standardized column names.
    """
    df = df.copy()
    mt = model_type.lower()
    if mt == "pseudobulk":
        df = df.rename(columns={
            "log2FoldChange": "logFC",
            "pvalue":         "P.Value",
            "padj":           "adj.P.Val",
        })
    elif mt in ("seurat", "wilcoxon", "wilcox"):
        # Seurat FindMarkers output (Wilcoxon / LR): p_val, avg_log2FC, p_val_adj.
        # avg_log2FC is already log2, matching Dream's logFC and DESeq2's
        # log2FoldChange, so this is a pure rename (no base conversion).
        df = df.rename(columns={
            "avg_log2FC": "logFC",
            "p_val":      "P.Value",
            "p_val_adj":  "adj.P.Val",
        })
    elif mt != "dream":
        warnings.warn(f"Unknown model type: {model_type}. No column renaming applied.")
    return df


def parse_de_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse a DE output filename to extract direction, cell type, model, and annotation.

    Supported patterns
    ------------------
    DREAM / Seurat::

        {UP|DOWN}_{celltype}_{model}_{annotation}.csv

    Pseudobulk::

        DESEQ2{UP|DOWN}_{celltype}_PB.csv

    Parameters
    ----------
    filename : str
        Filename, with or without a directory path.

    Returns
    -------
    dict with keys 'direction', 'cell_type', 'model', 'annotation', or None if
    the filename does not match a known pattern.
    """
    name = Path(filename).stem

    # Pseudobulk: DESEQ2{UP|DOWN}_{celltype}_PB
    if name.startswith("DESEQ2"):
        if "DESEQ2UP_" in name:
            direction = "UP"
            rest = name.split("DESEQ2UP_")[1]
        elif "DESEQ2DOWN_" in name:
            direction = "DOWN"
            rest = name.split("DESEQ2DOWN_")[1]
        else:
            return None

        cell_type = rest[:-3] if rest.endswith("_PB") else rest
        return {"direction": direction, "cell_type": cell_type,
                "model": "pseudobulk", "annotation": None}

    # DREAM / Seurat: {UP|DOWN}_{celltype}_{model}_{annotation}
    if name.startswith("UP_"):
        direction = "UP"
        rest = name[3:]
    elif name.startswith("DOWN_"):
        direction = "DOWN"
        rest = name[5:]
    else:
        return None

    annotation = None
    for ann in ["fixed_napari", "fixed_quint", "napari", "quint", "blind"]:
        if rest.endswith(f"_{ann}"):
            annotation = ann
            rest = rest[: -len(f"_{ann}")]
            break

    # Map the filename model token to a canonical model name. The two cell-level
    # Seurat FindMarkers tests share the Seurat column schema (see
    # standardize_columns) but keep distinct identities so the figure scripts can
    # filter on them: the Wilcoxon test (file token "wilcox") -> "wilcoxon", and
    # the logistic-regression test (token "LR") -> "seurat". "seaurat" is a legacy
    # misspelling of "seurat".
    _MODEL_ALIAS = {"seaurat": "seurat", "wilcox": "wilcoxon", "LR": "seurat"}
    model = None
    for mod in ["dream", "seurat", "seaurat", "wilcoxon", "wilcox", "LR"]:
        if rest.endswith(f"_{mod}"):
            model = _MODEL_ALIAS.get(mod, mod)
            rest  = rest[: -len(f"_{mod}")]
            break

    if model is None:
        return None
    # The region-blind Wilcoxon baseline legitimately has no annotation; every
    # other model must have parsed one.
    if annotation is None and model != "wilcoxon":
        return None

    return {"direction": direction, "cell_type": rest, "model": model, "annotation": annotation}


def load_seurat_dream_pairs(
    data_dir: Path,
    pseudobulk_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Tuple[Path, Path]]]:
    """
    Discover and pair UP/DOWN result files in a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing DREAM and Seurat CSV files.
    pseudobulk_dir : Path, optional
        Directory containing pseudobulk CSV files (may differ from data_dir).

    Returns
    -------
    dict
        {model_type: {comparison_key: (up_path, down_path)}}

    Comparison keys follow the pattern ``{celltype}_{annotation}``
    (or just ``{celltype}`` for pseudobulk).
    """
    pairs = {"dream": {}, "seurat": {}, "pseudobulk": {}}
    up_files   = {"dream": {}, "seurat": {}, "pseudobulk": {}}
    down_files = {"dream": {}, "seurat": {}, "pseudobulk": {}}

    def _process_dir(directory: Path, allowed_models=None):
        for filepath in directory.glob("*.csv"):
            parsed = parse_de_filename(filepath.name)
            if parsed is None:
                continue
            model     = parsed["model"]
            direction = parsed["direction"]
            cell_type = simplify_celltype_name(parsed["cell_type"])
            annotation = parsed["annotation"]

            if allowed_models and model not in allowed_models:
                continue

            key = f"{cell_type}_{annotation}" if annotation else cell_type
            bucket = up_files if direction == "UP" else down_files
            bucket[model][key] = filepath

    _process_dir(data_dir, allowed_models=["dream", "seurat"])
    if pseudobulk_dir and pseudobulk_dir.exists():
        _process_dir(pseudobulk_dir, allowed_models=["pseudobulk"])

    for model in ["dream", "seurat", "pseudobulk"]:
        for key in up_files[model]:
            if key in down_files[model]:
                pairs[model][key] = (up_files[model][key], down_files[model][key])

    print("Found file pairs:")
    for model in ["dream", "seurat", "pseudobulk"]:
        print(f"  {model.upper()}: {len(pairs[model])} pairs")
        for key in sorted(pairs[model]):
            print(f"    - {key}")

    return pairs


def load_all_seurat_dream_data(
    data_dir: Path,
    pseudobulk_dir: Optional[Path] = None,
    merge_on: str = "Gene",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load and merge all UP/DOWN pairs for DREAM, Seurat, and pseudobulk models.

    Parameters
    ----------
    data_dir : Path
        Directory with DREAM and Seurat CSV files.
    pseudobulk_dir : Path, optional
        Directory with pseudobulk CSV files.
    merge_on : str
        Column to inner-merge on (default: "Gene").

    Returns
    -------
    dict
        {model_type: {comparison_key: merged_DataFrame}}
    """
    pairs    = load_seurat_dream_pairs(data_dir, pseudobulk_dir)
    all_data = {"dream": {}, "seurat": {}, "pseudobulk": {}}

    print("\n" + "=" * 60)
    print("Loading and merging data...")
    print("=" * 60)

    for model in ["dream", "seurat", "pseudobulk"]:
        print(f"\n{model.upper()}:")
        for key, (up_path, down_path) in pairs[model].items():
            try:
                df_up   = standardize_columns(load_de_file(up_path),   model)
                df_down = standardize_columns(load_de_file(down_path), model)
                merged  = pd.merge(df_up, df_down, on=merge_on,
                                   suffixes=("_up", "_down"), how="inner")
                all_data[model][key] = merged
                print(f"  ✓ {key}: {len(merged)} genes")
            except Exception as e:
                print(f"  ✗ {key}: {e}")

    return all_data


def get_model_display_name(model_type: str) -> str:
    """Return a human-readable label for a model type string."""
    display_names = {
        "dream":      "DREAM",
        "seurat":     "Seurat",
        "pseudobulk": "Pseudobulk (DESeq2)",
    }
    return display_names.get(model_type.lower(), model_type)


# =============================================================================
# CELL TYPE NAME UTILITIES
# =============================================================================

def simplify_celltype_name(name: str) -> str:
    """
    Shorten repetitive or verbose cell type name strings.

    Examples
    --------
    'Astrocytes.cortex.hippocampus_Astrocytes_cortex_hippocampus' → 'Astrocytes_cortex_hippo'
    'Astrocytes_Astrocytes' → 'Astrocytes'
    """
    if 'cortex.hippocampus' in name or 'cortex_hippocampus' in name:
        return 'Astrocytes_cortex_hippo'

    if '_' in name:
        parts = name.split('_')
        if len(parts) >= 2 and parts[0].lower() == parts[1].lower():
            return parts[0]

    return name


def simplify_comparison_key(key: str) -> str:
    """
    Simplify a comparison key of the form 'celltype_annotation'.

    Examples
    --------
    'Astrocytes_Astrocytes_napari' → 'Astrocytes_napari'
    """
    for ann in ['_napari', '_quint', '_blind']:
        if key.endswith(ann):
            return f"{simplify_celltype_name(key[:-len(ann)])}{ann}"
    return simplify_celltype_name(key)


# =============================================================================
# CROSS-ANNOTATION COMPARISON
# =============================================================================

def load_files_by_annotation(
    data_dir: Path,
    direction: str = "UP",
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Load DE files organized by model → cell_type → annotation.

    Parameters
    ----------
    data_dir : Path
        Directory containing CSV files.
    direction : str
        'UP' or 'DOWN' — which condition to load.

    Returns
    -------
    dict
        {model: {cell_type: {annotation: DataFrame}}}
    """
    result = {}

    for filepath in data_dir.glob("*.csv"):
        parsed = parse_de_filename(filepath.name)
        if parsed is None or parsed["direction"] != direction or parsed["annotation"] is None:
            continue

        model      = parsed["model"]
        cell_type  = simplify_celltype_name(parsed["cell_type"])
        annotation = parsed["annotation"]

        result.setdefault(model, {}).setdefault(cell_type, {})
        df = standardize_columns(load_de_file(filepath), model)
        result[model][cell_type][annotation] = df

    return result


def create_annotation_pairs(
    data_by_annotation: Dict[str, pd.DataFrame],
    merge_on: str = "Gene",
) -> Dict[str, pd.DataFrame]:
    """
    Create merged DataFrames for all pairwise annotation combinations.

    Parameters
    ----------
    data_by_annotation : dict
        {annotation: DataFrame} for one cell type and model.
    merge_on : str
        Column to inner-merge on.

    Returns
    -------
    dict
        {'ann1_vs_ann2': merged_DataFrame, ...}
    """
    annotations = list(data_by_annotation.keys())
    pairs = {}
    for i, ann1 in enumerate(annotations):
        for ann2 in annotations[i + 1:]:
            merged = pd.merge(
                data_by_annotation[ann1],
                data_by_annotation[ann2],
                on=merge_on,
                suffixes=(f"_{ann1}", f"_{ann2}"),
                how="inner",
            )
            pairs[f"{ann1}_vs_{ann2}"] = merged
    return pairs


def load_all_annotation_comparisons(
    data_dir: Path,
    direction: str = "UP",
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Load all annotation-to-annotation comparisons, organized by model and cell type.

    Parameters
    ----------
    data_dir : Path
        Directory containing CSV files.
    direction : str
        'UP' or 'DOWN'.

    Returns
    -------
    dict
        {model: {cell_type: {pair_name: merged_DataFrame}}}

    Example structure::

        {
            'dream': {
                'Astrocytes': {
                    'napari_vs_quint': DataFrame,
                    'napari_vs_blind': DataFrame,
                    'quint_vs_blind':  DataFrame,
                },
            },
        }
    """
    data_by_annotation = load_files_by_annotation(data_dir, direction)
    result = {}

    print(f"\nLoading annotation comparisons ({direction} condition):")
    print("=" * 60)

    for model, model_data in data_by_annotation.items():
        result[model] = {}
        print(f"\n{get_model_display_name(model)}:")

        for cell_type, ann_data in model_data.items():
            if len(ann_data) < 2:
                print(f"  {cell_type}: only {len(ann_data)} annotation(s) — skipping")
                continue

            pairs = create_annotation_pairs(ann_data)
            result[model][cell_type] = pairs

            print(f"  {cell_type}:")
            for pair_name, merged_df in pairs.items():
                print(f"    - {pair_name}: {len(merged_df)} genes")

    return result


def calculate_annotation_correlation(
    merged_df: pd.DataFrame,
    ann1: str,
    ann2: str,
) -> Dict[str, float]:
    """
    Calculate Pearson and Spearman correlations between two annotation logFCs.

    Expects columns ``logFC_{ann1}`` and ``logFC_{ann2}`` in merged_df.
    """
    col1, col2 = f"logFC_{ann1}", f"logFC_{ann2}"
    if col1 not in merged_df.columns or col2 not in merged_df.columns:
        raise ValueError(f"Columns {col1} and {col2} not found in DataFrame")
    return calculate_correlations(merged_df, col1, col2)


def annotation_bland_altman_stats(
    merged_df: pd.DataFrame,
    ann1: str,
    ann2: str,
) -> Dict[str, float]:
    """
    Bland-Altman statistics comparing two annotation logFCs.

    Expects columns ``logFC_{ann1}`` and ``logFC_{ann2}`` in merged_df.
    """
    col1, col2 = f"logFC_{ann1}", f"logFC_{ann2}"
    if col1 not in merged_df.columns or col2 not in merged_df.columns:
        raise ValueError(f"Columns {col1} and {col2} not found in DataFrame")
    return bland_altman_stats(merged_df, col1, col2)


def create_annotation_summary_table(
    annotation_comparisons: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
) -> pd.DataFrame:
    """
    Create a summary table for all annotation-to-annotation comparisons.

    Parameters
    ----------
    annotation_comparisons : dict
        Output of load_all_annotation_comparisons().

    Returns
    -------
    pd.DataFrame with one row per model × cell_type × annotation pair.
    """
    rows = []
    for model, model_data in annotation_comparisons.items():
        for cell_type, pairs in model_data.items():
            for pair_name, merged_df in pairs.items():
                ann1, ann2 = pair_name.split("_vs_")
                try:
                    corr = calculate_annotation_correlation(merged_df, ann1, ann2)
                    ba   = annotation_bland_altman_stats(merged_df, ann1, ann2)

                    col1, col2 = f"logFC_{ann1}", f"logFC_{ann2}"
                    valid = ~(merged_df[col1].isna() | merged_df[col2].isna())
                    x, y = merged_df.loc[valid, col1].values, merged_df.loc[valid, col2].values
                    concordance = np.sum(np.sign(x) == np.sign(y)) / len(x) * 100

                    rows.append({
                        'Model':         get_model_display_name(model),
                        'Cell Type':     cell_type,
                        'Comparison':    pair_name,
                        'N Genes':       len(merged_df),
                        'Pearson R':     corr['pearson_r'],
                        'Spearman R':    corr['spearman_r'],
                        'Concordance %': concordance,
                        'Mean Bias':     ba['mean_diff'],
                        'Lower LoA':     ba['lower_loa'],
                        'Upper LoA':     ba['upper_loa'],
                    })
                except Exception as e:
                    print(f"Error processing {model}/{cell_type}/{pair_name}: {e}")

    return pd.DataFrame(rows)


# =============================================================================
# SEED AGGREGATION  (composition-engineering iterations)
# =============================================================================

def aggregate_seed_files(
    files: List[Path],
    model_type: str,
    alpha: float = 0.05,
    gene_col: str = "Gene",
) -> pd.DataFrame:
    """
    Collapse a set of per-seed DE result files into one per-gene summary.

    All ``files`` must be the *same* analysis (identical direction / cell type /
    model / annotation) differing only by composition-engineering iteration
    (``seed``). Because those seeds are re-subsamples of the SAME cells they are
    NOT independent replicates, so their p-values must not be pooled with
    Fisher / Stouffer (that would treat correlated draws as independent and
    massively inflate significance). Instead we summarise the DISTRIBUTION of
    each gene's estimate across seeds:

    - effect size        : mean / median / SD of logFC and a 2.5-97.5% empirical
                           band (the composition-induced spread of the effect);
    - selection frequency: fraction of seeds calling the gene significant
                           (``adj.P.Val < alpha``) -- how robust the DE call is
                           to composition;
    - direction stability: fraction of seeds agreeing on the dominant logFC sign;
    - representative p    : median P.Value / adj.P.Val (a location summary of the
                           distribution, NOT a combined test).

    The result keeps the standard schema (``logFC`` = across-seed mean, and
    ``P.Value`` / ``adj.P.Val`` = medians) so it drops straight into the existing
    per-file visualisations, and adds the stability columns ``n_seeds``,
    ``present_frac``, ``logFC_median``, ``logFC_sd``, ``logFC_lo``, ``logFC_hi``,
    ``sig_frac`` and ``dir_consistency``.

    Parameters
    ----------
    files : list of Path
        Per-seed CSVs for one analysis (e.g. every
        ``seed_*/.../UP_WHOLE_dream_napari.csv`` within one ``dev_*`` folder).
    model_type : str
        'pseudobulk', 'dream', or 'seurat' -- passed to standardize_columns().
    alpha : float
        Significance threshold for the selection-frequency metric.
    gene_col : str
        Gene identifier column (default 'Gene').

    Returns
    -------
    pd.DataFrame
        One row per gene, sorted by the aggregated adj.P.Val (then P.Value).
    """
    if not files:
        return pd.DataFrame()

    cols = [gene_col, "logFC", "P.Value", "adj.P.Val"]
    frames = []
    for fp in files:
        df = standardize_columns(load_de_file(fp), model_type)
        frames.append(df[[c for c in cols if c in df.columns]])
    n_files = len(frames)
    long = pd.concat(frames, ignore_index=True)
    if gene_col not in long.columns:
        raise KeyError(f"{gene_col!r} column missing from the seed files")

    has_p    = "P.Value"   in long.columns
    has_padj = "adj.P.Val" in long.columns

    rows = []
    for gene, g in long.groupby(gene_col):
        lfc     = g["logFC"].astype(float)
        n       = int(len(g))
        n_valid = int(lfc.notna().sum())
        padj = g["adj.P.Val"].astype(float) if has_padj else None
        pval = g["P.Value"].astype(float)   if has_p    else None
        rows.append({
            gene_col:          gene,
            "logFC":           float(np.nanmean(lfc))               if n_valid else np.nan,
            "P.Value":         float(np.nanmedian(pval))            if (has_p and n_valid)    else np.nan,
            "adj.P.Val":       float(np.nanmedian(padj))            if (has_padj and n_valid) else np.nan,
            "n_seeds":         n,
            "present_frac":    n / n_files,
            "logFC_median":    float(np.nanmedian(lfc))             if n_valid else np.nan,
            "logFC_sd":        float(np.nanstd(lfc, ddof=1))        if n_valid > 1 else 0.0,
            "logFC_lo":        float(np.nanpercentile(lfc, 2.5))    if n_valid else np.nan,
            "logFC_hi":        float(np.nanpercentile(lfc, 97.5))   if n_valid else np.nan,
            "sig_frac":        float((padj < alpha).mean())         if has_padj else np.nan,
            "dir_consistency": float(max((lfc > 0).mean(), (lfc < 0).mean())) if n_valid else np.nan,
        })

    out = pd.DataFrame(rows)
    sort_key = "adj.P.Val" if has_padj else ("P.Value" if has_p else gene_col)
    return out.sort_values(sort_key, na_position="last").reset_index(drop=True)


def aggregate_seed_directory(
    results_root: Path,
    out_root: Optional[Path] = None,
    subdirs: Tuple[str, ...] = ("Pseudobulk_Validation",
                                "Local_Regional_Analysis",
                                "Global_CT_Analysis",
                                "SingleCell_Tests"),
    alpha: float = 0.05,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Aggregate every per-seed DE result under ``results_root`` across iterations.

    Expects the ``LMM_all.ipynb`` layout::

        <results_root>/dev_<dev>/seed_<seed>/<subdir>/<file>.csv

    Within one ``dev_<dev>`` folder the same analysis writes an identically named
    file in every ``seed_<seed>`` folder (the seed lives in the folder, not the
    filename), so files are grouped by name and collapsed with
    ``aggregate_seed_files``. DESeq2 groups span all tagged seeds (e.g. 100)
    while the Dream LMM groups span only the seeds it ran (e.g. 3). Aggregated
    results mirror the layout one level up::

        <out_root>/dev_<dev>/<subdir>/<file>.csv

    Parameters
    ----------
    results_root : Path
        The DE results root (config ``outputs.lmm_results_dir``).
    out_root : Path, optional
        Where to write aggregates (default ``<results_root>/aggregated``).
    subdirs : tuple of str
        Per-iteration analysis subfolders to scan.
    alpha : float
        Significance threshold forwarded to ``aggregate_seed_files``.

    Returns
    -------
    dict
        Maps each written ``dev_<dev>/<subdir>/<file>`` to its seed count.
    """
    from collections import defaultdict

    results_root = Path(results_root)
    out_root = Path(out_root) if out_root is not None else results_root / "aggregated"

    written: Dict[str, int] = {}
    dev_dirs = sorted(d for d in results_root.glob("dev_*") if d.is_dir())
    if not dev_dirs and verbose:
        print(f"No dev_* folders found under {results_root}")

    for dev_dir in dev_dirs:
        for sub in subdirs:
            groups: Dict[str, List[Path]] = defaultdict(list)
            for seed_dir in sorted(dev_dir.glob("seed_*")):
                sub_dir = seed_dir / sub
                if not sub_dir.is_dir():
                    continue
                for fp in sub_dir.glob("*.csv"):
                    groups[fp.name].append(fp)
            if not groups:
                continue
            out_dir = out_root / dev_dir.name / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname, group_files in sorted(groups.items()):
                parsed = parse_de_filename(fname)
                if parsed is None:
                    if verbose:
                        print(f"  Skipping unparseable file: {fname}")
                    continue
                agg = aggregate_seed_files(group_files, parsed["model"], alpha=alpha)
                agg.to_csv(out_dir / fname, index=False)
                rel = f"{dev_dir.name}/{sub}/{fname}"
                written[rel] = len(group_files)
                if verbose:
                    print(f"  {rel}: {len(group_files)} seeds -> {len(agg)} genes")
    return written
