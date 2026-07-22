#!/usr/bin/env python3
"""
build_seurat_input.py

Exports the composition-engineered Zhuang-ABCA-1 dataset (tagged by
synthetic_composition_engineering.py) into the exact CSV layout the
manuscript's own R pipeline expects (`de_functions.R::make_cfg()` /
`build_seurat_from_folder()`), so the existing DE suite
(run_analysis_suite: Dream blind/napari/quint, DESeq2 pseudobulk, Seurat
LR blind/napari/quint, Wilcoxon) can run COMPLETELY UNMODIFIED on this
independent dataset -- the same principle already used for the synthetic
splatter ground-truth simulation.

Design recap (see CLAUDE.md "Independent-dataset validation" section for
full detail): there is no real treatment/control variable in this single
wild-type reference animal, so the role the manuscript's real
Stroke_FMT/Healthy_FMT "Treatment" plays is filled here by `pseudo_group`
(A/B), an arbitrary random split of pseudo-samples with NO systematic
biological difference by construction. G1 and G2 are the two composition-
IMBALANCE scenarios of that SAME pseudo_group contrast (Cortex enriched in
A vs. B, and the reverse), exactly mirroring how the manuscript's own real
G1/G2 are two composition scenarios of the SAME Stroke-vs-Healthy
contrast. Two independent exports are written (one per scenario), each
fed through `build_seurat_from_folder()` separately, exactly matching how
the manuscript's own LMM_all.ipynb loads "up" and "down" as two separate
Seurat objects.

Requires: anndata (pip install --user anndata; no scanpy needed since we
only read X + var, not run any scanpy processing).

Usage:
    python3 build_seurat_input.py <tagged_csv> <raw_h5ad> <log2_h5ad> <gene_csv> <out_root>
      tagged_csv : output of synthetic_composition_engineering.py
      raw_h5ad   : Zhuang-ABCA-1-raw.h5ad (raw counts)
      log2_h5ad  : Zhuang-ABCA-1-log2.h5ad (log2-normalized)
      gene_csv   : gene.csv (gene_identifier -> gene_symbol mapping)
      out_root   : writes <out_root>/g1/ and <out_root>/g2/, each with
                   raw_counts.csv, normalized_counts.csv, features_counts.csv,
                   cell_metadata.csv, coords_xy.csv
"""
import sys
import numpy as np
import pandas as pd
import anndata as ad

tagged_csv = sys.argv[1]
raw_h5ad   = sys.argv[2]
log2_h5ad  = sys.argv[3]
gene_csv   = sys.argv[4]
out_root   = sys.argv[5]
## Full G1/G2 scenarios are ~1.3-1.4M cells each -- the manuscript's own R DE
## suite (Dream LMM, Seurat LR) has only been exercised at ~20K-cell scale
## (both in the real manuscript and in our splatter ground-truth
## simulation). Subsample down to a comparable scale by default so the
## pipeline runs in a tractable time and doesn't inflate apparent
## statistical power via an unvalidated, much larger N. Set to 0 to disable
## (export all cells) -- NOT recommended without first testing runtime/memory
## at the default scale.
max_cells_per_scenario = int(sys.argv[6]) if len(sys.argv) > 6 else 20000
subsample_seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0

print(f"Loading {tagged_csv} ...")
## cell_label MUST be forced to string dtype -- this was the actual root
## cause of the "0 cells also present in the expression matrix" bug found
## during smoke-testing: without this, pandas silently parses these
## 39-digit all-numeric IDs as Python `int` (even though the resulting
## column/index still shows generic "object" dtype), and `int` never
## compares equal to `str` even for identical digits, so every join
## against the h5ad's (correctly string-typed) obs_names silently failed.
## Two earlier, incorrect diagnoses (embedded quote characters; unreliable
## pandas Index.intersection on backed AnnData) are documented in
## CLAUDE.md for the record -- this dtype fix is the real one, confirmed
## by a full-scale set-based cross-check.
meta = pd.read_csv(tagged_csv, low_memory=False, dtype={"cell_label": str})
meta = meta.set_index("cell_label")

print(f"Loading gene symbol mapping from {gene_csv} ...")
genes = pd.read_csv(gene_csv).set_index("gene_identifier")["gene_symbol"]

print(f"Loading raw counts from {raw_h5ad} (backed, not fully into memory) ...")
adata_raw = ad.read_h5ad(raw_h5ad, backed="r")
print(f"Loading log2-normalized data from {log2_h5ad} (backed) ...")
adata_log2 = ad.read_h5ad(log2_h5ad, backed="r")

# BUG FOUND during smoke-testing, TAKE 2: an initial diagnosis (embedded
# quote characters in obs_names) was WRONG -- an artifact of printing
# repr(x) values inside a list, which Python re-quotes with double quotes
# when the (already repr'd) string itself contains apostrophes. A
# comprehensive full-scale check (plain Python `set()` intersection, not
# pandas Index.intersection()) confirmed raw.h5ad, log2.h5ad, and
# cell_metadata.csv all align PERFECTLY: all 2,846,908 metadata cell_labels
# are an exact subset of both h5ad files' 4,167,870 cells, and raw/log2
# are positionally identical across all rows. The REAL bug is that
# `AnnData.obs_names.intersection(...)` (pandas Index.intersection) gives
# an incorrect empty result when the AnnData object is loaded `backed="r"`
# with a very-large-integer-like string index -- a plain Python set
# intersection on the same data works correctly. Use `set()` here, not
# pandas Index methods, on backed AnnData objects.

# Rename var_names (gene_identifier / Ensembl IDs) to gene symbols, matching
# how the manuscript's own gene panel is referenced (e.g. "Phactr1", "Dlg4").
for a in (adata_raw, adata_log2):
    a.var_names = [genes.get(g, g) for g in a.var_names]

for scenario, bool_col in [("g1", "G1"), ("g2", "G2")]:
    print(f"\n=== Building {scenario} (cells where {bool_col} == True) ===")
    scen_meta = meta[meta[bool_col] == True]
    cell_ids = scen_meta.index.to_numpy()
    print(f"  {len(cell_ids)} cells")

    # Intersect with what's actually present in the h5ad. Use a plain Python
    # set (NOT pandas Index.intersection -- see comment above, unreliable on
    # a backed AnnData's index for this data), then convert to a list for
    # anndata's fancy-indexing subsetting.
    common = pd.Index(sorted(set(adata_raw.obs_names) & set(cell_ids)))
    print(f"  {len(common)} cells also present in the expression matrix")

    if max_cells_per_scenario and len(common) > max_cells_per_scenario:
        rng = np.random.default_rng(subsample_seed)
        common = pd.Index(rng.choice(common.to_numpy(), size=max_cells_per_scenario, replace=False))
        print(f"  Subsampled down to {max_cells_per_scenario} cells "
              f"(random_state={subsample_seed})")

    scen_meta = scen_meta.loc[common]

    print("  Reading raw counts subset into memory ...")
    raw_sub = adata_raw[common].to_memory()
    print("  Reading log2 subset into memory ...")
    log2_sub = adata_log2[common].to_memory()

    def to_dense_df(a):
        X = a.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return pd.DataFrame(np.asarray(X), index=a.obs_names, columns=a.var_names)

    raw_df = to_dense_df(raw_sub)
    log2_df = to_dense_df(log2_sub)

    gene_names = list(raw_df.columns)
    features_df = pd.DataFrame({"gene_id": gene_names}, index=gene_names)

    # log_depth: identical definition used throughout the manuscript's real
    # analyses and the splatter simulation -- scale(log10(nFeature + 1)).
    n_features = (raw_df > 0).sum(axis=1)
    log_depth = np.log10(n_features + 1)
    log_depth = (log_depth - log_depth.mean()) / log_depth.std()

    cell_metadata_out = pd.DataFrame({
        "sample_ID":      scen_meta["pseudo_sample"],
        "Treatment":      scen_meta["pseudo_group"],
        "napari_region":  scen_meta["napari_region"],
        "quint_region":   scen_meta["quint_region"],
        "cell_type":      "AllCells",   # Zhuang-ABCA-1 has subclass/cluster labels
                                         # (cluster_alias) but no direct equivalent to
                                         # the manuscript's ct_simple; single-level
                                         # placeholder keeps run_de_dream's
                                         # num_cts>1 branch inert (matches how the
                                         # manuscript's own code handles a
                                         # single-cell-type subset).
        "ct_simple":      "AllCells",
        "nFeature_RNA":   n_features,
        "log_depth":      log_depth,
    }, index=raw_df.index)

    coords_df = scen_meta.loc[raw_df.index, ["x", "y"]]

    out_dir = f"{out_root}/{scenario}"
    import os
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Writing CSVs to {out_dir} ...")
    raw_df.to_csv(f"{out_dir}/raw_counts.csv")
    log2_df.to_csv(f"{out_dir}/normalized_counts.csv")
    features_df.to_csv(f"{out_dir}/features_counts.csv")
    cell_metadata_out.to_csv(f"{out_dir}/cell_metadata.csv")
    coords_df.to_csv(f"{out_dir}/coords_xy.csv")
    print(f"  Done: {len(raw_df)} cells x {len(gene_names)} genes")

print("\nAll scenarios exported.")
