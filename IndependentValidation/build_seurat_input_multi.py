#!/usr/bin/env python3
"""
build_seurat_input_multi.py

Multi-animal successor to build_seurat_input.py. Reads all 4 Zhuang-ABCA
h5ad pairs (one raw + one log2 per animal, since Allen ships expression
matrices per-dataset, not pooled) and exports G1/G2 Seurat-input CSVs in
the exact layout de_functions.R::make_cfg()/build_seurat_from_folder()
expects, so the manuscript's DE suite runs completely unmodified.

QC THRESHOLDS: applies the manuscript's OWN real-data QC thresholds
(ezy_seq.filter_and_normalize / Pre_DE_processing.ipynb call: min_gene_cnt
= 20 genes detected/cell, min_t_cnt = 100 total counts/cell, min_cell_cnt
= 100 cells/gene), NOT applied in the original single-animal
build_seurat_input.py. This keeps the independent-validation dataset on
identical QC footing to the manuscript's own real CosMx analysis rather
than trusting Allen's own (different) release-level QC alone.

Usage:
    python3 build_seurat_input_multi.py <tagged_csv> <raw_root> <out_root> [max_cells_per_scenario] [seed]
      tagged_csv : output of real_animal_composition_engineering.py
      raw_root   : dir containing Zhuang-ABCA-1/, -2/, -3/, -4/ subdirs,
                    each with expression_matrices/*-raw.h5ad, *-log2.h5ad,
                    and gene.csv (Zhuang-ABCA-1's own raw/gene files live
                    directly in raw_root for backward compatibility with
                    the original single-animal run)
      out_root   : writes <out_root>/g1/ and <out_root>/g2/
"""
import sys
import os
import numpy as np
import pandas as pd
import anndata as ad

tagged_csv = sys.argv[1]
raw_root   = sys.argv[2]
out_root   = sys.argv[3]
max_cells_per_scenario = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
subsample_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
## Optional 6th arg: restrict to one scenario (g1 or g2) -- lets a timed-out
## run resume just the unfinished scenario instead of redoing an already-
## successful one from scratch.
scenario_filter = sys.argv[6] if len(sys.argv) > 6 else None

DATASETS = ["Zhuang-ABCA-1", "Zhuang-ABCA-2", "Zhuang-ABCA-3", "Zhuang-ABCA-4"]

## Manuscript's own real-data QC thresholds (ezyfunctions.filter_and_normalize
## defaults / Pre_DE_processing.ipynb's explicit call: min_gene_cnt=20,
## min_t_cnt=100; min_cell_cnt left at the function's own default of 100).
MIN_GENE_CNT = 20   # min genes detected per cell
MIN_T_CNT = 100     # min total counts per cell
MIN_CELL_CNT = 100  # min cells a gene must appear in

print(f"Loading {tagged_csv} ...")
meta_all = pd.read_csv(tagged_csv, low_memory=False, dtype={"cell_label": str})
meta_all = meta_all.set_index("cell_label")

def ds_dir(ds):
    return raw_root if ds == "Zhuang-ABCA-1" else f"{raw_root}/{ds}"

def h5ad_paths(ds):
    d = f"{ds_dir(ds)}/expression_matrices"
    raw_f = [f for f in os.listdir(d) if f.endswith("-raw.h5ad")][0]
    log2_f = [f for f in os.listdir(d) if f.endswith("-log2.h5ad")][0]
    return f"{d}/{raw_f}", f"{d}/{log2_f}"

print("\nLoading per-dataset AnnData (backed) and gene-symbol maps ...")
adatas_raw, adatas_log2, gene_maps = {}, {}, {}
for ds in DATASETS:
    raw_p, log2_p = h5ad_paths(ds)
    print(f"  {ds}: {raw_p}")
    adatas_raw[ds] = ad.read_h5ad(raw_p, backed="r")
    adatas_log2[ds] = ad.read_h5ad(log2_p, backed="r")
    genes = pd.read_csv(f"{ds_dir(ds)}/gene.csv").set_index("gene_identifier")["gene_symbol"]
    gene_maps[ds] = genes
    for a in (adatas_raw[ds], adatas_log2[ds]):
        a.var_names = [genes.get(g, g) for g in a.var_names]

## Sanity check: all 4 Zhuang-ABCA datasets share the same 1,122-gene
## MERFISH panel (same Allen manifest version) -- confirm rather than assume.
gene_sets = [set(a.var_names) for a in adatas_raw.values()]
common_genes = set.intersection(*gene_sets)
print(f"\nGene panel overlap across all 4 animals: {len(common_genes)} genes common "
      f"(per-dataset sizes: {[len(g) for g in gene_sets]})")
if any(len(g) != len(common_genes) for g in gene_sets):
    print("NOTE: panels are not identical across all 4 animals -- restricting to the common set.")

for scenario, bool_col in [("g1", "G1"), ("g2", "G2")]:
    if scenario_filter and scenario != scenario_filter:
        print(f"\nSkipping {scenario} (scenario_filter={scenario_filter})")
        continue
    print(f"\n=== Building {scenario} (cells where {bool_col} == True) ===")
    scen_meta = meta_all[meta_all[bool_col] == True]
    print(f"  {len(scen_meta)} cells across all animals (pre-QC)")

    raw_parts, log2_parts = [], []
    for ds in DATASETS:
        ds_mask = scen_meta["dataset"] == ds
        ds_meta = scen_meta[ds_mask]
        if len(ds_meta) == 0:
            continue
        ## strip the "Zhuang-ABCA-N__" prefix added in map_regions_multi.py
        ## to recover this dataset's own h5ad obs_names.
        raw_ids = pd.Index([c.split("__", 1)[1] for c in ds_meta.index])
        common = pd.Index(sorted(set(adatas_raw[ds].obs_names) & set(raw_ids)))
        print(f"  {ds}: {len(ds_meta)} tagged, {len(common)} present in expression matrix")
        if len(common) == 0:
            continue

        raw_sub = adatas_raw[ds][common].to_memory()
        log2_sub = adatas_log2[ds][common].to_memory()

        def to_dense_df(a, cols):
            X = a.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            df = pd.DataFrame(np.asarray(X), index=a.obs_names, columns=a.var_names)
            return df[cols]

        raw_df = to_dense_df(raw_sub, sorted(common_genes))
        log2_df = to_dense_df(log2_sub, sorted(common_genes))

        ## Apply the manuscript's own QC thresholds (min genes/cell, min
        ## counts/cell) at the per-animal level, matching how QC is applied
        ## per-sample in the real pipeline before pooling.
        n_genes_detected = (raw_df > 0).sum(axis=1)
        total_counts = raw_df.sum(axis=1)
        qc_pass = (n_genes_detected >= MIN_GENE_CNT) & (total_counts >= MIN_T_CNT)
        n_before = len(raw_df)
        raw_df = raw_df.loc[qc_pass]
        log2_df = log2_df.loc[qc_pass]
        print(f"    QC (min_gene_cnt={MIN_GENE_CNT}, min_t_cnt={MIN_T_CNT}): "
              f"{n_before} -> {len(raw_df)} cells")

        prefixed_index = [f"{ds}__{c}" for c in raw_df.index]
        raw_df.index = prefixed_index
        log2_df.index = prefixed_index

        raw_parts.append(raw_df)
        log2_parts.append(log2_df)

    raw_df = pd.concat(raw_parts, axis=0)
    log2_df = pd.concat(log2_parts, axis=0)
    scen_meta_kept = scen_meta.reindex(raw_df.index)

    ## Gene-level QC: drop genes detected in fewer than MIN_CELL_CNT cells,
    ## applied AFTER pooling across all 4 animals (matches filter_genes
    ## being applied on the full multi-sample AnnData in the manuscript's
    ## own pipeline, not per-sample).
    n_cells_per_gene = (raw_df > 0).sum(axis=0)
    genes_keep = n_cells_per_gene[n_cells_per_gene >= MIN_CELL_CNT].index
    print(f"  Gene QC (min_cell_cnt={MIN_CELL_CNT}): {raw_df.shape[1]} -> {len(genes_keep)} genes")
    raw_df = raw_df[genes_keep]
    log2_df = log2_df[genes_keep]

    if max_cells_per_scenario and len(raw_df) > max_cells_per_scenario:
        rng = np.random.default_rng(subsample_seed)
        ## Balanced per-animal subsample (not pooled-then-random) so no
        ## single animal dominates the final N, matching the manuscript's
        ## own per-sample balanced subsampling principle.
        keep_ids = []
        per_animal_target = max_cells_per_scenario // scen_meta_kept["dataset"].nunique()
        for ds, ds_ids in scen_meta_kept.groupby("dataset").groups.items():
            ids_in_raw = [i for i in ds_ids if i in raw_df.index]
            n_take = min(len(ids_in_raw), per_animal_target)
            keep_ids.extend(rng.choice(ids_in_raw, size=n_take, replace=False).tolist())
        raw_df = raw_df.loc[keep_ids]
        log2_df = log2_df.loc[keep_ids]
        scen_meta_kept = scen_meta_kept.loc[keep_ids]
        print(f"  Balanced per-animal subsample -> {len(raw_df)} cells total "
              f"(target {per_animal_target}/animal, random_state={subsample_seed})")

    gene_names = list(raw_df.columns)
    features_df = pd.DataFrame({"gene_id": gene_names}, index=gene_names)

    n_features = (raw_df > 0).sum(axis=1)
    log_depth = np.log10(n_features + 1)
    log_depth = (log_depth - log_depth.mean()) / log_depth.std()

    cell_metadata_out = pd.DataFrame({
        "sample_ID":      scen_meta_kept["donor_label"].str.replace("-", "", regex=False),   # dashes stripped: de_functions.R run_pseudobulk_deseq2 gsubs "-" to "_" on aggregated colnames but not on colData$sample_ID, so any dash here breaks the match and causes duplicate-rowname failures
        "Treatment":      scen_meta_kept["pseudo_group"],
        "napari_region":  scen_meta_kept["napari_region"],
        "quint_region":   scen_meta_kept["quint_region"],
        "cell_type":      "AllCells",
        "ct_simple":      "AllCells",
        "nFeature_RNA":   n_features,
        "log_depth":      log_depth,
    }, index=raw_df.index)

    coords_df = scen_meta_kept.loc[raw_df.index, ["x", "y"]]

    out_dir = f"{out_root}/{scenario}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Writing CSVs to {out_dir} ...")
    raw_df.to_csv(f"{out_dir}/raw_counts.csv")
    log2_df.to_csv(f"{out_dir}/normalized_counts.csv")
    features_df.to_csv(f"{out_dir}/features_counts.csv")
    cell_metadata_out.to_csv(f"{out_dir}/cell_metadata.csv")
    coords_df.to_csv(f"{out_dir}/coords_xy.csv")
    print(f"  Done: {len(raw_df)} cells x {len(gene_names)} genes, "
          f"animals represented: {sorted(scen_meta_kept['donor_label'].unique())}")

print("\nAll scenarios exported.")
