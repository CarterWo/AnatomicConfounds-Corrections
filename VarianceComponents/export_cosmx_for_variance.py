#!/usr/bin/env python3
"""Exports the CosMx cells used for the variance-component analysis (Supplementary Table 3).

Same export as DE_scripts/export_real_data_for_permutation.py, except that olfactory
bulb cells (napari_region == "Olfactory") are removed BEFORE the per-sample
subsampling, matching Pre_DE_processing.ipynb, which removes them before any
composition subsampling. Cntrl samples are dropped as in prepare_metadata().

usage: python3 export_cosmx_for_variance.py <h5ad> <out_dir> [cells_per_sample] [seed]
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

h5ad_path, out_dir = sys.argv[1], sys.argv[2]
cells_per_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

a = ad.read_h5ad(h5ad_path, backed="r")
obs = a.obs
keep = obs["FMT"].isin(["Stroke_FMT", "Healthy_FMT"]) & (obs["napari_region"].astype(str) != "Olfactory")
pool = obs[keep]
print(f"{a.n_obs} cells total; {keep.sum()} Stroke/Healthy FMT cells outside the olfactory bulb")
print(pool.groupby("sample_ID", observed=True).size())

rng = np.random.default_rng(seed)
chosen = []
for sample, grp in pool.groupby("sample_ID", observed=True):
    idx = grp.index.to_numpy()
    if len(idx) > cells_per_sample:
        idx = rng.choice(idx, size=cells_per_sample, replace=False)
    chosen.extend(idx.tolist())
sub = a[pd.Index(sorted(set(chosen)))].to_memory()


def dense(mat, index, columns):
    mat = mat.toarray() if hasattr(mat, "toarray") else mat
    return pd.DataFrame(np.asarray(mat), index=index, columns=columns)


genes = list(sub.var_names)
raw = dense(sub.layers["counts"], sub.obs_names, genes)
norm = dense(sub.X, sub.obs_names, genes)
n_features = (raw > 0).sum(axis=1)
log_depth = np.log10(n_features + 1)
log_depth = (log_depth - log_depth.mean()) / log_depth.std()

meta = pd.DataFrame({
    "sample_ID": sub.obs["sample_ID"].astype(str),
    "Treatment": sub.obs["FMT"].astype(str),
    "napari_region": sub.obs["napari_region"].astype(str),
    "quint_region": sub.obs["quint_region"].astype(str),
    "cell_type": sub.obs["ct_simple"].astype(str),
    "ct_simple": sub.obs["ct_simple"].astype(str),
    "nFeature_RNA": n_features,
    "log_depth": log_depth,
}, index=sub.obs_names)
assert (meta["napari_region"] != "Olfactory").all()

os.makedirs(out_dir, exist_ok=True)
norm.to_csv(f"{out_dir}/normalized_counts.csv")
meta.to_csv(f"{out_dir}/cell_metadata.csv")
print(f"Wrote {len(meta)} cells x {len(genes)} genes")
print(meta.groupby(["sample_ID", "Treatment"]).size())
print("cell types:", meta["cell_type"].nunique(), " napari regions:", sorted(meta["napari_region"].unique()))
