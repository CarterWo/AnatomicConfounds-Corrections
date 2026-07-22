#!/usr/bin/env python3
"""
compose_via_ezyseq_multi.py

Replaces real_animal_composition_engineering.py's hand-rolled subsampling
with a DIRECT call to Carter's actual ezy_seq.tag_region_abundance_by_FMT
(the repo's own EzySeq_Library package -- install with
`pip install -e EzySeq_Library/mypythonlibrary` from the repo root), using
the SAME parameters the real manuscript pipeline uses in
Pre_DE_processing.ipynb:

    ezy.tag_region_abundance_by_FMT(adata, dev=<x>, random_state=<seed>,
                                     balance="sample")

(region_of_interest="Cortex", fill_other="baseline", match_scenarios=False --
all defaults, unchanged, exactly matching the production call.)

The only adaptation is column names, since Zhuang-ABCA has no FMT/sample_ID
columns: donor_label -> sample_col, pseudo_group (A/B) -> fmt_col (up_fmt="A",
down_fmt="B"), napari_region -> region_col (already matches).

Unlike real_animal_composition_engineering.py (which kept ~100% of each
animal's own cell pool and only varied the internal Cortex/other split),
balance="sample" subsamples every animal DOWN to a shared, feasibility-
capped N -- this is expected to change both the total retained cell count
and the achieved Cortex-fraction gap relative to the earlier script.

Usage:
    python3 compose_via_ezyseq_multi.py <mapped_csv> <out_csv> [dev] [seed]
"""
import sys
import numpy as np
import pandas as pd
from anndata import AnnData

# Requires the repo's ezy_seq package to be installed:
# `pip install -e EzySeq_Library/mypythonlibrary` from the repo root.
import ezy_seq as ezy

mapped_csv = sys.argv[1] if len(sys.argv) > 1 else "cell_metadata_with_regions_multi.csv"
out_csv    = sys.argv[2] if len(sys.argv) > 2 else "cell_metadata_tagged_multi_ezyseq.csv"
dev        = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
seed       = int(sys.argv[4]) if len(sys.argv) > 4 else 0

print(f"Loading {mapped_csv} ...")
df = pd.read_csv(mapped_csv, dtype={"cell_label": str})

animals = sorted(df["donor_label"].unique())
print(f"\n{len(animals)} real animals found: {animals}")
if len(animals) != 4:
    print(f"WARNING: expected 4 animals (Zhuang-ABCA-1..4), found {len(animals)}.")

# Random 2-vs-2 split into pseudo-groups A/B -- same as
# real_animal_composition_engineering.py: no systematic biological
# difference by construction (all 4 are the same wild-type genotype).
rng = np.random.default_rng(seed)
animals_shuffled = list(animals)
rng.shuffle(animals_shuffled)
half = len(animals_shuffled) // 2
animal_to_group = {a: ("A" if i < half else "B") for i, a in enumerate(animals_shuffled)}
df["pseudo_group"] = df["donor_label"].map(animal_to_group)
print(f"pseudo-group A: {[a for a in animals_shuffled[:half]]}; "
      f"B: {[a for a in animals_shuffled[half:]]} (random_state={seed})")

# Build a metadata-only AnnData: tag_region_abundance_by_FMT touches only
# .obs / .obs_names, never .X, so a placeholder expression matrix is fine
# and keeps this composition-tagging step cheap.
df = df.set_index("cell_label", drop=False)
adata = AnnData(
    X=np.zeros((len(df), 1), dtype=np.float32),
    obs=df,
)

print(f"\nCalling ezy.tag_region_abundance_by_FMT(dev={dev:g}, random_state={seed}, "
      f"balance='sample', up_fmt='A', down_fmt='B', region_col='napari_region', "
      f"fmt_col='pseudo_group', sample_col='donor_label') -- production defaults "
      f"otherwise (region_of_interest='Cortex', fill_other='baseline', "
      f"match_scenarios=False), identical to Pre_DE_processing.ipynb's real call.")

adata = ezy.tag_region_abundance_by_FMT(
    adata,
    dev=dev,
    random_state=seed,
    up_fmt="A",
    down_fmt="B",
    region_col="napari_region",
    fmt_col="pseudo_group",
    sample_col="donor_label",
    balance="sample",
)

seed_tag = seed if seed is not None else "None"
g1_col = f"G1_{seed_tag}_{dev:g}"
g2_col = f"G2_{seed_tag}_{dev:g}"

out_df = adata.obs.copy()
out_df["G1"] = out_df[g1_col]
out_df["G2"] = out_df[g2_col]

print(f"\nCortex fraction achieved, by pseudo_group x scenario:")
for scen_col in ("G1", "G2"):
    sub = out_df[out_df[scen_col]]
    print(f"  {scen_col}:", sub.groupby("pseudo_group")["napari_region"]
          .apply(lambda s: f"{(s == 'Cortex').mean():.4f} (n={len(s)})").to_dict())

print(f"\nWriting {out_csv} ...")
out_df.drop(columns=[g1_col, g2_col]).to_csv(out_csv, index=False)
print("Done.")
