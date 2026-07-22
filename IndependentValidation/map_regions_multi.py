#!/usr/bin/env python3
"""
map_regions_multi.py

Extends map_regions.py (Zhuang-ABCA-1 only) to pool all 4 Zhuang-ABCA
datasets (Allen Brain Cell Atlas, AWS public dataset, CC BY 4.0). Each of
Zhuang-ABCA-1/2/3/4 is a single, distinct wild-type reference animal, so
pooling all 4 gives n=4 real distinct animals instead of n=1 -- directly
addressing the single-animal limitation of the original independent-
validation workstream (R2 Item 1), while reusing the identical CCF-based
region-mapping logic (no new annotation).

Same Allen-CCF-2020 division -> Napari-equivalent-category mapping as
map_regions.py. Adds a `dataset` column (Zhuang-ABCA-1..4) and prefixes
donor_label so each animal remains distinguishable after pooling (donor
IDs are dataset-specific, e.g. each dataset's single donor is literally
named "Zhuang-ABCA-N").

Usage:
    python3 map_regions_multi.py <raw_root> <out_csv>
      raw_root : directory containing Zhuang-ABCA-1/ (already present from
                 the original run) and Zhuang-ABCA-2/, -3/, -4/ (newly
                 downloaded), each with cell_metadata.csv + ccf_coordinates.csv
      out_csv  : combined output, one row per cell across all 4 animals
"""
import sys
import pandas as pd

raw_root = sys.argv[1] if len(sys.argv) > 1 else "raw"
out_csv  = sys.argv[2] if len(sys.argv) > 2 else "cell_metadata_with_regions_multi.csv"

DATASETS = ["Zhuang-ABCA-1", "Zhuang-ABCA-2", "Zhuang-ABCA-3", "Zhuang-ABCA-4"]

DIVISION_TO_NAPARI = {
    "Isocortex": "Cortex",
    "Hippocampal formation": "Hippocampus",
    "Striatum": "Striatum",
    "Cerebellum": "Cerebellum",
    "Olfactory areas": "Olfactory",
}

## Same pandas int-vs-str gotcha as map_regions.py -- cell_label is a
## 39-digit all-numeric string that pandas will silently parse as Python
## int without an explicit dtype, breaking every downstream h5ad join.
CELL_LABEL_DTYPE = {"cell_label": str}

## Shared across all 4 datasets -- same Allen-CCF-2020 atlas, downloaded
## once for Zhuang-ABCA-1 and reused here (no per-dataset CCF variant).
print("Loading shared parcellation_term_membership_name.csv ...")
terms = pd.read_csv(f"{raw_root}/parcellation_term_membership_name.csv")
terms = terms[["parcellation_index", "division", "structure"]].drop_duplicates("parcellation_index")

frames = []
for ds in DATASETS:
    ds_dir = f"{raw_root}/{ds}" if ds != "Zhuang-ABCA-1" else raw_root
    print(f"\n=== {ds} ({ds_dir}) ===")

    print("  Loading ccf_coordinates.csv ...")
    ccf = pd.read_csv(f"{ds_dir}/ccf_coordinates.csv",
                       usecols=["cell_label", "parcellation_index"],
                       dtype=CELL_LABEL_DTYPE)

    print("  Loading cell_metadata.csv ...")
    meta = pd.read_csv(
        f"{ds_dir}/cell_metadata.csv",
        usecols=["cell_label", "brain_section_label", "donor_label", "donor_genotype",
                  "donor_sex", "cluster_alias", "x", "y", "z",
                  "subclass_confidence_score", "cluster_confidence_score", "high_quality_transfer"],
        dtype=CELL_LABEL_DTYPE,
    )

    df = meta.merge(ccf, on="cell_label", how="left").merge(terms, on="parcellation_index", how="left")
    df["dataset"] = ds
    ## cell_label is only unique WITHIN a dataset's own h5ad; prefix so the
    ## pooled table has a globally unique key for the later Seurat-input join.
    df["cell_label"] = ds + "__" + df["cell_label"]

    n_total = len(df)
    print(f"  {n_total} cells loaded.")
    frames.append(df)

print("\nConcatenating all 4 datasets ...")
combined = pd.concat(frames, ignore_index=True)

combined["napari_region"] = combined["division"].map(DIVISION_TO_NAPARI).fillna("Unassigned")
combined["quint_region"] = combined["structure"].fillna("Unassigned")

n_total = len(combined)
n_mapped = (combined["napari_region"] != "Unassigned").sum()
print(f"\nTotal cells across 4 animals: {n_total}")
print(f"Cells mapped to one of the 5 Napari-equivalent regions: {n_mapped} ({100*n_mapped/n_total:.1f}%)")
print("\nCells per animal (donor_label):")
print(combined["donor_label"].value_counts())
print("\nnapari_region distribution:")
print(combined["napari_region"].value_counts())
print("\nCortex fraction per animal:")
print(combined.groupby("donor_label")["napari_region"].apply(lambda s: (s == "Cortex").mean()))

print(f"\nWriting {out_csv} ...")
combined.to_csv(out_csv, index=False)
print("Done.")
