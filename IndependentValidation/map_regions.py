#!/usr/bin/env python3
"""
map_regions.py

Maps every cell in the Zhuang-ABCA-1 MERFISH dataset (Allen Brain Cell
Atlas) onto the manuscript's existing two region-labeling schemes
(napari_region: 5 broad categories; quint_region: finer granularity),
using ONLY the Allen CCFv3 annotations already shipped with the dataset --
no new Napari polygon annotation, no new QUINT atlas registration.

Reviewer 1's manuscript already registers its own CosMx data to this same
CCF (Wang et al. 2020) via a modified QUINT pipeline, so reusing the CCF's
own hierarchy here keeps the independent-validation dataset on equivalent
anatomical footing without inventing a new ontology.

Inputs (already downloaded to independent_validation/raw/ on Hellbender):
  - cell_metadata.csv                       (cell_label, donor info, etc.)
  - ccf_coordinates.csv                     (cell_label -> parcellation_index)
  - parcellation_term_membership_name.csv   (parcellation_index -> organ,
                                              category, division, structure,
                                              substructure)

Output:
  - cell_metadata_with_regions.csv: one row per cell, with two new columns:
      napari_region : mapped from `division`, renamed to match the
                      manuscript's 5 Napari categories (Cortex, Hippocampus,
                      Striatum, Cerebellum, Olfactory); anything else ->
                      "Unassigned" (mirrors the manuscript's own treatment
                      of cells outside its 5 annotated regions).
      quint_region  : the `structure` column used as-is (finer granularity,
                      analogous role to the manuscript's 36-category QUINT
                      taxonomy -- both are CCF-derived, just at different
                      levels of the same hierarchy).

Usage:
    python3 map_regions.py <raw_dir> <out_csv>
"""
import sys
import pandas as pd

raw_dir = sys.argv[1] if len(sys.argv) > 1 else "raw"
out_csv = sys.argv[2] if len(sys.argv) > 2 else "cell_metadata_with_regions.csv"

# Division (CCF) -> manuscript's Napari region name. Only the manuscript's
# 5 annotated regions get a real label; everything else is "Unassigned",
# mirroring how the manuscript itself only labels these 5 broad structures
# in its own Napari workflow (README: "segmentation was focused on broad
# neuroanatomical structures, specifically the iso-cortex (cortex),
# hippocampus, olfactory bulb, striatum, cerebellum, or remained unlabeled").
DIVISION_TO_NAPARI = {
    "Isocortex": "Cortex",
    "Hippocampal formation": "Hippocampus",
    "Striatum": "Striatum",
    "Cerebellum": "Cerebellum",
    "Olfactory areas": "Olfactory",
}

## CRITICAL: cell_label is a 39-digit all-numeric string. Without an
## explicit dtype, pandas silently parses it as a Python arbitrary-
## precision int (NOT a string), even though the resulting column shows
## generic "object" dtype -- `int` and `str` never compare equal even for
## identical digits, so every downstream cell-ID join against the h5ad's
## (correctly string-typed) obs_names silently returns zero matches. This
## is exactly why Allen's own official ABC atlas tutorial explicitly passes
## `dtype={"cell_label": str}` to every metadata read -- do the same here.
CELL_LABEL_DTYPE = {"cell_label": str}

print(f"Loading ccf_coordinates.csv ...")
ccf = pd.read_csv(f"{raw_dir}/ccf_coordinates.csv", usecols=["cell_label", "parcellation_index"],
                   dtype=CELL_LABEL_DTYPE)

print(f"Loading parcellation_term_membership_name.csv ...")
terms = pd.read_csv(f"{raw_dir}/parcellation_term_membership_name.csv")
terms = terms[["parcellation_index", "division", "structure"]].drop_duplicates("parcellation_index")

print(f"Loading cell_metadata.csv ...")
meta = pd.read_csv(
    f"{raw_dir}/cell_metadata.csv",
    usecols=["cell_label", "brain_section_label", "donor_label", "donor_genotype",
             "donor_sex", "cluster_alias", "x", "y", "z",
             "subclass_confidence_score", "cluster_confidence_score", "high_quality_transfer"],
    dtype=CELL_LABEL_DTYPE,
)

print("Joining cell_metadata + ccf_coordinates + region-name lookup ...")
df = meta.merge(ccf, on="cell_label", how="left").merge(terms, on="parcellation_index", how="left")

df["napari_region"] = df["division"].map(DIVISION_TO_NAPARI).fillna("Unassigned")
df["quint_region"] = df["structure"].fillna("Unassigned")

n_total = len(df)
n_mapped = (df["napari_region"] != "Unassigned").sum()
print(f"\nTotal cells: {n_total}")
print(f"Cells mapped to one of the 5 Napari-equivalent regions: {n_mapped} ({100*n_mapped/n_total:.1f}%)")
print("\nnapari_region distribution:")
print(df["napari_region"].value_counts())

print(f"\nWriting {out_csv} ...")
df.to_csv(out_csv, index=False)
print("Done.")
