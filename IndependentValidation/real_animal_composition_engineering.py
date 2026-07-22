#!/usr/bin/env python3
"""
real_animal_composition_engineering.py

Real-animal successor to synthetic_composition_engineering.py. With only
Zhuang-ABCA-1 (a single animal), the resampling unit had to be an
AP-interleaved "pseudo-sample" built from one brain's own sections, and the
Cortex-fraction shift magnitude had to be borrowed from the manuscript's
own reported cross-animal SD (7.4 percentage points), because a single
animal cannot supply a real between-sample variance estimate.

With Zhuang-ABCA-1/2/3/4 pooled (map_regions_multi.py output), the
resampling unit becomes each dataset's own single real animal (donor_label
is literally "Zhuang-ABCA-N", one per dataset) -- 4 real biological
replicates, not synthetic slices of one brain. This lets the Cortex-
fraction shift magnitude be computed directly from these 4 real animals'
own cross-animal SD, rather than borrowed from the manuscript.

HONEST LIMITATION, stated plainly (do not silently omit): n=4 animals is
still a very small sample for estimating a between-animal SD -- the
computed value carries substantial sampling uncertainty at this N, the
same small-N caveat that applies elsewhere in this project. This script
reports BOTH the newly-computed real 4-animal SD and the manuscript's own
7.4pp figure side by side, and uses the LARGER of the two as the shift
magnitude (the more conservative, harder-to-achieve target) unless
overridden -- rather than picking whichever number is more convenient.

Design otherwise mirrors synthetic_composition_engineering.py exactly:
2 real animals to pseudo-group A, 2 to pseudo-group B (random, no
systematic biological difference by construction -- all 4 are the same
wild-type reference genotype), then the manuscript's own Cortex-fraction
subsampling recipe applied per animal.

Usage:
    python3 real_animal_composition_engineering.py <mapped_csv> <out_csv> [dev] [seed]
"""
import sys
import numpy as np
import pandas as pd

mapped_csv = sys.argv[1] if len(sys.argv) > 1 else "cell_metadata_with_regions_multi.csv"
out_csv    = sys.argv[2] if len(sys.argv) > 2 else "cell_metadata_tagged_multi.csv"
dev        = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
seed       = int(sys.argv[4]) if len(sys.argv) > 4 else 0

roi = "Cortex"
rng = np.random.default_rng(seed)

print(f"Loading {mapped_csv} ...")
df = pd.read_csv(mapped_csv, dtype={"cell_label": str})

animals = sorted(df["donor_label"].unique())
print(f"\n{len(animals)} real animals found: {animals}")
if len(animals) != 4:
    print(f"WARNING: expected 4 animals (Zhuang-ABCA-1..4), found {len(animals)}.")

frac_by_animal = df.groupby("donor_label")["napari_region"].apply(lambda s: (s == roi).mean())
print("\nCortex fraction per real animal:")
print(frac_by_animal.to_string())

real_mean = frac_by_animal.mean()
real_sd = frac_by_animal.std(ddof=1)
MANUSCRIPT_ROI_SD_PP = 0.074
shift_used = max(real_sd, MANUSCRIPT_ROI_SD_PP)
print(f"\nReal 4-animal Cortex-fraction mean={real_mean:.4f}, SD={real_sd:.4f} (n=4 -- "
      f"high sampling uncertainty at this N, reported for transparency, not treated "
      f"as a precise estimate).")
print(f"Manuscript's own cross-animal (Napari-measured, real study) SD = {MANUSCRIPT_ROI_SD_PP:.4f}.")
print(f"Using shift magnitude = max(real, manuscript) = {shift_used:.4f} "
      f"(the more conservative / harder-to-achieve of the two, not cherry-picked).")

# Random 2-vs-2 split of the 4 real animals into pseudo-groups -- no
# systematic biological difference by construction (all 4 are the same
# wild-type reference genotype across different individual mice).
animals_shuffled = list(animals)
rng.shuffle(animals_shuffled)
half = len(animals_shuffled) // 2
animal_to_group = {a: ("A" if i < half else "B") for i, a in enumerate(animals_shuffled)}
df["pseudo_group"] = df["donor_label"].map(animal_to_group)
print(f"\n4 real animals split into pseudo-group A ({[a for a in animals_shuffled[:half]]}) / "
      f"B ({[a for a in animals_shuffled[half:]]}), random_state={seed}.")

hi = min(0.9, real_mean + dev * shift_used)
lo = max(0.05, real_mean - dev * shift_used)
print(f"\nTarget Cortex fractions: hi={hi:.4f}, lo={lo:.4f} (dev={dev:g} x {shift_used:.4f})")


def subsample_to_target(sub_df: pd.DataFrame, target_frac: float, rng) -> pd.Index:
    """Per-animal subsample to target_frac for `roi`, keeping other regions
    at their original relative proportions -- same recipe as
    ezy_seq._allocate_group / _alloc_by_baseline_with_caps."""
    selected = []
    for animal, a_df in sub_df.groupby("donor_label"):
        roi_pool = a_df.index[a_df["napari_region"] == roi]
        other_pool = a_df.index[a_df["napari_region"] != roi]
        n_total = len(a_df)
        n_roi_target = min(len(roi_pool), round(target_frac * n_total))
        n_other_target = min(len(other_pool), n_total - n_roi_target)
        if n_roi_target > 0:
            selected.extend(rng.choice(roi_pool, size=n_roi_target, replace=False).tolist())
        if n_other_target > 0:
            selected.extend(rng.choice(other_pool, size=n_other_target, replace=False).tolist())
    return pd.Index(selected)


def build_scenario(target_A: float, target_B: float, rng) -> pd.Index:
    idx_a = subsample_to_target(df[df["pseudo_group"] == "A"], target_A, rng)
    idx_b = subsample_to_target(df[df["pseudo_group"] == "B"], target_B, rng)
    return idx_a.union(idx_b)


print("\nBuilding G1 (pseudo-A enriched, pseudo-B depleted) ...")
g1_idx = build_scenario(hi, lo, np.random.default_rng(seed))
print("Building G2 (pseudo-A depleted, pseudo-B enriched) ...")
g2_idx = build_scenario(lo, hi, np.random.default_rng(seed))

df["G1"] = df.index.isin(g1_idx)
df["G2"] = df.index.isin(g2_idx)

print(f"\nTagged {df['G1'].sum()} cells -> G1; {df['G2'].sum()} cells -> G2")
print("\nCortex fraction achieved, by pseudo_group x scenario:")
for scen_col in ("G1", "G2"):
    sub = df[df[scen_col]]
    print(f"  {scen_col}:", sub.groupby("pseudo_group")["napari_region"]
          .apply(lambda s: f"{(s == roi).mean():.4f}").to_dict())

print(f"\nWriting {out_csv} ...")
df.to_csv(out_csv, index=False)
print("Done.")
