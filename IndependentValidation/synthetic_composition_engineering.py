#!/usr/bin/env python3
"""
synthetic_composition_engineering.py

Adapts the manuscript's own composition-engineering methodology
(ezy_seq.tag_region_abundance_by_FMT) to the Zhuang-ABCA-1 independent
MERFISH dataset, which -- unlike the manuscript's real CosMx data -- has NO
real treatment/control variable (single wild-type reference animal; see
CLAUDE.md for why a fabricated biological contrast was rejected in favor of
this design).

DESIGN, stated plainly for the response letter / methods text:
  Because there is no real treatment/control arm in this dataset, this
  script does NOT claim to test a real biological effect. Instead, it
  builds synthetic "pseudo-samples" and randomly splits them into two
  pseudo-groups with NO systematic relationship to any real biological
  variable. The only difference deliberately introduced between the two
  pseudo-groups is the SAME Cortex-fraction manipulation (+-1 SD) already
  used and defended in the manuscript's real composition-engineering
  experiment.

  IMPORTANT CORRECTION vs. a naive first attempt: Zhuang-ABCA-1's 147
  `brain_section_label` values are serial anterior-posterior (AP) slices
  of ONE brain, not independent whole-brain replicates the way the
  manuscript's real animals are -- a posterior section can be ~0% cortex
  (pure cerebellum/hindbrain) while an anterior section can be mostly
  cortex, purely as a function of AP position, nothing to do with
  "sample-level" variability. Using raw sections as the resampling unit
  was tried first and produced a cross-section Cortex-fraction SD
  numerically equal to the mean (extreme, physiologically-driven
  heterogeneity, not sampling noise) -- the ±1 SD targets became
  unreachable and the achieved fractions did not track the intended
  hi/lo split at all.
  FIX: sections are first sorted by their mean AP (z) coordinate, then
  interleaved into `n_pseudo_samples` synthetic "pseudo-samples" (section
  rank i -> pseudo-sample i % n_pseudo_samples), so each pseudo-sample
  spans the FULL AP extent of the brain and has whole-brain-like Cortex
  composition -- analogous to how each of the manuscript's real animals
  is a full brain, not a single AP position. This mirrors the "null"
  scenario in the ground-truth splatter simulation (no true treatment
  effect, only composition imbalance) -- but on real, independent MERFISH
  data. Any DE signal that emerges is by construction a pure
  composition-driven artifact; testing whether anatomical-awareness
  suppresses it here is the actual generalizability claim Reviewer 2
  asked for.

Usage:
    python3 synthetic_composition_engineering.py <mapped_csv> <out_csv> [dev] [seed] [n_pseudo_samples]
      mapped_csv       : output of map_regions.py (must have
                         brain_section_label, napari_region, z columns)
      out_csv          : where to write the tagged cell table (adds columns
                         pseudo_sample, pseudo_group in {"A","B"}, G1, G2 boolean)
      dev              : SD-multiple for the Cortex-fraction shift (default
                         1.0, matches the manuscript's original design)
      seed             : RNG seed (default 0)
      n_pseudo_samples : number of AP-interleaved pseudo-samples to build
                         (default 8, analogous to the manuscript's 6
                         real animals but with a bit more resolution)
"""
import sys
import numpy as np
import pandas as pd

mapped_csv       = sys.argv[1] if len(sys.argv) > 1 else "cell_metadata_with_regions.csv"
out_csv          = sys.argv[2] if len(sys.argv) > 2 else "cell_metadata_tagged.csv"
dev              = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
seed             = int(sys.argv[4]) if len(sys.argv) > 4 else 0
n_pseudo_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 8

roi = "Cortex"
rng = np.random.default_rng(seed)

print(f"Loading {mapped_csv} ...")
## cell_label must be forced to string dtype -- without this, pandas
## silently parses these 39-digit all-numeric IDs as Python int, which
## then never compares equal to the h5ad's string-typed obs_names
## downstream (found and fixed during Seurat-input export smoke testing;
## see CLAUDE.md and map_regions.py for full detail).
df = pd.read_csv(mapped_csv, dtype={"cell_label": str})

# --- Build AP-interleaved pseudo-samples (see module docstring for why raw
# brain_section_label cannot be used directly as the resampling unit). ---
section_ap = df.groupby("brain_section_label")["z"].mean().sort_values()
sorted_sections = section_ap.index.to_numpy()
section_to_pseudo_sample = {
    s: f"pseudo_{i % n_pseudo_samples}" for i, s in enumerate(sorted_sections)
}
df["pseudo_sample"] = df["brain_section_label"].map(section_to_pseudo_sample)

pseudo_samples = sorted(df["pseudo_sample"].unique())
print(f"{len(sorted_sections)} AP-sorted brain sections interleaved into "
      f"{len(pseudo_samples)} pseudo-samples (i.e. section rank i -> "
      f"pseudo_{{i % {n_pseudo_samples}}}), each spanning the full AP extent.")

# Sanity check: pseudo-sample Cortex fractions should now be much closer to
# each other than the raw per-section fractions were.
frac_by_pseudo = (
    df.groupby("pseudo_sample")["napari_region"]
      .apply(lambda s: (s == roi).mean())
)
print("Cortex fraction per pseudo-sample:")
print(frac_by_pseudo.to_string())

# --- Random split of pseudo-samples (NOT raw sections) into two pseudo-groups
# with no systematic biological difference by construction. ---
pseudo_samples = list(pseudo_samples)
rng.shuffle(pseudo_samples)
half = len(pseudo_samples) // 2
pseudo_sample_to_group = {
    s: ("A" if i < half else "B") for i, s in enumerate(pseudo_samples)
}
df["pseudo_group"] = df["pseudo_sample"].map(pseudo_sample_to_group)
print(f"\n{len(pseudo_samples)} pseudo-samples split into pseudo-group A "
      f"(n={half}) / B (n={len(pseudo_samples)-half}), random_state={seed}.")

roi_mean = frac_by_pseudo.mean()
## NOT using this dataset's own cross-pseudo-sample SD as the shift
## magnitude: a single animal cannot supply a statistically meaningful
## between-sample variance estimate (confirmed empirically -- AP-
## interleaving that fixes the resampling-unit problem above also, by
## construction, averages out nearly all cross-pseudo-sample composition
## variance: measured SD was 0.0051 against a mean of 0.2118, an order of
## magnitude too small to produce any real separation between G1/G2).
## Instead, reuse the manuscript's OWN already-reported, already-defended
## cross-ANIMAL SD for the isocortex fraction (7.4 percentage points,
## Napari-measured across the real 6-8 animal study -- see
## Simulation/simulate_ground_truth.R and CLAUDE.md for the same number
## used the same way in the splatter ground-truth simulation) as an
## external, literature/manuscript-anchored shift magnitude, applied here
## as an absolute percentage-point shift off THIS dataset's own baseline
## mean -- exactly the same principle as reusing 26.4%/7.4% for the
## synthetic simulation rather than inventing a new number from whatever
## data happens to be at hand.
MANUSCRIPT_ROI_SD_PP = 0.074
hi = min(0.9, roi_mean + dev * MANUSCRIPT_ROI_SD_PP)
lo = max(0.05, roi_mean - dev * MANUSCRIPT_ROI_SD_PP)
print(f"\nCortex fraction across pseudo-samples: mean={roi_mean:.4f} "
      f"(this dataset's own baseline; SD NOT used -- see comment above) "
      f"-> hi={hi:.4f}, lo={lo:.4f} (dev={dev:g} x manuscript-reported SD "
      f"{MANUSCRIPT_ROI_SD_PP:.3f})")


def subsample_to_target(sub_df: pd.DataFrame, target_frac: float, rng) -> pd.Index:
    """Per-pseudo-sample subsample: hit target_frac for `roi`, keep all
    other regions at their original relative proportions -- same recipe as
    ezy_seq._allocate_group / _alloc_by_baseline_with_caps, simplified to a
    single scalar target."""
    selected = []
    for pseudo_sample, ps_df in sub_df.groupby("pseudo_sample"):
        roi_pool = ps_df.index[ps_df["napari_region"] == roi]
        other_pool = ps_df.index[ps_df["napari_region"] != roi]
        n_total = len(ps_df)
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
