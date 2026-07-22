# Independent-Platform Validation (Zhuang-ABCA-1-4, MERFISH)

Tests whether the DE pipeline's region-blind-vs-region-aware behavior
generalizes to an independent platform (MERFISH, Allen Brain Cell Atlas)
and independent tissue replicates (4 real mice), rather than only the
manuscript's own CosMx data. Since these are wild-type reference animals
with no real treatment/control variable, this reuses the manuscript's own
composition-engineering methodology (forced Cortex-fraction imbalance
between two subsampled pseudo-groups) as the test signal, exactly as
already defended for the manuscript's real G1/G2 experiment.

Requires this repo's own `ezy_seq` package to be installed first:
```
pip install -e ../EzySeq_Library/mypythonlibrary
```

## Current pipeline (4-animal, uses Carter's real ezy_seq selection engine)

1. `map_regions_multi.py` — pools all 4 Zhuang-ABCA datasets' cell metadata,
   maps each cell's Allen CCFv3 `parcellation_index` to the manuscript's 5
   Napari-equivalent region categories (reuses the existing CCF atlas, no new
   annotation).
2. `compose_via_ezyseq_multi.py` — splits the 4 real animals 2-vs-2 into
   pseudo-groups and calls `ezy.tag_region_abundance_by_FMT(balance="sample")`
   directly (Carter's actual production function, not a re-implementation).
3. `build_seurat_input_multi.py` — exports G1/G2 Seurat-compatible folders,
   applying the manuscript's own real QC thresholds
   (`filter_and_normalize`'s `min_gene_cnt=20`, `min_t_cnt=100`).
4. `run_independent_de_multi_ezyseq.R` — runs the manuscript's actual
   `de_functions.R::run_analysis_suite` (unmodified) on the exported data.
5. `compare_g1_g2_multi.py` — DEG overlap / logFC & significance correlation /
   bias metrics between G1 and G2.

`*_devsweep*` / `*_ezyseq_array*` scripts run the same pipeline across a
composition-imbalance magnitude sweep (0.5x-3.0x the manuscript's own design
magnitude) rather than a single fixed point.

## Superseded (kept for history, not the current result)

- `real_animal_composition_engineering.py` — the pre-fix hand-rolled
  subsampling this project found to silently diverge from Carter's real
  algorithm; replaced by `compose_via_ezyseq_multi.py`. See root `CLAUDE.md`,
  "we really messed up" entry, for the full incident writeup.
- `synthetic_composition_engineering.py`, `map_regions.py`,
  `build_seurat_input.py`, `compare_g1_g2.py`, `run_independent_de.R` /
  `.sbatch` and their `_remaining` recovery variants — the original
  single-animal (Zhuang-ABCA-1 only) version of this workstream, fully
  superseded by the 4-animal pipeline above.

## SLURM job-recovery scripts

Several `*_remaining*` / `*_g2only*` files are targeted reruns for specific
tasks that hit transient HPC failures (timeouts, port-collision races) during
the original submissions — kept for reproducibility of exactly what was run,
not meant as general-purpose entry points. `*.sbatch` files also hardcode this
project's own HPC account paths and will need adjusting for your own cluster.
