# Ground-Truth Simulation

Literature-parameterized synthetic testbed (splatter, Bioconductor) with a
known injected treatment effect, used to score the DE pipeline's FDR, power,
effect-size bias, RMSE, and CI coverage against a known ground truth. Built
for the R1 "we need to see this validated against a ground truth" request;
fully independent of the manuscript's own CosMx data (only the composition-
imbalance *magnitude*, 26.4% +/- 7.4%, is anchored to the manuscript's own
Napari-measured isocortex statistic; every gene-expression parameter and the
injected treatment effect are literature-sourced or synthetic).

Requires this repo's own `ezy_seq` package to be installed first:
```
pip install -e ../EzySeq_Library/mypythonlibrary
```
(`simulate_ground_truth.R` shells out to `compose_via_ezyseq_sim.py`, which
calls Carter's real composition-engineering selection engine directly rather
than a re-implementation of it.)

## Entry points

- `simulate_ground_truth.R <seed> <scenario> <out_root> [dev]` — generates one
  synthetic dataset (4 scenarios: `null`, `shared`, `region_specific`,
  `interaction`; cell types `Excitatory_Neuron`/`Astrocyte`/`Microglia` with
  real marker-gene expression). `dev` (default 1.0) scales the composition-
  imbalance magnitude in SD units.
- `run_simulation_de.R` — loads one simulated dataset and runs the manuscript's
  actual `de_functions.R::run_analysis_suite` (Dream blind/napari/quint,
  DESeq2 pseudobulk, Seurat LR blind/napari/quint, Wilcoxon) unmodified.
- `run_simulation_interaction_only.R` — adds the exploratory
  `run_de_dream_interaction` model (Treatment x region as a fixed effect); only
  meaningful for `region_specific`/`interaction` scenarios, and expects the
  main array to have already been run for that seed/scenario.
- `run_stratified_de.R` — per-region-stratified DE comparison (R2 Item 3); also
  expects the main array's results to already exist for that seed/scenario.
- `score_simulation.R` / `score_dev_sweep.R` / `score_stratified.R` (share
  `scoring_functions.R`) — aggregate raw DE output into FDR/power/bias/RMSE/CI
  coverage summaries. Pre-computed summary CSVs are in `results/`.
- `*.sbatch` — SLURM array submission scripts (HPC-specific job-scheduler
  paths inside will need adjusting for your own cluster/account).

## Results

`results/*.csv` are the final, cell-type-aware (50 seeds/scenario) scoring
summaries referenced in the manuscript and response letter. See each script's
header comment for exactly which analysis produced which file.
