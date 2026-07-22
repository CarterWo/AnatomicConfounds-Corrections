#!/usr/bin/env Rscript
##
## run_stratified_de.R
##
## R2 item "No comparison with stratification": the reviewer argues the
## manuscript claims stratification reduces power relative to the LMM
## approach, but never directly compares the two on the same data. This
## script adds that direct comparison on the ground-truth simulation (where
## power/FDR against a KNOWN true effect can actually be measured, unlike on
## the real data).
##
## For each region present in a given simulated seed/scenario, subsets to
## ONLY that region's cells and runs run_de_blind() (unmodified from
## de_functions.R -- a stratified analysis by definition drops the region
## covariate/random-intercept, since only one region's cells are present) on
## that subset alone. This mirrors what "stratified analysis" means in
## practice: analyze each anatomical region separately rather than pooling
## with region as a covariate/random effect.
##
## Does NOT re-run the manuscript's seven standard models or the interaction
## model -- those already exist on disk. Only regenerates raw/normalized
## counts (deleted after the original run to save disk; deterministic given
## the same seed) if not already present, exactly as
## run_simulation_interaction_only.R does.
##
## Usage:
##   Rscript run_stratified_de.R <seed> <scenario> <sim_root> <results_root>
##
suppressPackageStartupMessages({
  library(Seurat)
})

args         <- commandArgs(trailingOnly = TRUE)
seed         <- as.integer(ifelse(length(args) >= 1, args[1], Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
scenario     <- ifelse(length(args) >= 2, args[2], "region_specific")
sim_root     <- ifelse(length(args) >= 3, args[3], "sim_out")
results_root <- ifelse(length(args) >= 4, args[4], "sim_results")

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "..", "DE_scripts", "de_functions.R"))

param <- SnowParam(workers = 4, type = "SOCK", progressbar = FALSE, exportglobals = FALSE)

data_dir <- file.path(sim_root, scenario, paste0("seed_", seed))
out_dir  <- file.path(results_root, scenario, paste0("seed_", seed))
strat_dir <- file.path(out_dir, "Stratified_Analysis")

if (!dir.exists(out_dir)) stop(paste("Expected existing result dir not found:", out_dir,
                                      "- run the original array job for this seed/scenario first."))
dir.create(strat_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(file.path(data_dir, "raw_counts.csv"))) {
  message(sprintf(">>> Regenerating simulated data for seed=%d scenario=%s", seed, scenario))
  system2("Rscript", c(file.path(script_dir, "simulate_ground_truth.R"), seed, scenario, sim_root))
}

cfg <- make_cfg(data_dir)
seu <- build_seurat_from_folder(cfg)
seu <- prepare_metadata(seu)

regions <- sort(unique(seu$napari_region))
message(sprintf(">>> Stratified DE for seed=%d scenario=%s across %d regions: %s",
                 seed, scenario, length(regions), paste(regions, collapse = ", ")))

MIN_CELLS_PER_ARM <- 20  # skip a region/treatment-arm too small to fit reliably

for (r in regions) {
  seu_r <- subset(seu, subset = napari_region == r)
  arm_counts <- table(seu_r$Treatment)
  if (length(arm_counts) < 2 || any(arm_counts < MIN_CELLS_PER_ARM)) {
    message(sprintf("    Skipping region=%s (insufficient cells per arm: %s)",
                     r, paste(names(arm_counts), arm_counts, collapse = ", ")))
    next
  }
  message(sprintf("    Region=%s: %d cells (%s)", r, ncol(seu_r),
                   paste(names(arm_counts), arm_counts, collapse = ", ")))
  run_de_blind(seu_r,
    dataset_name = paste0("SIM_stratified_", scenario, "_seed", seed, "_", r),
    out_prefix   = paste0("SIM_", scenario, "_seed", seed, "_stratified_", r, "_dream_blind"),
    save_dir     = strat_dir)
}

## Free disk again -- same reasoning as run_simulation_array.sbatch.
if (file.exists(file.path(data_dir, "raw_counts.csv")))        file.remove(file.path(data_dir, "raw_counts.csv"))
if (file.exists(file.path(data_dir, "normalized_counts.csv"))) file.remove(file.path(data_dir, "normalized_counts.csv"))

message(sprintf(">>> DONE (stratified) seed=%d scenario=%s -> %s", seed, scenario, strat_dir))
