#!/usr/bin/env Rscript
##
## run_simulation_interaction_only.R
##
## Adds the EXPLORATORY Treatment x region interaction model (see
## de_functions.R::run_de_dream_interaction) to an ALREADY-COMPLETED
## simulation result set, without re-running the manuscript's seven standard
## models (dream blind/napari/quint, DESeq2 PB, Seurat LR x3, Wilcoxon) --
## those results already exist on disk from the original full array run.
##
## Only re-does the (cheap) simulate_ground_truth.R step to regenerate the
## raw/normalized count CSVs that run_simulation_array.sbatch deletes after
## each task to save disk (kept for that reason -- see CLAUDE.md), then
## builds the Seurat object and fits ONLY the interaction model.
##
## Usage:
##   Rscript run_simulation_interaction_only.R <seed> <scenario> <sim_root> <results_root>
##   scenario should be "region_specific" or "interaction" (the only ones
##   where this model is meaningful -- see de_functions.R for rationale).
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
global_dir <- file.path(out_dir, "Global_CT_Analysis")

if (!dir.exists(global_dir)) stop(paste("Expected existing result dir not found:", global_dir,
                                         "- run the original array job for this seed/scenario first."))

## Regenerate raw/normalized counts (deleted after the original run to save
## disk) -- deterministic/reproducible given the same seed.
if (!file.exists(file.path(data_dir, "raw_counts.csv"))) {
  message(sprintf(">>> Regenerating simulated data for seed=%d scenario=%s", seed, scenario))
  system2("Rscript", c(file.path(script_dir, "simulate_ground_truth.R"), seed, scenario, sim_root))
}

cfg <- make_cfg(data_dir)
seu <- build_seurat_from_folder(cfg)
seu <- prepare_metadata(seu)

message(sprintf(">>> Running interaction model ONLY for seed=%d scenario=%s (%d cells, %d genes)",
                 seed, scenario, ncol(seu), nrow(seu)))
run_de_dream_interaction(seu,
  dataset_name = paste0("SIM_dream_interaction_", scenario, "_seed", seed),
  out_prefix   = paste0("SIM_", scenario, "_seed", seed, "_dream_interaction"),
  save_dir     = global_dir,
  region_col   = "napari_region",
  region_ref   = "Cortex")

## Free disk again -- same reasoning as run_simulation_array.sbatch.
file.remove(file.path(data_dir, "raw_counts.csv"))
file.remove(file.path(data_dir, "normalized_counts.csv"))

message(sprintf(">>> DONE (interaction-only) seed=%d scenario=%s -> %s", seed, scenario, global_dir))
