#!/usr/bin/env Rscript
##
## run_simulation_de.R
##
## Runs the manuscript's actual DE pipeline (unmodified, sourced from
## DE_scripts/de_functions.R) on one simulated dataset (one seed x scenario
## produced by simulate_ground_truth.R). This is what makes the ground-truth
## simulation a genuine test of the deployed models, not a re-implementation.
##
## Usage:
##   Rscript run_simulation_de.R <seed> <scenario> <sim_root> <results_root>
##
suppressPackageStartupMessages({
  library(Seurat)
})

args         <- commandArgs(trailingOnly = TRUE)
seed         <- as.integer(ifelse(length(args) >= 1, args[1], Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
scenario     <- ifelse(length(args) >= 2, args[2], "null")
sim_root     <- ifelse(length(args) >= 3, args[3], "sim_out")
results_root <- ifelse(length(args) >= 4, args[4], "sim_results")

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "..", "DE_scripts", "de_functions.R"))

param <- SnowParam(workers = 4, type = "SOCK", progressbar = FALSE, exportglobals = FALSE)

data_dir <- file.path(sim_root, scenario, paste0("seed_", seed))
if (!dir.exists(data_dir)) stop(paste("Simulated data not found:", data_dir,
                                       "- run simulate_ground_truth.R first."))

cfg <- make_cfg(data_dir)
seu <- build_seurat_from_folder(cfg)
seu <- prepare_metadata(seu)
seu$cell_type <- seu$cell_type   # already present from simulation metadata

out_dir <- file.path(results_root, scenario, paste0("seed_", seed))
dirs <- list(
  global = file.path(out_dir, "Global_CT_Analysis"),
  local  = file.path(out_dir, "Local_Regional_Analysis"),
  pb     = file.path(out_dir, "Pseudobulk_Validation")
)
sapply(dirs, function(x) if (!dir.exists(x)) dir.create(x, recursive = TRUE))

message(sprintf(">>> Running analysis suite for seed=%d scenario=%s (%d cells, %d genes)",
                 seed, scenario, ncol(seu), nrow(seu)))
run_analysis_suite(seu, label = "SIM", file_tag = paste0(scenario, "_seed", seed), dirs = dirs)

## Copy ground truth alongside results for the scoring step.
file.copy(file.path(data_dir, "ground_truth.csv"), file.path(out_dir, "ground_truth.csv"), overwrite = TRUE)

message(sprintf(">>> DONE seed=%d scenario=%s -> %s", seed, scenario, out_dir))
