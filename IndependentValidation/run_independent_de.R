#!/usr/bin/env Rscript
##
## run_independent_de.R
##
## Runs the manuscript's actual DE pipeline (unmodified, sourced from
## DE_scripts/de_functions.R) on one composition-engineered scenario (g1 or
## g2) of the independent Zhuang-ABCA-1 MERFISH dataset -- exactly the same
## principle already used for the synthetic splatter ground-truth
## simulation: the deployed models are tested completely as-is, not
## reimplemented, so this is a genuine test of generalization to an
## independent platform (Reviewer 2's request), not a new analysis.
##
## `Treatment` here is `pseudo_group` (A/B), a random split with no real
## biological difference by construction -- see CLAUDE.md "Independent-
## dataset validation" section for the full design rationale (no real
## treatment/control variable exists in this single wild-type reference
## animal).
##
## Usage:
##   Rscript run_independent_de.R <scenario> <seurat_input_root> <results_root>
##   scenario in {g1, g2}
##
suppressPackageStartupMessages({
  library(Seurat)
})

args             <- commandArgs(trailingOnly = TRUE)
scenario         <- ifelse(length(args) >= 1, args[1], "g1")
seurat_input_root<- ifelse(length(args) >= 2, args[2], "seurat_input")
results_root     <- ifelse(length(args) >= 3, args[3], "de_results")

stopifnot(scenario %in% c("g1", "g2"))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "..", "DE_scripts", "de_functions.R"))

param <- SnowParam(workers = 4, type = "SOCK", progressbar = FALSE, exportglobals = FALSE)

data_dir <- file.path(seurat_input_root, scenario)
if (!dir.exists(data_dir)) stop(paste("Seurat input not found:", data_dir,
                                       "- run build_seurat_input.py first."))

cfg <- make_cfg(data_dir)
seu <- build_seurat_from_folder(cfg)
seu <- prepare_metadata(seu)

out_dir <- file.path(results_root, scenario)
dirs <- list(
  global = file.path(out_dir, "Global_CT_Analysis"),
  local  = file.path(out_dir, "Local_Regional_Analysis"),
  pb     = file.path(out_dir, "Pseudobulk_Validation")
)
sapply(dirs, function(x) if (!dir.exists(x)) dir.create(x, recursive = TRUE))

message(sprintf(">>> Running analysis suite for scenario=%s (%d cells, %d genes)",
                 scenario, ncol(seu), nrow(seu)))
message(sprintf(">>> Treatment level counts: %s",
                 paste(capture.output(print(table(seu$Treatment))), collapse = " | ")))
run_analysis_suite(seu, label = "INDEP", file_tag = scenario, dirs = dirs)

message(sprintf(">>> DONE scenario=%s -> %s", scenario, out_dir))
