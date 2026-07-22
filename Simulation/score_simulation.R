#!/usr/bin/env Rscript
##
## score_simulation.R
##
## Aggregates the ground-truth simulation results across all seeds and the
## four scenarios (null, shared, region_specific, interaction), for every
## model in the manuscript's pipeline (Dream blind/napari/quint, DESeq2
## pseudobulk, Seurat LR blind/napari/quint, Wilcoxon). Produces exactly the
## metrics Reviewer 1 requested (Weakness 1 and Question 3):
##   - False discovery rate (FDR): FP / (TP + FP) among called-significant genes
##   - False positive rate (FPR): FP / (total true-null genes) -- distinct from
##     FDR (R1 explicitly requests both); well-defined even under the null
##     scenario where FDR degenerates to a trivial 100%-or-NA
##   - Power / sensitivity: TP / (TP + FN) among true-DE genes
##   - Effect-size bias: mean(estimate - truth) among true-DE genes
##   - RMSE: sqrt(mean((estimate - truth)^2)) among true-DE genes
##   - 95% CI coverage (where the model reports one): fraction of true-DE
##     genes whose reported CI contains the true log2FC
## reported per scenario x model, with across-seed mean/SD/2.5-97.5% CI, so
## precision is explicit (directly addressing R1 Question 2's complaint about
## unreplicated point estimates).
##
## Scoring logic itself lives in scoring_functions.R (shared with
## score_dev_sweep.R, the composition-imbalance-magnitude sweep).
##
## Usage:
##   Rscript score_simulation.R <results_root> <out_csv>
##
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "scoring_functions.R"))

args         <- commandArgs(trailingOnly = TRUE)
results_root <- ifelse(length(args) >= 1, args[1], "sim_results")
out_csv      <- ifelse(length(args) >= 2, args[2], "simulation_scoring_summary.csv")

scenarios <- c("null", "shared", "region_specific", "interaction")
alpha     <- 0.05

per_seed <- score_results_root(results_root, scenarios, alpha)
if (is.null(per_seed)) stop("No scoreable results found under: ", results_root)

write.csv(per_seed, file.path(dirname(out_csv), paste0("per_seed_", basename(out_csv))), row.names = FALSE)

summary_df <- summarize_per_seed(per_seed, group_cols = "scenario")
write.csv(summary_df, out_csv, row.names = FALSE)

cat("\n===== SIMULATION SCORING SUMMARY =====\n")
print(summary_df[, c("scenario", "model", "n_seeds", "fdr_adj_mean", "fpr_adj_mean", "power_adj_mean",
                      "bias_mean", "rmse_mean", "ci_coverage_mean")], row.names = FALSE, digits = 3)
cat("\nFull summary  -> ", out_csv, "\n")
cat("Per-seed detail -> ", file.path(dirname(out_csv), paste0("per_seed_", basename(out_csv))), "\n")
