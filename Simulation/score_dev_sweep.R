#!/usr/bin/env Rscript
##
## score_dev_sweep.R
##
## Aggregates the composition-imbalance-magnitude sweep (run_dev_sweep_array.sbatch)
## across all dev levels, tagging each row with its `dev` value, so bias/FPR/
## power can be reported as a function of imbalance magnitude -- directly
## answering R1 Question 2's "test several levels of imbalance... show
## whether the observed bias increases smoothly with regional imbalance and
## whether the proposed adjustment remains stable under mild and severe
## perturbations." This is a SYNTHETIC/simulation-side complement to
## Carter's real-data composition-engineering resampling (Weakness 2 /
## Question 2), not a duplicate of it -- dev=1.0 from the main 200-task
## array is included for the full 0.5-1.0-1.5-2.0 dose-response curve.
##
## Usage:
##   Rscript score_dev_sweep.R <devsweep_results_base> <main_results_root> <out_csv>
##   devsweep_results_base: parent dir containing dev_<X> subdirs (each laid
##     out exactly like a normal sim_results root: dev_<X>/<scenario>/seed_<n>/...)
##   main_results_root: the original 4-scenario x 50-seed results root, used
##     to pull in dev=1.0 (already run, no need to re-simulate it)
##
script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "scoring_functions.R"))

args               <- commandArgs(trailingOnly = TRUE)
devsweep_base       <- ifelse(length(args) >= 1, args[1], "sim_results_devsweep")
main_results_root  <- ifelse(length(args) >= 2, args[2], "sim_results")
out_csv            <- ifelse(length(args) >= 3, args[3], "dev_sweep_scoring_summary.csv")

alpha <- 0.05
## Only the two scenarios the dev-sweep array actually ran.
sweep_scenarios <- c("null", "region_specific")

dev_dirs <- list.dirs(devsweep_base, recursive = FALSE)
dev_labels <- gsub("^dev_", "", basename(dev_dirs))

all_per_seed <- list()

for (i in seq_along(dev_dirs)) {
  ps <- score_results_root(dev_dirs[i], sweep_scenarios, alpha)
  if (is.null(ps)) { message("No scoreable results under: ", dev_dirs[i]); next }
  ps$dev <- dev_labels[i]
  all_per_seed[[length(all_per_seed) + 1]] <- ps
}

## Pull in dev=1.0 from the main (already-scored) results root for the full
## dose-response curve, restricted to the same two scenarios.
ps_main <- score_results_root(main_results_root, sweep_scenarios, alpha)
if (!is.null(ps_main)) {
  ps_main$dev <- "1.0"
  all_per_seed[[length(all_per_seed) + 1]] <- ps_main
} else {
  message("Warning: no results found under main_results_root (", main_results_root,
          ") -- dose-response curve will be missing the dev=1.0 reference point.")
}

if (length(all_per_seed) == 0) stop("No scoreable results found in the dev sweep or main results root.")

per_seed <- do.call(rbind, all_per_seed)
per_seed$dev <- as.numeric(per_seed$dev)

write.csv(per_seed, file.path(dirname(out_csv), paste0("per_seed_", basename(out_csv))), row.names = FALSE)

summary_df <- summarize_per_seed(per_seed, group_cols = c("dev", "scenario"))
summary_df <- summary_df[order(summary_df$scenario, summary_df$model, summary_df$dev), ]
write.csv(summary_df, out_csv, row.names = FALSE)

cat("\n===== COMPOSITION-IMBALANCE MAGNITUDE SWEEP SUMMARY =====\n")
print(summary_df[, c("dev", "scenario", "model", "n_seeds", "fpr_adj_mean", "power_adj_mean",
                      "bias_mean", "rmse_mean")], row.names = FALSE, digits = 3)
cat("\nFull summary  -> ", out_csv, "\n")
cat("Per-seed detail -> ", file.path(dirname(out_csv), paste0("per_seed_", basename(out_csv))), "\n")
