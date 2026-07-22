#!/usr/bin/env Rscript
##
## score_stratified.R
##
## Scores the region-stratified DE results (run_stratified_de.R) against the
## same ground truth used for the whole-dataset models, and produces two
## comparisons:
##
## 1. Per-region power/FDR/bias/RMSE (e.g. testing ONLY within Cortex, where
##    the region_specific/interaction scenarios place the true effect) vs.
##    the already-scored whole-dataset dream_blind/dream_napari/
##    dream_interaction power on the SAME seeds -- isolates how much power is
##    lost purely from analyzing a smaller per-region cell count.
## 2. A "stratified_combined" pseudo-model: for each gene, take the minimum
##    raw p-value across all regions tested, Bonferroni-correct within-gene
##    for the number of regions it was tested in, then BH-adjust across
##    genes -- the standard way multiple independent regional tests would be
##    combined/corrected in practice. Compared against the null scenario's
##    whole-dataset FPR to test whether stratifying and combining inflates
##    false positives from multiple testing across regions.
##
## Usage:
##   Rscript score_stratified.R <results_root> <out_csv_prefix>
##
suppressPackageStartupMessages({ library(dplyr) })

args           <- commandArgs(trailingOnly = TRUE)
results_root   <- ifelse(length(args) >= 1, args[1], "sim_results")
out_prefix     <- ifelse(length(args) >= 2, args[2], "stratified_scoring")

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "scoring_functions.R"))

scenarios <- c("null", "shared", "region_specific", "interaction")
alpha <- 0.05

per_region_rows   <- list()
combined_rows     <- list()

for (scenario in scenarios) {
  scen_dir <- file.path(results_root, scenario)
  if (!dir.exists(scen_dir)) next
  seed_dirs <- list.dirs(scen_dir, recursive = FALSE)

  for (sd in seed_dirs) {
    seed <- gsub("seed_", "", basename(sd))
    gt_fp <- file.path(sd, "ground_truth.csv")
    if (!file.exists(gt_fp)) next
    gt <- read.csv(gt_fp)
    gt$is_true_de <- abs(gt$true_log2fc) > 1e-8

    strat_dir <- file.path(sd, "Stratified_Analysis")
    if (!dir.exists(strat_dir)) next
    files <- list.files(strat_dir, pattern = "_stratified_.*_dream_blind\\.csv$", full.names = TRUE)
    if (length(files) == 0) next

    region_spec <- model_specs$dream_blind  # same columns as run_de_blind output
    region_results <- list()

    for (fp in files) {
      region <- sub(".*_stratified_(.*)_dream_blind\\.csv$", "\\1", basename(fp))
      res <- load_and_normalize(fp, region_spec)
      if (is.null(res)) next
      sc <- score_one(res, gt, alpha)
      if (!is.null(sc)) {
        sc$scenario <- scenario
        sc$seed     <- seed
        sc$region   <- region
        sc$model    <- paste0("stratified_", region)
        per_region_rows[[length(per_region_rows) + 1]] <- sc
      }
      res$region <- region
      region_results[[region]] <- res
    }

    ## --- combined-across-regions pseudo-model ---
    if (length(region_results) > 0) {
      all_res <- do.call(rbind, region_results)
      n_regions_tested <- length(region_results)
      combined <- all_res %>%
        group_by(Gene) %>%
        summarise(
          n_tested = n(),
          min_p    = min(pval, na.rm = TRUE),
          log2fc   = log2fc[which.min(pval)],
          .groups = "drop"
        ) %>%
        mutate(
          pval_bonf = pmin(1, min_p * n_tested),
        )
      combined$padj <- p.adjust(combined$pval_bonf, method = "BH")
      combined$pval <- combined$pval_bonf
      sc <- score_one(combined[, c("Gene", "log2fc", "pval", "padj")] %>%
                         mutate(ci_lo = NA_real_, ci_hi = NA_real_),
                       gt, alpha)
      if (!is.null(sc)) {
        sc$scenario <- scenario
        sc$seed     <- seed
        sc$model    <- "stratified_combined"
        combined_rows[[length(combined_rows) + 1]] <- sc
      }
    }
  }
}

per_region_df <- if (length(per_region_rows) > 0) do.call(rbind, per_region_rows) else NULL
combined_df   <- if (length(combined_rows)   > 0) do.call(rbind, combined_rows)   else NULL

if (!is.null(per_region_df)) {
  write.csv(per_region_df, paste0(out_prefix, "_per_region_per_seed.csv"), row.names = FALSE)
  summ <- per_region_df %>%
    group_by(scenario, region) %>%
    summarise(across(all_of(scoring_metrics), ~ mean(.x, na.rm = TRUE)), n_seeds = n(), .groups = "drop")
  write.csv(summ, paste0(out_prefix, "_per_region_summary.csv"), row.names = FALSE)
  message("Wrote per-region scoring: ", paste0(out_prefix, "_per_region_summary.csv"))
}

if (!is.null(combined_df)) {
  write.csv(combined_df, paste0(out_prefix, "_combined_per_seed.csv"), row.names = FALSE)
  summ <- combined_df %>%
    group_by(scenario) %>%
    summarise(across(all_of(scoring_metrics), ~ mean(.x, na.rm = TRUE)), n_seeds = n(), .groups = "drop")
  write.csv(summ, paste0(out_prefix, "_combined_summary.csv"), row.names = FALSE)
  message("Wrote combined-across-regions scoring: ", paste0(out_prefix, "_combined_summary.csv"))
}

if (is.null(per_region_df) && is.null(combined_df)) {
  stop("No stratified results found under ", results_root, " -- did run_stratified_array.sbatch complete?")
}
