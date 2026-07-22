## scoring_functions.R
##
## Shared scoring logic for simulation results, sourced by both
## score_simulation.R (main 4-scenario x 50-seed array) and
## score_dev_sweep.R (composition-imbalance magnitude sweep). Extracted so
## the two entry points share identical scoring logic rather than
## duplicating it -- no behavior change from the original score_simulation.R.

suppressPackageStartupMessages({
  library(dplyr)
})

## Each entry: model label -> list(subdir, filename_pattern, logfc_col, p_col, padj_col, ci_lo_col, ci_hi_col)
model_specs <- list(
  dream_blind  = list(dir = "Local_Regional_Analysis", pattern = "_dream_blind.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = "CI.L", ci_hi = "CI.R"),
  dream_napari = list(dir = "Global_CT_Analysis",       pattern = "_dream_napari.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = "CI.L", ci_hi = "CI.R"),
  dream_quint  = list(dir = "Global_CT_Analysis",       pattern = "_dream_quint.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = "CI.L", ci_hi = "CI.R"),
  ## EXPLORATORY, not one of the manuscript's four deployed models -- see
  ## de_functions.R::run_de_dream_interaction. Only generated for
  ## region_specific/interaction scenarios (where the ground-truth effect is
  ## Cortex-specific), so absent for null/shared -- score_one/the main loop
  ## already skip silently when a model's files don't exist for a scenario.
  dream_interaction = list(dir = "Global_CT_Analysis", pattern = "_dream_interaction.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = "CI.L", ci_hi = "CI.R"),
  seurat_blind = list(dir = "Global_CT_Analysis",       pattern = "_seurat_blind.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = NA, ci_hi = NA),
  seurat_napari= list(dir = "Global_CT_Analysis",       pattern = "_seurat_napari.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = NA, ci_hi = NA),
  seurat_quint = list(dir = "Global_CT_Analysis",       pattern = "_seurat_quint.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = NA, ci_hi = NA),
  wilcoxon     = list(dir = "Global_CT_Analysis",       pattern = "_wilcoxon.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = NA, ci_hi = NA),
  deseq2_pb    = list(dir = "Pseudobulk_Validation",    pattern = "PB.csv",
                       logfc = "log2FoldChange", p = "pvalue", padj = "padj", ci_lo = NA, ci_hi = NA, se = "lfcSE"),
  ## scVI (mode="change") DE, run via run_scvi_de.py -- a rough, minimally-
  ## tuned empirical comparison point for R2 Item 2 ("missing comparison
  ## with recent alternatives"), not one of the manuscript's four deployed
  ## models. P.Value/adj.P.Val here are NOT frequentist p-values -- scVI's
  ## differential_expression() returns a posterior probability of DE
  ## (proba_de) rather than a p-value, so P.Value is a proxy
  ## (1 - proba_de) BH-corrected the same way as every other model's p-values
  ## so the existing alpha=0.05 threshold and scoring machinery can be
  ## reused unmodified; this approximation is stated explicitly wherever
  ## these numbers are reported.
  scvi_blind   = list(dir = "scVI_Analysis",            pattern = "_scvi_blind.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = NA, ci_hi = NA),
  scvi_aware   = list(dir = "scVI_Analysis",            pattern = "_scvi_aware.csv",
                       logfc = "logFC", p = "P.Value", padj = "adj.P.Val", ci_lo = NA, ci_hi = NA)
)

load_and_normalize <- function(fp, spec) {
  if (!file.exists(fp)) return(NULL)
  df <- tryCatch(read.csv(fp, check.names = FALSE), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0 || !("Gene" %in% names(df))) return(NULL)

  out <- data.frame(Gene = df$Gene)
  out$log2fc <- suppressWarnings(as.numeric(df[[spec$logfc]]))
  out$pval   <- suppressWarnings(as.numeric(df[[spec$p]]))
  out$padj   <- suppressWarnings(as.numeric(df[[spec$padj]]))

  if (!is.null(spec$ci_lo) && !is.na(spec$ci_lo) && spec$ci_lo %in% names(df)) {
    out$ci_lo <- suppressWarnings(as.numeric(df[[spec$ci_lo]]))
    out$ci_hi <- suppressWarnings(as.numeric(df[[spec$ci_hi]]))
  } else if (!is.null(spec$se) && spec$se %in% names(df)) {
    se <- suppressWarnings(as.numeric(df[[spec$se]]))
    out$ci_lo <- out$log2fc - 1.96 * se
    out$ci_hi <- out$log2fc + 1.96 * se
  } else {
    out$ci_lo <- NA_real_
    out$ci_hi <- NA_real_
  }
  out
}

score_one <- function(res, gt, alpha) {
  m <- merge(res, gt, by.x = "Gene", by.y = "gene")
  if (nrow(m) == 0) return(NULL)

  sig_adj <- !is.na(m$padj) & m$padj <= alpha
  sig_raw <- !is.na(m$pval) & m$pval <= alpha

  fdr_of <- function(sig) {
    tp <- sum(sig & m$is_true_de, na.rm = TRUE)
    fp <- sum(sig & !m$is_true_de, na.rm = TRUE)
    if (tp + fp == 0) return(NA_real_)
    fp / (tp + fp)
  }
  power_of <- function(sig) {
    n_true <- sum(m$is_true_de, na.rm = TRUE)
    if (n_true == 0) return(NA_real_)
    sum(sig & m$is_true_de, na.rm = TRUE) / n_true
  }
  ## False positive rate (Type I error rate): fraction of TRUE NULL genes
  ## called significant. Distinct from FDR (which is the fraction of
  ## SIGNIFICANT calls that are wrong) -- R1 comment 3 explicitly requests
  ## both as separate metrics. FPR is well-defined even when there are zero
  ## true-DE genes (e.g. the null scenario, where FDR degenerates to a
  ## trivial 100%-or-NA once is_true_de is correctly computed).
  fpr_of <- function(sig) {
    n_null <- sum(!m$is_true_de, na.rm = TRUE)
    if (n_null == 0) return(NA_real_)
    sum(sig & !m$is_true_de, na.rm = TRUE) / n_null
  }

  true_de <- m[m$is_true_de, , drop = FALSE]
  bias <- if (nrow(true_de) > 0) mean(true_de$log2fc - true_de$true_log2fc, na.rm = TRUE) else NA_real_
  rmse <- if (nrow(true_de) > 0) sqrt(mean((true_de$log2fc - true_de$true_log2fc)^2, na.rm = TRUE)) else NA_real_

  coverage <- NA_real_
  if (nrow(true_de) > 0 && any(!is.na(true_de$ci_lo))) {
    covered <- with(true_de, true_log2fc >= ci_lo & true_log2fc <= ci_hi)
    coverage <- mean(covered, na.rm = TRUE)
  }

  data.frame(
    fdr_adj   = fdr_of(sig_adj),
    fdr_raw   = fdr_of(sig_raw),
    fpr_adj   = fpr_of(sig_adj),
    fpr_raw   = fpr_of(sig_raw),
    power_adj = power_of(sig_adj),
    power_raw = power_of(sig_raw),
    bias      = bias,
    rmse      = rmse,
    ci_coverage = coverage,
    n_genes_scored = nrow(m),
    n_true_de      = nrow(true_de)
  )
}

## Scores every scenario/seed/model found under `results_root`, matching the
## directory layout written by run_simulation_de.R
## (results_root/<scenario>/seed_<n>/...). Returns a per-seed-per-model
## data.frame, or NULL if nothing scoreable was found (caller decides
## whether that's fatal).
score_results_root <- function(results_root, scenarios, alpha = 0.05) {
  results <- list()
  for (scenario in scenarios) {
    scen_dir <- file.path(results_root, scenario)
    if (!dir.exists(scen_dir)) { message("Missing scenario dir: ", scen_dir); next }
    seed_dirs <- list.dirs(scen_dir, recursive = FALSE)

    for (sd in seed_dirs) {
      seed <- gsub("seed_", "", basename(sd))
      gt_fp <- file.path(sd, "ground_truth.csv")
      if (!file.exists(gt_fp)) next
      gt <- read.csv(gt_fp)
      ## Recompute from true_log2fc rather than trusting the stored is_true_de
      ## column -- see CLAUDE.md for the mislabeling bug this guards against.
      gt$is_true_de <- abs(gt$true_log2fc) > 1e-8

      for (model in names(model_specs)) {
        spec <- model_specs[[model]]
        candidate_dir <- file.path(sd, spec$dir)
        if (!dir.exists(candidate_dir)) next
        files <- list.files(candidate_dir, pattern = spec$pattern, full.names = TRUE)
        if (length(files) == 0) next
        res <- load_and_normalize(files[1], spec)
        if (is.null(res)) next
        sc <- score_one(res, gt, alpha)
        if (is.null(sc)) next
        sc$scenario <- scenario
        sc$seed     <- seed
        sc$model    <- model
        results[[length(results) + 1]] <- sc
      }
    }
  }
  if (length(results) == 0) return(NULL)
  do.call(rbind, results)
}

summarize_metric <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(mean = NA, sd = NA, lo95 = NA, hi95 = NA, n = 0))
  c(mean = mean(x), sd = sd(x),
    lo95 = as.numeric(quantile(x, 0.025)), hi95 = as.numeric(quantile(x, 0.975)),
    n = length(x))
}

scoring_metrics <- c("fdr_adj", "fdr_raw", "fpr_adj", "fpr_raw", "power_adj", "power_raw",
                      "bias", "rmse", "ci_coverage")

## Aggregates a per-seed data.frame (as returned by score_results_root) into
## a per-(group_cols, model) summary with mean/SD/2.5-97.5% CI per metric.
## group_cols is typically "scenario" or c("dev", "scenario").
summarize_per_seed <- function(per_seed, group_cols = "scenario") {
  group_vals <- lapply(group_cols, function(g) unique(per_seed[[g]]))
  names(group_vals) <- group_cols
  group_grid <- expand.grid(group_vals, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)

  summary_rows <- list()
  for (i in seq_len(nrow(group_grid))) {
    for (model in names(model_specs)) {
      keep <- rep(TRUE, nrow(per_seed))
      for (g in group_cols) keep <- keep & (per_seed[[g]] == group_grid[i, g])
      keep <- keep & (per_seed$model == model)
      sub <- per_seed[keep, ]
      if (nrow(sub) == 0) next
      row <- as.list(group_grid[i, , drop = FALSE])
      row$model <- model
      for (met in scoring_metrics) {
        s <- summarize_metric(sub[[met]])
        row[[paste0(met, "_mean")]] <- s["mean"]
        row[[paste0(met, "_sd")]]   <- s["sd"]
        row[[paste0(met, "_lo95")]] <- s["lo95"]
        row[[paste0(met, "_hi95")]] <- s["hi95"]
      }
      row$n_seeds <- nrow(sub)
      summary_rows[[length(summary_rows) + 1]] <- as.data.frame(row)
    }
  }
  if (length(summary_rows) == 0) return(NULL)
  do.call(rbind, summary_rows)
}
