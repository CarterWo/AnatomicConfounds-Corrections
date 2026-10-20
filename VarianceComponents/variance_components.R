#!/usr/bin/env Rscript
# Per-gene variance components for the real datasets (CosMx and Zhuang-ABCA MERFISH).
#
# For each gene we fit a linear mixed model with the same covariates as the Dream
# models and report how much of the expression variance each term accounts for:
#   random intercepts: cell_type, region, sample_ID
#   fixed effects:     log_depth (and Treatment for CosMx), measured as the variance
#                      of each term's fitted contribution
#   residual
# Each gene's components are divided by their sum, so they add up to 1.
#
# We compute this directly from lme4 rather than with
# variancePartition::fitExtractVarPartModel, which returned zero region variance
# for every gene with the lme4 version installed on our cluster.
#
# Usage:
#   Rscript variance_components.R <export_dir> <out_csv> <region_col> \
#       <include_treatment: 0/1> <n_cores> [region_to_drop]
#
# region_to_drop removes cells with that napari_region before fitting. For CosMx we
# drop "Olfactory", as in Pre_DE_processing.ipynb, because the olfactory bulb is
# only present in one treatment group.

suppressPackageStartupMessages({
  library(lme4)
  library(data.table)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
export_dir     <- args[1]
out_csv        <- args[2]
region_col     <- args[3]
with_treatment <- args[4] == "1"
n_cores        <- as.integer(args[5])
region_to_drop <- if (length(args) >= 6) args[6] else NA

# ---- load data ----
meta <- read.csv(file.path(export_dir, "cell_metadata.csv"), row.names = 1,
                 check.names = FALSE, colClasses = "character")
expr <- fread(file.path(export_dir, "normalized_counts.csv"))  # cells x genes
cell_ids <- expr[[1]]
expr[[1]] <- NULL
stopifnot(all(cell_ids %in% rownames(meta)))
meta <- meta[cell_ids, ]

if (!is.na(region_to_drop)) {
  keep <- meta$napari_region != region_to_drop
  message("Dropping ", sum(!keep), " cells with napari_region == ", region_to_drop)
  expr <- expr[keep]
  meta <- meta[keep, ]
}

cell_data <- data.frame(
  log_depth = as.numeric(meta$log_depth),
  cell_type = factor(meta$cell_type),
  region    = factor(meta[[region_col]]),
  sample_ID = factor(meta$sample_ID),
  Treatment = factor(meta$Treatment)
)

# skip genes detected in <1% of cells or with no variance
genes <- names(expr)
detected <- vapply(genes, function(g) mean(expr[[g]] > 0) >= 0.01 && var(expr[[g]]) > 0,
                   logical(1))
genes <- genes[detected]

message(sprintf("%d cells, %d genes, %d %s levels, %d samples, %d cell types",
                nrow(cell_data), length(genes), nlevels(cell_data$region), region_col,
                nlevels(cell_data$sample_ID), nlevels(cell_data$cell_type)))

# ---- model ----
fixed_terms <- if (with_treatment) "log_depth + Treatment" else "log_depth"
model_formula <- as.formula(paste("y ~", fixed_terms,
                                  "+ (1 | cell_type) + (1 | region) + (1 | sample_ID)"))

fit_gene <- function(gene) {
  dat <- cell_data
  dat$y <- expr[[gene]]
  fit <- tryCatch(
    suppressMessages(suppressWarnings(
      lmer(model_formula, data = dat, REML = TRUE,
           control = lmerControl(calc.derivs = FALSE)))),
    error = function(e) NULL)
  if (is.null(fit)) return(NULL)

  # random-effect and residual variances
  vc <- as.data.frame(VarCorr(fit))
  variances <- setNames(vc$vcov, ifelse(vc$grp == "Residual", "Residuals", vc$grp))

  # fixed effects: variance of each term's fitted contribution
  X <- getME(fit, "X")
  beta <- fixef(fit)
  variances["log_depth"] <- var(X[, "log_depth"] * beta["log_depth"])
  if (with_treatment) {
    trt_cols <- grep("^Treatment", colnames(X), value = TRUE)
    variances["Treatment"] <- var(as.vector(X[, trt_cols, drop = FALSE] %*% beta[trt_cols]))
  }

  data.frame(gene = gene, t(variances / sum(variances)),
             singular = isSingular(fit), check.names = FALSE)
}

results <- rbindlist(mclapply(genes, fit_gene, mc.cores = n_cores), fill = TRUE)
message(sprintf("Fit %d of %d genes (%d singular fits)",
                nrow(results), length(genes), sum(results$singular)))
fwrite(results, out_csv)

# ---- summary (% of total variance across genes) ----
components <- setdiff(names(results), c("gene", "singular"))
summary_table <- sapply(components, function(comp) {
  x <- results[[comp]]
  c(median = median(x), q25 = unname(quantile(x, 0.25)),
    q75 = unname(quantile(x, 0.75)), mean = mean(x))
})
print(round(100 * summary_table, 2))
