#!/usr/bin/env Rscript
# Sanity check for variance_components.R on a few CosMx genes (olfactory bulb removed).
# Compares the region share from the full model (REML, as reported) with:
#   - the same model fit by ML
#   - a model with only a region random intercept
#   - one-way ANOVA eta-squared for region
# The numbers should be of the same order; ML and eta-squared are expected to be lower.

suppressPackageStartupMessages({
  library(lme4)
  library(data.table)
})

export_dir <- "~/data/AnatomicalRegions_iScience/real_data/permutation_export"
genes_to_check <- c("Camk2a", "Nrgn", "Ptpro")

meta <- read.csv(file.path(export_dir, "cell_metadata.csv"), row.names = 1,
                 check.names = FALSE, colClasses = "character")
expr <- fread(file.path(export_dir, "normalized_counts.csv"))
cell_ids <- expr[[1]]
expr[[1]] <- NULL
meta <- meta[cell_ids, ]
keep <- meta$napari_region != "Olfactory"
expr <- expr[keep]
meta <- meta[keep, ]

cell_data <- data.frame(
  log_depth = as.numeric(meta$log_depth),
  cell_type = factor(meta$cell_type),
  region    = factor(meta$napari_region),
  sample_ID = factor(meta$sample_ID),
  Treatment = factor(meta$Treatment)
)

region_share <- function(fit) {
  vc <- as.data.frame(VarCorr(fit))
  variances <- setNames(vc$vcov, vc$grp)
  X <- getME(fit, "X")
  beta <- fixef(fit)
  if ("log_depth" %in% colnames(X)) {
    variances["log_depth"] <- var(X[, "log_depth"] * beta["log_depth"])
  }
  trt_cols <- grep("^Treatment", colnames(X), value = TRUE)
  if (length(trt_cols) > 0) {
    variances["Treatment"] <- var(as.vector(X[, trt_cols, drop = FALSE] %*% beta[trt_cols]))
  }
  100 * variances["region"] / sum(variances)
}

for (gene in genes_to_check) {
  cell_data$y <- expr[[gene]]
  full_reml   <- lmer(y ~ log_depth + Treatment + (1 | cell_type) + (1 | region) + (1 | sample_ID),
                      data = cell_data, REML = TRUE)
  full_ml     <- update(full_reml, REML = FALSE)
  region_only <- lmer(y ~ (1 | region), data = cell_data, REML = TRUE)
  anova_tab   <- summary(aov(y ~ region, data = cell_data))[[1]]
  eta_sq      <- 100 * anova_tab[1, "Sum Sq"] / sum(anova_tab[, "Sum Sq"])

  cat(sprintf("%-8s full REML %5.1f%% | full ML %5.1f%% | region only %5.1f%% | eta^2 %5.1f%%\n",
              gene, region_share(full_reml), region_share(full_ml),
              region_share(region_only), eta_sq))
}
