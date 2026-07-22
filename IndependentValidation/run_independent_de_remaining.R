#!/usr/bin/env Rscript
##
## run_independent_de_remaining.R
##
## Runs ONLY the two DE steps that never finished in run_independent_de.R
## (seurat_quint, wilcoxon) -- dream_blind/napari/quint, DESeq2 PB, and
## seurat_blind/napari already completed and saved successfully for both
## g1 and g2 across two prior attempts (2h and 4h time limits, both timed
## out mid-seurat_quint with the R process confirmed still actively
## computing, not hung -- the 306-category quint_region factor makes
## FindMarkers(test.use="LR")'s per-gene GLM fit genuinely this slow at
## 20K-cell scale). Re-running the whole suite from scratch would waste
## another ~1h re-deriving already-correct output; this only re-derives
## what's missing, using the exact same unmodified functions from
## de_functions.R (run_seurat_lr, run_wilcox) as run_analysis_suite would.
##
## Usage:
##   Rscript run_independent_de_remaining.R <scenario> <seurat_input_root> <results_root>
##   scenario in {g1, g2}
##
suppressPackageStartupMessages({
  library(Seurat)
  library(BiocParallel)
})

## run_seurat_lr_parallel: same statistical test as de_functions.R's
## run_seurat_lr (FindMarkers, test.use="LR", identical latent.vars) --
## NOT a methodology change. Only the EXECUTION is different: genes are
## split into chunks and each chunk's FindMarkers(features=chunk) call
## runs in a separate worker via BiocParallel::MulticoreParam, then
## results are row-bound back together. Needed because quint_region's
## 306-level factor makes the per-gene GLM fit in run_seurat_lr genuinely
## too slow to finish serially at this scale (confirmed: 2h and 4h
## attempts both timed out with zero completions, R process actively
## computing, not hung). MulticoreParam (fork-based) chosen over
## SnowParam to avoid the SnowParam port-collision races already seen
## elsewhere in this project's array jobs.
run_seurat_lr_parallel <- function(seurat_obj, dataset_name, out_prefix, save_dir,
                                    region_col, n_workers = 8) {
  message(paste0("\n>>> [SEURAT LR / PARALLEL] STARTING FOR: ", dataset_name))
  Idents(seurat_obj) <- "Treatment"
  ref_level    <- levels(seurat_obj$Treatment)[1]
  target_level <- levels(seurat_obj$Treatment)[2]
  seurat_obj@meta.data[[region_col]] <- as.factor(seurat_obj@meta.data[[region_col]])
  covariates <- c("log_depth", region_col)
  message(paste("    Covariates:", paste(covariates, collapse = ", ")))

  all_genes <- rownames(seurat_obj)
  chunks <- split(all_genes, cut(seq_along(all_genes), n_workers, labels = FALSE))
  message(sprintf("    Splitting %d genes into %d chunks across %d workers",
                   length(all_genes), length(chunks), n_workers))

  param <- MulticoreParam(workers = n_workers, progressbar = FALSE)

  tryCatch({
    chunk_results <- bplapply(chunks, function(gene_chunk) {
      FindMarkers(seurat_obj,
                  ident.1         = target_level,
                  ident.2         = ref_level,
                  test.use        = "LR",
                  latent.vars     = covariates,
                  features        = gene_chunk,
                  logfc.threshold = 0,
                  min.pct         = 0,
                  verbose         = FALSE)
    }, BPPARAM = param)

    de_res <- do.call(rbind, chunk_results)
    de_res$Gene <- rownames(de_res)
    colnames(de_res)[colnames(de_res) == "pct.1"]      <- "pct_target"
    colnames(de_res)[colnames(de_res) == "pct.2"]      <- "pct_ref"
    colnames(de_res)[colnames(de_res) == "avg_log2FC"] <- "logFC"
    colnames(de_res)[colnames(de_res) == "p_val_adj"]  <- "adj.P.Val"
    colnames(de_res)[colnames(de_res) == "p_val"]      <- "P.Value"
    write.csv(de_res, file = file.path(save_dir, paste0(out_prefix, ".csv")), row.names = FALSE)
    message(paste("    -> Saved:", file.path(save_dir, paste0(out_prefix, ".csv"))))
  }, error = function(e) message(paste("    !! SEURAT PARALLEL FAILED:", e$message)))
}

args             <- commandArgs(trailingOnly = TRUE)
scenario         <- ifelse(length(args) >= 1, args[1], "g1")
seurat_input_root<- ifelse(length(args) >= 2, args[2], "seurat_input")
results_root     <- ifelse(length(args) >= 3, args[3], "de_results")

stopifnot(scenario %in% c("g1", "g2"))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "..", "DE_scripts", "de_functions.R"))

data_dir <- file.path(seurat_input_root, scenario)
if (!dir.exists(data_dir)) stop(paste("Seurat input not found:", data_dir))

cfg <- make_cfg(data_dir)
seu <- build_seurat_from_folder(cfg)
seu <- prepare_metadata(seu)

out_dir <- file.path(results_root, scenario)
global_dir <- file.path(out_dir, "Global_CT_Analysis")
if (!dir.exists(global_dir)) dir.create(global_dir, recursive = TRUE)

label <- "INDEP"

message(sprintf(">>> Running REMAINING steps (seurat_quint, wilcoxon) for scenario=%s (%d cells, %d genes)",
                 scenario, ncol(seu), nrow(seu)))

run_seurat_lr_parallel(seu,
  dataset_name = paste0(label, "_seurat_quint"),
  out_prefix   = paste0(label, "_", scenario, "_seurat_quint"),
  save_dir     = global_dir,
  region_col   = "quint_region",
  n_workers    = 8)

run_wilcox(seu,
  dataset_name = paste0(label, "_wilcoxon"),
  out_prefix   = paste0(label, "_", scenario, "_wilcoxon"),
  save_dir     = global_dir)

message(sprintf(">>> DONE (remaining steps) scenario=%s -> %s", scenario, out_dir))
