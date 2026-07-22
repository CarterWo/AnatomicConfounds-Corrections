#!/usr/bin/env Rscript
##
## run_independent_de_multi_remaining.R
##
## Resumes ONLY seurat_quint + wilcoxon for a (dev, scenario) pair whose
## run_independent_de_multi.R / run_independent_de_devsweep.sbatch run
## timed out at 4h mid-seurat_quint. dream_blind/napari/quint, PB DESeq2,
## and seurat_blind/napari already saved successfully before the timeout
## (verified via find on the results dir) -- NOT re-run here, exactly the
## same "only redo what didn't finish" principle used for the single-
## animal workstream's run_independent_de_remaining.R.
##
## Usage:
##   Rscript run_independent_de_multi_remaining.R <scenario> <seurat_input_root> <results_root>
##   scenario in {g1, g2}
##
suppressPackageStartupMessages({
  library(Seurat)
  library(BiocParallel)
})

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

args              <- commandArgs(trailingOnly = TRUE)
scenario          <- ifelse(length(args) >= 1, args[1], "g1")
seurat_input_root <- ifelse(length(args) >= 2, args[2], "seurat_input")
results_root      <- ifelse(length(args) >= 3, args[3], "de_results")
n_workers         <- ifelse(length(args) >= 4, as.integer(args[4]), 12)

stopifnot(scenario %in% c("g1", "g2"))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "..", "DE_scripts", "de_functions.R"))

param <- SnowParam(workers = 4, type = "SOCK", progressbar = FALSE, exportglobals = FALSE)

data_dir <- file.path(seurat_input_root, scenario)
if (!dir.exists(data_dir)) stop(paste("Seurat input not found:", data_dir))

cfg <- make_cfg(data_dir)
seu <- build_seurat_from_folder(cfg)
seu <- prepare_metadata(seu)

out_dir <- file.path(results_root, scenario)
dirs <- list(global = file.path(out_dir, "Global_CT_Analysis"))
sapply(dirs, function(x) if (!dir.exists(x)) dir.create(x, recursive = TRUE))

label <- "INDEP_MULTI"

message(sprintf(">>> RESUMING (seurat_quint + wilcoxon only) scenario=%s (%d cells, %d genes, %d animals, %d workers)",
                 scenario, ncol(seu), nrow(seu), length(unique(seu$sample_ID)), n_workers))

run_seurat_lr_parallel(seu,
  dataset_name = paste0(label, "_seurat_quint"),
  out_prefix   = paste0(label, "_", scenario, "_seurat_quint"),
  save_dir     = dirs$global,
  region_col   = "quint_region",
  n_workers    = n_workers)
run_wilcox(seu,
  dataset_name = paste0(label, "_wilcoxon"),
  out_prefix   = paste0(label, "_", scenario, "_wilcoxon"),
  save_dir     = dirs$global)

message(sprintf(">>> DONE (remaining) scenario=%s -> %s", scenario, out_dir))
