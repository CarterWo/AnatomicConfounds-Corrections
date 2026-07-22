#!/usr/bin/env Rscript
##
## run_independent_de_multi.R
##
## Multi-animal successor to run_independent_de.R -- runs the manuscript's
## unmodified DE functions (de_functions.R) on the 4-animal-pooled
## composition-engineered Zhuang-ABCA-1/2/3/4 dataset. Same principle as
## before: deployed models tested completely as-is, not reimplemented.
##
## Learned from the single-animal run (documented in CLAUDE.md): calling
## run_analysis_suite() serially hit the seurat_quint bottleneck (~306
## quint_region categories -> genuinely slow per-gene GLM fit) and timed
## out twice (2h, then 4h) before a parallelized version was written as a
## SEPARATE follow-up script. Building that parallelization in from the
## start here rather than repeating the same two-timeout cycle: every
## model is the exact same unmodified de_functions.R function EXCEPT
## seurat_quint, which uses the already-validated run_seurat_lr_parallel
## (identical FindMarkers/test.use="LR" call, only the gene-loop execution
## is chunked across BiocParallel::MulticoreParam workers).
##
## `Treatment` here is `pseudo_group` (A/B), a random split of animals with
## no real biological difference by construction -- see CLAUDE.md.
##
## Usage:
##   Rscript run_independent_de_multi.R <scenario> <seurat_input_root> <results_root>
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

stopifnot(scenario %in% c("g1", "g2"))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "..", "DE_scripts", "de_functions.R"))

param <- SnowParam(workers = 4, type = "SOCK", progressbar = FALSE, exportglobals = FALSE)

data_dir <- file.path(seurat_input_root, scenario)
if (!dir.exists(data_dir)) stop(paste("Seurat input not found:", data_dir,
                                       "- run build_seurat_input_multi.py first."))

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

label <- "INDEP_MULTI"

message(sprintf(">>> Running analysis suite for scenario=%s (%d cells, %d genes, %d animals)",
                 scenario, ncol(seu), nrow(seu), length(unique(seu$sample_ID))))
message(sprintf(">>> Treatment level counts: %s",
                 paste(capture.output(print(table(seu$Treatment))), collapse = " | ")))
message(sprintf(">>> Animals (sample_ID) x Treatment: %s",
                 paste(capture.output(print(table(seu$sample_ID, seu$Treatment))), collapse = " | ")))

run_pseudobulk_deseq2(seu,
  dataset_name = paste0(label, "_PB"),
  out_prefix   = paste0(label, "_", scenario, "_PB"),
  save_dir     = dirs$pb)
run_de_blind(seu,
  dataset_name = paste0(label, "_dream_blind"),
  out_prefix   = paste0(label, "_", scenario, "_dream_blind"),
  save_dir     = dirs$local)
run_de_dream(seu,
  dataset_name = paste0(label, "_dream_napari"),
  out_prefix   = paste0(label, "_", scenario, "_dream_napari"),
  save_dir     = dirs$global,
  region_col   = "napari_region")
run_de_dream(seu,
  dataset_name = paste0(label, "_dream_quint"),
  out_prefix   = paste0(label, "_", scenario, "_dream_quint"),
  save_dir     = dirs$global,
  region_col   = "quint_region")
run_seurat_lr(seu,
  dataset_name = paste0(label, "_seurat_blind"),
  out_prefix   = paste0(label, "_", scenario, "_seurat_blind"),
  save_dir     = dirs$global,
  region_col   = NULL)
run_seurat_lr(seu,
  dataset_name = paste0(label, "_seurat_napari"),
  out_prefix   = paste0(label, "_", scenario, "_seurat_napari"),
  save_dir     = dirs$global,
  region_col   = "napari_region")
## seurat_quint uses the parallelized executor from the start (see header) --
## the serial run_seurat_lr call already timed out twice on the single-
## animal quint_region factor (~306 levels) in the earlier workstream.
run_seurat_lr_parallel(seu,
  dataset_name = paste0(label, "_seurat_quint"),
  out_prefix   = paste0(label, "_", scenario, "_seurat_quint"),
  save_dir     = dirs$global,
  region_col   = "quint_region",
  n_workers    = 12)
run_wilcox(seu,
  dataset_name = paste0(label, "_wilcoxon"),
  out_prefix   = paste0(label, "_", scenario, "_wilcoxon"),
  save_dir     = dirs$global)

message(sprintf(">>> DONE scenario=%s -> %s", scenario, out_dir))
