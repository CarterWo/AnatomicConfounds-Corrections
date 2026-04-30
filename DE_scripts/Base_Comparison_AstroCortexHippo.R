#!/usr/bin/env Rscript
# =============================================================================
# Base_Comparison_AstroCortexHippo.R
#
# Re-run of the original Base_Comparison_2.ipynb DE pipeline, restricted to
# the Astrocytes.cortex.hippocampus subgroup ONLY.  Identical formulas,
# identical helpers, identical output naming; just the WHOLE block and the
# Astrocytes / Microglia loops are removed.
# =============================================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(variancePartition)
  library(BiocParallel)
  library(limma)
  library(ggplot2)
  library(dplyr)
  library(DESeq2)
  library(edgeR)
})

set.seed(42)

# -----------------------------------------------------------------------------
# 0. PATHS & DIRECTORIES
# -----------------------------------------------------------------------------
data_root <- "/mnt/pixstor/data/cdw5fz/AtomX_files/Labeling_manuscript/csvs_20g_100t/base"
base_dir  <- "/mnt/pixstor/data/cdw5fz/AtomX_files/Labeling_manuscript/LMM_base_20g_100t_AstroCortexHippo"

dirs <- list(
  global = file.path(base_dir, "Global_CT_Analysis"),
  local  = file.path(base_dir, "Local_Regional_Analysis"),
  pb     = file.path(base_dir, "Pseudobulk_Validation")
)
invisible(lapply(dirs, function(x) if (!dir.exists(x)) dir.create(x, recursive = TRUE)))

# -----------------------------------------------------------------------------
# 1. LOADER (verbatim from notebook)
# -----------------------------------------------------------------------------
make_cfg <- function(root_dir,
                     raw_counts_name    = "raw_counts.csv",
                     logged_counts_name = "normalized_counts.csv",
                     features_name      = "features_counts.csv",
                     metadata_name      = "cell_metadata.csv",
                     coords_name        = "coords_xy.csv") {
  list(
    root_dir      = root_dir,
    raw_counts    = file.path(root_dir, raw_counts_name),
    logged_counts = file.path(root_dir, logged_counts_name),
    features      = file.path(root_dir, features_name),
    metadata      = file.path(root_dir, metadata_name),
    coords        = file.path(root_dir, coords_name)
  )
}

build_seurat_from_folder <- function(cfg, assay_name = "RNA") {
  message(paste0("Loading from: ", cfg$root_dir))

  counts        <- read.csv(cfg$raw_counts,    row.names = 1, check.names = FALSE)
  logged_counts <- read.csv(cfg$logged_counts, row.names = 1, check.names = FALSE)
  features      <- read.csv(cfg$features,      row.names = 1, check.names = FALSE)
  meta          <- read.csv(cfg$metadata,      row.names = 1, check.names = FALSE)
  coords        <- read.csv(cfg$coords,        row.names = 1, check.names = FALSE)

  if (ncol(counts) != nrow(features)) {
    stop(sprintf("Mismatch: counts has %d genes, features has %d",
                 ncol(counts), nrow(features)))
  }

  colnames(counts)        <- rownames(features)
  colnames(logged_counts) <- rownames(features)

  common_cells <- Reduce(intersect, list(
    rownames(counts), rownames(meta), rownames(coords), rownames(logged_counts)
  ))
  if (length(common_cells) == 0) stop("No common Cell IDs found.")
  message(sprintf("  Matched %d cells across all files.", length(common_cells)))

  counts        <- counts[common_cells, ]
  logged_counts <- logged_counts[common_cells, ]
  meta          <- meta[common_cells, ]
  coords        <- coords[common_cells, ]

  seu <- CreateSeuratObject(counts = t(counts), meta.data = meta, assay = assay_name)
  seu <- SetAssayData(seu, layer = "data", new.data = as.matrix(t(logged_counts)))
  seu <- AddMetaData(seu, metadata = coords)
  seu
}

# -----------------------------------------------------------------------------
# 2. METADATA PREP (verbatim)
# -----------------------------------------------------------------------------
prepare_metadata <- function(seurat_obj) {
  names(seurat_obj@meta.data)[names(seurat_obj@meta.data) == "FMT"] <- "Treatment"
  seurat_obj$log_depth <- as.numeric(scale(log10(seurat_obj$nFeature_RNA + 1)))
  seurat_obj <- subset(seurat_obj, subset = Treatment != "Cntrl")
  seurat_obj$Treatment <- as.factor(seurat_obj$Treatment)
  if ("Healthy_FMT" %in% levels(seurat_obj$Treatment)) {
    seurat_obj$Treatment <- relevel(seurat_obj$Treatment, ref = "Healthy_FMT")
  }
  seurat_obj
}

# -----------------------------------------------------------------------------
# 3. DE FUNCTIONS (verbatim from notebook — only file naming preserved)
# -----------------------------------------------------------------------------
run_de_napari <- function(seurat_obj, dataset_name, out_prefix, save_dir) {
  message(paste0("\n>>> [GLOBAL] STARTING DREAM FOR: ", dataset_name))

  num_cts <- length(unique(seurat_obj$cell_type))
  if (num_cts > 1) {
    form_de <- ~ Treatment + log_depth + (1 | napari_region) + (1 | cell_type) + (1 | sample_ID)
  } else {
    form_de <- ~ Treatment + log_depth + (1 | napari_region) + (1 | sample_ID)
  }
  message(paste("    Formula:", deparse(form_de)))

  counts <- as.matrix(GetAssayData(seurat_obj, layer = "counts"))
  info   <- seurat_obj@meta.data
  keep_genes <- rowSums(counts) > 0
  counts <- counts[keep_genes, ]

  dge  <- DGEList(counts); dge <- calcNormFactors(dge)
  message("    Calculating voom weights...")
  vobj <- voomWithDreamWeights(dge, form_de, info, BPPARAM = param)
  rm(dge); gc()
  fit  <- dream(vobj, form_de, info, BPPARAM = param)
  fit  <- eBayes(fit)

  target_coef <- grep("Treatment", colnames(fit$coefficients), value = TRUE)
  target_coef <- target_coef[length(target_coef)]
  if (length(target_coef) > 0) {
    de_res <- topTable(fit, coef = target_coef, number = Inf, sort.by = "P", confint = TRUE)
    de_res$Gene  <- rownames(de_res)
    de_res$pct_1 <- rowMeans(counts[rownames(de_res), , drop = FALSE] > 0)
    outfile <- file.path(save_dir, paste0("Jan_26", out_prefix, ".csv"))
    write.csv(de_res, file = outfile, row.names = FALSE)
    message(paste("    -> Saved:", outfile))
  }
}

run_de_quint <- function(seurat_obj, dataset_name, out_prefix, save_dir) {
  message(paste0("\n>>> [GLOBAL] STARTING DREAM FOR: ", dataset_name))

  num_cts <- length(unique(seurat_obj$cell_type))
  if (num_cts > 1) {
    form_de <- ~ Treatment + log_depth + (1 | quint_region) + (1 | cell_type) + (1 | sample_ID)
  } else {
    form_de <- ~ Treatment + log_depth + (1 | quint_region) + (1 | sample_ID)
  }
  message(paste("    Formula:", deparse(form_de)))

  counts <- as.matrix(GetAssayData(seurat_obj, layer = "counts"))
  info   <- seurat_obj@meta.data
  keep_genes <- rowSums(counts) > 0
  counts <- counts[keep_genes, ]

  dge  <- DGEList(counts); dge <- calcNormFactors(dge)
  message("    Calculating voom weights...")
  vobj <- voomWithDreamWeights(dge, form_de, info, BPPARAM = param)
  rm(dge); gc()
  fit  <- dream(vobj, form_de, info, BPPARAM = param)
  fit  <- eBayes(fit)

  target_coef <- grep("Treatment", colnames(fit$coefficients), value = TRUE)
  target_coef <- target_coef[length(target_coef)]
  if (length(target_coef) > 0) {
    de_res <- topTable(fit, coef = target_coef, number = Inf, sort.by = "P", confint = TRUE)
    de_res$Gene  <- rownames(de_res)
    de_res$pct_1 <- rowMeans(counts[rownames(de_res), , drop = FALSE] > 0)
    outfile <- file.path(save_dir, paste0("Jan_26", out_prefix, ".csv"))
    write.csv(de_res, file = outfile, row.names = FALSE)
    message(paste("    -> Saved:", outfile))
  }
}

run_de_blind <- function(seurat_obj, dataset_name, out_prefix, save_dir) {
  form_de <- ~ Treatment + log_depth + (1 | sample_ID)
  message(paste0("    >>> [LOCAL] Fitting Dream for: ", dataset_name))

  geneExpr <- as.matrix(GetAssayData(seurat_obj, layer = "data"))
  counts   <- as.matrix(GetAssayData(seurat_obj, layer = "counts"))
  info     <- seurat_obj@meta.data

  tryCatch({
    keep_genes <- rowSums(counts) > 0
    counts_sub <- counts[keep_genes, ]
    dge <- DGEList(counts_sub); dge <- calcNormFactors(dge)
    vobj <- voomWithDreamWeights(dge, form_de, info, BPPARAM = param)
    rm(dge); gc()
    fit  <- dream(vobj, form_de, info, BPPARAM = param)
    fit  <- eBayes(fit)

    target_coef <- grep("Treatment", colnames(fit$coefficients), value = TRUE)
    target_coef <- target_coef[length(target_coef)]
    if (length(target_coef) > 0) {
      de_res <- topTable(fit, coef = target_coef, number = Inf, sort.by = "P", confint = TRUE)
      de_res$Gene <- rownames(de_res)

      ref_level    <- levels(seurat_obj$Treatment)[1]
      target_level <- levels(seurat_obj$Treatment)[2]
      cells_ref    <- which(seurat_obj$Treatment == ref_level)
      cells_target <- which(seurat_obj$Treatment == target_level)

      de_res$pct_ref    <- rowMeans(geneExpr[rownames(de_res), cells_ref,    drop = FALSE] > 0)
      de_res$pct_target <- rowMeans(geneExpr[rownames(de_res), cells_target, drop = FALSE] > 0)
      de_res$avg_ref    <- rowMeans(geneExpr[rownames(de_res), cells_ref,    drop = FALSE])
      de_res$avg_target <- rowMeans(geneExpr[rownames(de_res), cells_target, drop = FALSE])

      outfile <- file.path(save_dir, paste0("Jan_26", out_prefix, ".csv"))
      write.csv(de_res, file = outfile, row.names = FALSE)
      message(paste("    -> Saved:", outfile))
    }
  }, error = function(e) {
    message(paste("    !! SKIPPING blind:", e$message))
  })
}

run_pseudobulk_deseq2 <- function(seurat_obj, dataset_name, out_prefix, save_dir) {
  message(paste0("    >>> [PB-VAL] Running Pseudo-bulk DESeq2: ", dataset_name))

  cts <- AggregateExpression(seurat_obj, group.by = "sample_ID",
                             assays = "RNA", slot = "counts")$RNA
  colnames(cts) <- gsub("^g", "", colnames(cts))
  colnames(cts) <- gsub("-", "_", colnames(cts))

  colData <- seurat_obj@meta.data %>%
    dplyr::select(sample_ID, Treatment) %>%
    dplyr::distinct(sample_ID, .keep_all = TRUE)
  colData <- colData[match(colnames(cts), colData$sample_ID), ]
  rownames(colData) <- colData$sample_ID

  tryCatch({
    dds <- DESeqDataSetFromMatrix(countData = cts, colData = colData, design = ~ Treatment)
    keep <- rowSums(counts(dds)) >= 10
    dds  <- dds[keep, ]
    dds  <- DESeq(dds, quiet = TRUE)
    res  <- results(dds)
    res_df <- as.data.frame(res)
    res_df$Gene <- rownames(res_df)
    outfile <- file.path(save_dir, paste0("Jan_26_DESEQ2", out_prefix, ".csv"))
    write.csv(res_df, file = outfile, row.names = FALSE)
    message(paste("    -> Saved PB Validation:", outfile))
  }, error = function(e) {
    message(paste("    !! PB FAILED:", e$message))
  })
}

# -----------------------------------------------------------------------------
# 4. ANALYSIS SUITE (mirrors notebook)
# -----------------------------------------------------------------------------
run_analysis_suite <- function(obj, label, file_tag) {
  run_pseudobulk_deseq2(obj,
                        dataset_name = paste0(label, "_pb"),
                        out_prefix   = paste0(label, "_", file_tag, "_PB"),
                        save_dir     = dirs$pb)

  run_de_blind(obj,
               dataset_name = paste0(label, "_dream_blind"),
               out_prefix   = paste0(label, "_", file_tag, "_dream_blind"),
               save_dir     = dirs$local)

  run_de_napari(obj,
                dataset_name = paste0(label, "_dream_napari"),
                out_prefix   = paste0(label, "_", file_tag, "_dream_napari"),
                save_dir     = dirs$global)

  run_de_quint(obj,
               dataset_name = paste0(label, "_dream_quint"),
               out_prefix   = paste0(label, "_", file_tag, "_dream_quint"),
               save_dir     = dirs$global)
}

# -----------------------------------------------------------------------------
# 5. EXECUTION
# -----------------------------------------------------------------------------
param <- SnowParam(workers = 5, type = "SOCK",
                   progressbar = TRUE, exportglobals = FALSE)

message("\n>>> [1] Loading base dataset...")
cfg  <- make_cfg(data_root)
base <- build_seurat_from_folder(cfg)

message("\n>>> [2] Dropping Olfactory napari_region...")
base <- subset(base, subset = napari_region != "Olfactory")
message(paste0("    napari_region kept: ",
               paste(unique(base$napari_region), collapse = ", ")))

message("\n>>> [3] prepare_metadata: drop Cntrl, set Treatment factor, log_depth...")
base <- prepare_metadata(base)

# --- Subset to Astrocytes.cortex.hippocampus ONLY ---
ct       <- "Astrocytes.cortex.hippocampus"
cell_col <- "cell_type"
message(sprintf("\n>>> [4] Subsetting to %s using column %s ...", ct, cell_col))
obj_ct <- base[, base@meta.data[[cell_col]] == ct]
rm(base); gc()

n_cells <- ncol(obj_ct)
message(sprintf("    Cells: %d", n_cells))
if (n_cells < 100) stop("Too few cells for DE.")

ct_clean <- gsub("[^A-Za-z0-9]", "_", ct)
message("\n>>> [5] Running analysis suite...")
run_analysis_suite(obj_ct,
                   label    = paste0("BASE_", ct),
                   file_tag = ct_clean)

message("\n--- DONE: Astrocytes.cortex.hippocampus only ---")
message(paste("Results in:", base_dir))
