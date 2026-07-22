#!/usr/bin/env Rscript
# =============================================================================
# DE_Base_Analysis.R
# -----------------------------------------------------------------------------
# Baseline differential expression (no composition engineering), runnable
# straight through on an HPC node:
#
#   Rscript DE_scripts/DE_Base_Analysis.R [--config PATH] [--workers N] [--framework NAME] [--resume]
#
# --resume skips any model whose output CSV already exists and is complete, so a
# rerun only attempts what a previous run left missing (walltime kill, or a
# runner whose error was caught and logged while the job still exited 0).
#
# Runs DESeq2 (pseudobulk), the Dream LMM (blind / napari / quint), and the
# cell-level Wilcoxon / LR tests once on the selected framework's analysis-cohort
# export, contrasting composition.group_col (renamed to 'Treatment'). Framework is
# chosen by --framework / $DE_FRAMEWORK / config active_framework.
#
# Order: pass 1 runs DESeq2 + Dream over the whole cohort and every cell type,
# pass 2 then runs Wilcoxon + LR over the same. The slow single-threaded
# cell-level tests therefore never delay the sample-aware results.
# Results -> <lmm_results_dir>/base/{Pseudobulk_Validation,Local_Regional_Analysis,
# Global_CT_Analysis,SingleCell_Tests}/ (lmm_results_dir is per-framework).
# =============================================================================

# Locate + source the shared module (independent of the working directory).
.this <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
SCRIPT_DIR <- dirname(normalizePath(.this, winslash = "/", mustWork = FALSE))
source(file.path(SCRIPT_DIR, "de_common.R"))

args      <- de_parse_args(list(config = NULL, workers = NULL, framework = NULL,
                                resume = NULL, models = NULL))

# --resume: skip any model whose output CSV is already present and complete, so a
# rerun only attempts what a walltime kill or a caught error left missing.
DE_RESUME <- isTRUE(args$resume) || identical(tolower(as.character(args$resume)), "true")

# --models: which families to run, same vocabulary as LMM_all.R. Defaults to all
# four so omitting the flag preserves the original behaviour.
ALL_MODELS <- c("deseq2", "dream", "wilcox", "lr")
models <- if (!is.null(args$models)) {
  tolower(trimws(strsplit(as.character(args$models), ",")[[1]]))
} else ALL_MODELS
unknown <- setdiff(models, ALL_MODELS)
if (length(unknown))
  stop("Unknown --models entries: ", paste(unknown, collapse = ", "),
       ". Valid: ", paste(ALL_MODELS, collapse = ", "))
use_deseq2 <- "deseq2" %in% models
use_dream  <- "dream"  %in% models
use_wilcox <- "wilcox" %in% models
use_lr     <- "lr"     %in% models
cfg_all   <- de_load_config(args$config)
fw        <- de_select_framework(cfg_all, args$framework)
COMP      <- fw$composition
OUT       <- fw$outputs
group_col <- COMP$group_col
ref_level <- COMP$reference
comp_level <- COMP$comparison   # non-reference contrast group; restricts the DE to
                                # reference vs comparison (drops e.g. the FMT 'Cntrl' arm)
param     <- make_bpparam(args$workers)
message("Contrast: ", comp_level, " vs ", ref_level, " (column ", group_col, ")")
message("Models: ", paste(models, collapse = ", "))
message("Resume: ", if (DE_RESUME) "ON - existing complete outputs are skipped" else "off")

# ── Load the analysis-cohort export and drop the Olfactory bulb ───────────────
cfg_file        <- make_cfg(de_path(OUT$de_export_base))
base            <- build_seurat_from_folder(cfg_file)
seurat_obj_no_O <- subset(base, subset = napari_region != "Olfactory")

base_dir <- file.path(de_path(OUT$lmm_results_dir), "base")
dirs <- list(
  global = file.path(base_dir, "Global_CT_Analysis"),
  local  = file.path(base_dir, "Local_Regional_Analysis"),
  pb     = file.path(base_dir, "Pseudobulk_Validation"),
  sc     = file.path(base_dir, "SingleCell_Tests")
)
sapply(dirs, function(x) if (!dir.exists(x)) dir.create(x, recursive = TRUE))

# Every model family, once, on the unperturbed cohort: pseudobulk DESeq2, the
# Dream LMM trio, and the cell-level Wilcoxon / LR tests. This is the reference
# each composition-engineered seed in LMM_all.R is compared against, so it has
# to cover the same families those runs produce.
#
# The suite is split into two passes (see the driver at the bottom): every
# sample-aware model first, then every cell-level one. Wilcoxon/LR go through
# FindMarkers, which ignores the BiocParallel backend and so runs effectively
# single-threaded. Running them last means a walltime kill leaves the DESeq2 and
# Dream results complete for ALL cell types rather than only the first few.
#
# ct_col = NULL marks a cell-type-specific run and keeps the cell-type random
# effect out of the Dream formula; the whole-cohort call leaves it at its default.
run_sample_models <- function(obj, label, file_tag, ct_col = "cell_type") {
  if (use_deseq2) {
  run_pseudobulk_deseq2(obj,
    dataset_name = paste0(label, "_PB"),
    out_prefix   = paste0(label, "_", file_tag, "_PB"),
    save_dir     = dirs$pb)
  }
  if (!use_dream) return(invisible(NULL))
  run_de_blind(obj,
    dataset_name = paste0(label, "_dream_blind"),
    out_prefix   = paste0(label, "_", file_tag, "_dream_blind"),
    save_dir     = dirs$local)
  run_de_dream(obj,
    dataset_name = paste0(label, "_dream_napari"),
    out_prefix   = paste0(label, "_", file_tag, "_dream_napari"),
    save_dir     = dirs$global,
    region_col   = "napari_region",
    ct_col       = ct_col)
  run_de_dream(obj,
    dataset_name = paste0(label, "_dream_quint"),
    out_prefix   = paste0(label, "_", file_tag, "_dream_quint"),
    save_dir     = dirs$global,
    region_col   = "quint_region",
    ct_col       = ct_col)
}

run_cell_models <- function(obj, label, file_tag) {
  if (use_wilcox) {
  run_de_wilcox(obj,
    dataset_name = paste0(label, "_wilcox"),
    out_prefix   = paste0(label, "_", file_tag, "_wilcox"),
    save_dir     = dirs$sc)
  }
  if (!use_lr) return(invisible(NULL))
  run_de_lr(obj,
    dataset_name = paste0(label, "_LR_blind"),
    out_prefix   = paste0(label, "_", file_tag, "_LR_blind"),
    save_dir     = dirs$sc)
  run_de_lr(obj,
    dataset_name = paste0(label, "_LR_napari"),
    out_prefix   = paste0(label, "_", file_tag, "_LR_napari"),
    save_dir     = dirs$sc,
    region_col   = "napari_region")
  run_de_lr(obj,
    dataset_name = paste0(label, "_LR_quint"),
    out_prefix   = paste0(label, "_", file_tag, "_LR_quint"),
    save_dir     = dirs$sc,
    region_col   = "quint_region")
}

target_cell_types <- c("Astrocytes.cortex.hippocampus", "Astrocytes", "Microglia")

# `pass` is the function to apply to each cell-type subset — run_sample_models on
# the first sweep, run_cell_models on the second. The subsetting is repeated on
# the second sweep rather than holding every subset in memory between passes.
run_ct_loop <- function(data_obj, dataset_label, pass) {
  message(paste0("\n>>> STARTING HIERARCHICAL LOOP: ", dataset_label))
  for (ct in target_cell_types) {
    cell_col <- if (ct == "Astrocytes.cortex.hippocampus") "cell_type" else "ct_simple"
    obj_ct   <- data_obj[, data_obj@meta.data[[cell_col]] == ct]
    if (ncol(obj_ct) < 100) { message(paste("Skipping low cell count:", ct)); next }
    ct_clean <- gsub("[^A-Za-z0-9]", "_", ct)
    message(paste0("   Processing: ", ct))
    # already one cell type - never model cell type here (run_cell_models has no
    # cell-type term to begin with, so it takes no ct_col)
    if (identical(pass, run_sample_models)) {
      pass(obj_ct, label = paste0(dataset_label, "_", ct),
           file_tag = ct_clean, ct_col = NULL)
    } else {
      pass(obj_ct, label = paste0(dataset_label, "_", ct), file_tag = ct_clean)
    }
    rm(obj_ct); gc()
  }
}

seurat_obj_no_O <- prepare_metadata(seurat_obj_no_O, group_col, ref_level, comp_level)

# ── Pass 1: sample-aware models (pseudobulk DESeq2 + Dream LMM trio) ──────────
if (use_deseq2 || use_dream) {
  message("\n>>> PASS 1/2: DESeq2 + Dream")
  message("\n>>> RUNNING WHOLE DATASET...")
  run_sample_models(seurat_obj_no_O, label = "BASE", file_tag = "WHOLE")
  message("\n>>> RUNNING CT LOOP...")
  run_ct_loop(seurat_obj_no_O, "BASE", run_sample_models)
} else {
  message("\n>>> PASS 1/2 skipped (neither deseq2 nor dream selected)")
}

# ── Pass 2: cell-level models (Wilcoxon + LR trio) ────────────────────────────
if (use_wilcox || use_lr) {
  message("\n>>> PASS 2/2: Wilcoxon + LR")
  message("\n>>> RUNNING WHOLE DATASET...")
  run_cell_models(seurat_obj_no_O, label = "BASE", file_tag = "WHOLE")
  message("\n>>> RUNNING CT LOOP...")
  run_ct_loop(seurat_obj_no_O, "BASE", run_cell_models)
} else {
  message("\n>>> PASS 2/2 skipped (neither wilcox nor lr selected)")
}

message("\n--- ALL ANALYSES COMPLETE ---")
