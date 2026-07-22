#!/usr/bin/env Rscript
# =============================================================================
# LMM_all.R
# -----------------------------------------------------------------------------
# Composition-engineered differential expression across iterations, runnable
# straight through on an HPC node OR split across array jobs.
#
#   Rscript DE_scripts/LMM_all.R                         # all devs, all seeds
#   Rscript DE_scripts/LMM_all.R --framework fmt         # pick the dataset/contrast
#   Rscript DE_scripts/LMM_all.R --devs 1 --seeds 0-49   # one array-task slice
#   Rscript DE_scripts/LMM_all.R --workers 16 --config /path/config.yaml
#
# For each requested deviation magnitude and iteration (seed), reconstructs the
# cortex-up / cortex-down subsets from the G1/G2 masks and runs:
#   DESeq2      -> every requested seed
#   Dream LMM   -> only seeds < --dream-seeds (default composition.dream_iterations)
#   Wilcoxon/LR -> only seeds < --sc-seeds    (default composition.sc_iterations)
# Results -> <lmm_results_dir>/dev_<dev>/seed_<seed>/{...}/  (per-iteration
# subfolders, so array tasks never overwrite each other).
#
# --models selects which model families run at all, so separate array jobs can
# cover disjoint slices of the same seed space without racing to write the same
# CSVs. The seed thresholds above still apply within whatever is selected:
#   --models deseq2                  seeds 3-99  (deseq_seeds.sbatch)
#   --models wilcox,lr               seeds 3-9   (sc_tests.sbatch)
#   --models deseq2,dream,wilcox,lr  seeds 0-2   (dream_0-2.sbatch, the default)
#
# Options (defaults come from the selected framework's config.yaml `composition`):
#   --config PATH     config.yaml (else DE_CONFIG env / auto-discovered)
#   --framework NAME  which frameworks.<name> block to use (else $DE_FRAMEWORK /
#                     config active_framework). Selects dataset, contrast, and
#                     the labelled de_export_base / lmm_results_dir paths.
#   --workers N       BiocParallel workers (else $SLURM_CPUS_PER_TASK, else 5)
#   --devs   LIST     deviation magnitudes, e.g. "1.5,1,0.5,0.25" or "1"
#   --seeds  RANGE    iterations, e.g. "0-99", "0:49", or "3,7,10"
#   --dream-seeds N   number of leading seeds the Dream LMM runs per deviation
#   --sc-seeds N      number of leading seeds Wilcoxon/LR run per deviation
#   --models LIST     comma list from {deseq2,dream,wilcox,lr}; default all four
# =============================================================================

# Locate + source the shared module (independent of the working directory).
.this <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
SCRIPT_DIR <- dirname(normalizePath(.this, winslash = "/", mustWork = FALSE))
source(file.path(SCRIPT_DIR, "de_common.R"))

args      <- de_parse_args(list(config = NULL, workers = NULL, framework = NULL,
                                devs = NULL, seeds = NULL, `dream-seeds` = NULL,
                                `sc-seeds` = NULL, models = NULL, resume = NULL))

# --resume: skip any model whose output CSV is already present and complete, so a
# rerun only attempts what a walltime kill or a caught error left missing.
DE_RESUME <- isTRUE(args$resume) || identical(tolower(as.character(args$resume)), "true")
cfg_all   <- de_load_config(args$config)
fw        <- de_select_framework(cfg_all, args$framework)
COMP      <- fw$composition
OUT       <- fw$outputs
group_col <- COMP$group_col
ref_level <- COMP$reference
comp_level <- COMP$comparison   # non-reference contrast group; restricts the DE to
                                # reference vs comparison (drops e.g. the FMT 'Cntrl' arm)

devs         <- if (!is.null(args$devs))  de_parse_num_list(args$devs)  else unlist(COMP$deviations)
seeds        <- if (!is.null(args$seeds)) de_parse_int_range(args$seeds) else seq.int(0, COMP$deseq2_iterations - 1)
n_iter_dream <- if (!is.null(args$`dream-seeds`)) as.integer(args$`dream-seeds`) else COMP$dream_iterations
n_iter_sc    <- if (!is.null(args$`sc-seeds`))    as.integer(args$`sc-seeds`)    else (COMP$sc_iterations %||% 10L)
param        <- make_bpparam(args$workers)

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

base_dir <- de_path(OUT$lmm_results_dir)

message("Framework: ", fw$name)
message("Contrast: ", comp_level, " vs ", ref_level, " (column ", group_col, ")")
message("Deviations: ", paste(devs, collapse = ", "))
message("Models: ", paste(models, collapse = ", "))
message("Resume: ", if (DE_RESUME) "ON - existing complete outputs are skipped" else "off")
message("Seeds: ", paste(range(seeds), collapse = ".."), " (n=", length(seeds), ")",
        " | Dream LMM on seeds < ", n_iter_dream,
        " | Wilcoxon/LR on seeds < ", n_iter_sc)

# ── Load the single full-cohort export ────────────────────────────────────────
message("Loading full cohort")
full <- build_seurat_from_folder(make_cfg(de_path(OUT$de_export_base)))

# Each family is gated independently so disjoint array jobs can split the work:
# DESeq2 on every requested seed, Dream and the cell-level tests only on the
# leading seeds their thresholds allow.
#
# ct_col = NULL marks a cell-type-specific run and keeps the cell-type random
# effect out of the Dream formula; the whole-cohort calls leave it at its default.
run_iter_suite <- function(obj, label, file_tag, dirs, run_dream, run_wilcox, run_lr,
                           ct_col = "cell_type") {
  if (use_deseq2) {
    run_pseudobulk_deseq2(obj,
      dataset_name = paste0(label, "_PB"),
      out_prefix   = paste0(label, "_", file_tag, "_PB"),
      save_dir     = dirs$pb)
  }
  if (run_dream) {
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
  if (run_wilcox) {
    run_de_wilcox(obj,
      dataset_name = paste0(label, "_wilcox"),
      out_prefix   = paste0(label, "_", file_tag, "_wilcox"),
      save_dir     = dirs$sc)
  }
  if (run_lr) {
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
}

target_cell_types <- c("Astrocytes", "Microglia")

run_ct_loop <- function(data_obj, dataset_label, dirs, run_dream, run_wilcox, run_lr) {
  for (ct in target_cell_types) {
    cell_col <- if (ct == "Astrocytes.cortex.hippocampus") "cell_type" else "ct_simple"
    obj_ct   <- data_obj[, data_obj@meta.data[[cell_col]] == ct]
    if (ncol(obj_ct) < 100) { message(paste("Skipping low cell count:", ct)); next }
    ct_clean <- gsub("[^A-Za-z0-9]", "_", ct)
    run_iter_suite(obj_ct,
      label      = paste0(dataset_label, "_", ct),
      file_tag   = ct_clean,
      dirs       = dirs,
      run_dream  = run_dream,
      run_wilcox = run_wilcox,
      run_lr     = run_lr,
      ct_col     = NULL)   # already one cell type - never model cell type here
    rm(obj_ct); gc()
  }
}

# ── Main loop: deviation magnitude x iteration x {UP, DOWN} ───────────────────
for (dev in devs) {
  dlab <- dev_lab(dev)
  for (seed in seeds) {
    g1_col <- paste0("G1_", seed, "_", dlab)   # cortex-up   (up_group high)
    g2_col <- paste0("G2_", seed, "_", dlab)   # cortex-down (mirror)
    if (!all(c(g1_col, g2_col) %in% colnames(full@meta.data))) {
      message(paste0("Skipping missing masks: ", g1_col, " / ", g2_col)); next
    }
    run_dream  <- use_dream  && seed < n_iter_dream
    run_wilcox <- use_wilcox && seed < n_iter_sc
    run_lr     <- use_lr     && seed < n_iter_sc
    if (!any(use_deseq2, run_dream, run_wilcox, run_lr)) {
      message(paste0("Nothing selected for dev=", dlab, " seed=", seed, "; skipping.")); next
    }

    iter_root <- file.path(base_dir, paste0("dev_", dlab), paste0("seed_", seed))
    dirs <- list(
      global = file.path(iter_root, "Global_CT_Analysis"),
      local  = file.path(iter_root, "Local_Regional_Analysis"),
      pb     = file.path(iter_root, "Pseudobulk_Validation"),
      sc     = file.path(iter_root, "SingleCell_Tests")
    )
    sapply(dirs, function(x) if (!dir.exists(x)) dir.create(x, recursive = TRUE))

    message(paste0("\n>>> dev=", dlab, " seed=", seed,
                   "  (deseq2=", use_deseq2, " dream=", run_dream,
                   " wilcox=", run_wilcox, " lr=", run_lr, ")"))
    up   <- prepare_metadata(full[, which(as.logical(full@meta.data[[g1_col]]))], group_col, ref_level, comp_level)
    down <- prepare_metadata(full[, which(as.logical(full@meta.data[[g2_col]]))], group_col, ref_level, comp_level)

    run_iter_suite(up,   label = "UP",   file_tag = "WHOLE", dirs = dirs,
                   run_dream = run_dream, run_wilcox = run_wilcox, run_lr = run_lr)
    run_iter_suite(down, label = "DOWN", file_tag = "WHOLE", dirs = dirs,
                   run_dream = run_dream, run_wilcox = run_wilcox, run_lr = run_lr)

    run_ct_loop(up,   "UP",   dirs, run_dream, run_wilcox, run_lr)
    run_ct_loop(down, "DOWN", dirs, run_dream, run_wilcox, run_lr)

    rm(up, down); gc()
  }
}

message("\n--- ALL ANALYSES COMPLETE ---")
