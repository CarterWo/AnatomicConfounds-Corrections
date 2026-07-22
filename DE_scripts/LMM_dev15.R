#!/usr/bin/env Rscript
# =============================================================================
# LMM_dev15.R
# -----------------------------------------------------------------------------
# Deviation-1.5-only rerun of LMM_all.R, for finishing the seeds that the
# original dev_1.5 array task never produced.
#
#   Rscript DE_scripts/LMM_dev15.R                        # seeds 3-99, DESeq2 only
#   Rscript DE_scripts/LMM_dev15.R --seeds 3-12           # one array-task slice
#   Rscript DE_scripts/LMM_dev15.R --dream-seeds 3        # also run the Dream LMM
#
# Differences from LMM_all.R:
#   * the deviation is fixed at 1.5 (no --devs; nothing else can be requested)
#   * seeds default to 3-99, i.e. the ones missing from LMM_output/<fw>/dev_1.5
#   * the Dream LMM is OFF by default (--dream-seeds 0), so each seed runs the
#     pseudobulk DESeq2 arm only -- seeds 0-2 already have their Dream results
#
# Output layout is identical to LMM_all.R, so the new seeds drop straight in
# beside the existing ones:
#   <lmm_results_dir>/dev_1.5/seed_<seed>/{Pseudobulk_Validation,
#                                          Local_Regional_Analysis,
#                                          Global_CT_Analysis}/
#
# Options (defaults come from the selected framework's config.yaml `composition`):
#   --config PATH     config.yaml (else DE_CONFIG env / auto-discovered)
#   --framework NAME  which frameworks.<name> block to use (else $DE_FRAMEWORK /
#                     config active_framework)
#   --workers N       BiocParallel workers (else $SLURM_CPUS_PER_TASK, else 5)
#   --seeds  RANGE    iterations, e.g. "3-99", "3:12", or "7,11,40"
#   --dream-seeds N   run the Dream LMM on seeds < N (default 0 = never)
# =============================================================================

# Locate + source the shared module (independent of the working directory).
.this <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
SCRIPT_DIR <- dirname(normalizePath(.this, winslash = "/", mustWork = FALSE))
source(file.path(SCRIPT_DIR, "de_common.R"))

DEV <- 1.5   # fixed for this script

args      <- de_parse_args(list(config = NULL, workers = NULL, framework = NULL,
                                seeds = NULL, `dream-seeds` = NULL))
cfg_all   <- de_load_config(args$config)
fw        <- de_select_framework(cfg_all, args$framework)
COMP      <- fw$composition
OUT       <- fw$outputs
group_col <- COMP$group_col
ref_level <- COMP$reference
comp_level <- COMP$comparison   # non-reference contrast group; restricts the DE to
                                # reference vs comparison (drops e.g. the FMT 'Cntrl' arm)

# Seeds 0-2 already exist (they carry the Dream results); default to the rest.
seeds        <- if (!is.null(args$seeds)) de_parse_int_range(args$seeds) else seq.int(3, COMP$deseq2_iterations - 1)
n_iter_dream <- if (!is.null(args$`dream-seeds`)) as.integer(args$`dream-seeds`) else 0L
param        <- make_bpparam(args$workers)

dlab     <- dev_lab(DEV)
base_dir <- de_path(OUT$lmm_results_dir)

message("Framework: ", fw$name)
message("Contrast: ", comp_level, " vs ", ref_level, " (column ", group_col, ")")
message("Deviation: ", dlab, " (fixed)")
message("Seeds: ", paste(range(seeds), collapse = ".."), " (n=", length(seeds), ")",
        " | Dream LMM on seeds < ", n_iter_dream,
        if (n_iter_dream <= 0) "  -> DESeq2 only" else "")

# ── Load the single full-cohort export ────────────────────────────────────────
message("Loading full cohort")
full <- build_seurat_from_folder(make_cfg(de_path(OUT$de_export_base)))

# Fail loudly if this export predates the dev-1.5 masks, rather than silently
# skipping every seed and exiting 0 (which is how an empty dev_1.5 looks).
present <- vapply(seeds, function(s)
  all(c(paste0("G1_", s, "_", dlab), paste0("G2_", s, "_", dlab)) %in% colnames(full@meta.data)),
  logical(1))
if (!any(present))
  stop(sprintf(paste0("No G1_/G2_ mask columns for dev %s at any requested seed. ",
                      "The export at %s is missing the dev-%s composition masks -- ",
                      "rerun Composition_Engineering and re-export before this script."),
               dlab, de_path(OUT$de_export_base), dlab))
if (!all(present))
  message("NOTE: ", sum(!present), " of ", length(seeds),
          " requested seeds have no dev-", dlab, " masks and will be skipped: ",
          paste(seeds[!present], collapse = ", "))

# DESeq2 for every iteration; Dream (blind + region-aware) only when run_dream.
# ct_col = NULL marks a cell-type-specific run and keeps the cell-type random
# effect out of the Dream formula; the whole-cohort calls leave it at its default.
run_iter_suite <- function(obj, label, file_tag, dirs, run_dream, ct_col = "cell_type") {
  run_pseudobulk_deseq2(obj,
    dataset_name = paste0(label, "_PB"),
    out_prefix   = paste0(label, "_", file_tag, "_PB"),
    save_dir     = dirs$pb)
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
}

target_cell_types <- c("Astrocytes", "Microglia")

run_ct_loop <- function(data_obj, dataset_label, dirs, run_dream) {
  for (ct in target_cell_types) {
    cell_col <- if (ct == "Astrocytes.cortex.hippocampus") "cell_type" else "ct_simple"
    obj_ct   <- data_obj[, data_obj@meta.data[[cell_col]] == ct]
    if (ncol(obj_ct) < 100) { message(paste("Skipping low cell count:", ct)); next }
    ct_clean <- gsub("[^A-Za-z0-9]", "_", ct)
    run_iter_suite(obj_ct,
      label     = paste0(dataset_label, "_", ct),
      file_tag  = ct_clean,
      dirs      = dirs,
      run_dream = run_dream,
      ct_col    = NULL)   # already one cell type - never model cell type here
    rm(obj_ct); gc()
  }
}

# ── Main loop: dev 1.5 x iteration x {UP, DOWN} ───────────────────────────────
for (seed in seeds) {
  g1_col <- paste0("G1_", seed, "_", dlab)   # cortex-up   (up_group high)
  g2_col <- paste0("G2_", seed, "_", dlab)   # cortex-down (mirror)
  if (!all(c(g1_col, g2_col) %in% colnames(full@meta.data))) {
    message(paste0("Skipping missing masks: ", g1_col, " / ", g2_col)); next
  }
  run_dream <- seed < n_iter_dream

  iter_root <- file.path(base_dir, paste0("dev_", dlab), paste0("seed_", seed))
  dirs <- list(
    global = file.path(iter_root, "Global_CT_Analysis"),
    local  = file.path(iter_root, "Local_Regional_Analysis"),
    pb     = file.path(iter_root, "Pseudobulk_Validation")
  )
  sapply(dirs, function(x) if (!dir.exists(x)) dir.create(x, recursive = TRUE))

  message(paste0("\n>>> dev=", dlab, " seed=", seed, "  (dream=", run_dream, ")"))
  up   <- prepare_metadata(full[, which(as.logical(full@meta.data[[g1_col]]))], group_col, ref_level, comp_level)
  down <- prepare_metadata(full[, which(as.logical(full@meta.data[[g2_col]]))], group_col, ref_level, comp_level)

  run_iter_suite(up,   label = "UP",   file_tag = "WHOLE", dirs = dirs, run_dream = run_dream)
  run_iter_suite(down, label = "DOWN", file_tag = "WHOLE", dirs = dirs, run_dream = run_dream)

  run_ct_loop(up,   "UP",   dirs, run_dream)
  run_ct_loop(down, "DOWN", dirs, run_dream)

  rm(up, down); gc()
}

message("\n--- DEV 1.5 ANALYSES COMPLETE ---")
