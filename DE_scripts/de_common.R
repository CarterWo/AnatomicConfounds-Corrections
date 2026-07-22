# =============================================================================
# de_common.R
# -----------------------------------------------------------------------------
# Shared setup + DE model functions for the HPC-runnable DE scripts
# (DE_Base_Analysis.R and LMM_all.R). `source()` this from a driver script.
#
# Everything the two drivers share lives here: library loading, robust
# config discovery (works from any working directory on a cluster), the
# BiocParallel backend, CLI-arg helpers, the CSV->Seurat loader, and the DE
# model families themselves:
#   sample-aware : pseudobulk DESeq2, Dream LMM (blind / napari / quint)
#   cell-level   : Wilcoxon, logistic-regression LR (blind / napari / quint)
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
  library(yaml)
})

# ── future globals ceiling (required by the cell-level tests) ─────────────────
# Seurat's FindMarkers dispatches through the *future* framework rather than the
# BiocParallel backend used by Dream/DESeq2, and future refuses by default to
# export any global larger than 500 MiB. A whole-cohort object carries 1.7-2.7
# GiB of globals (the expression matrix appears twice: once as `data.use` and
# once captured in the `FUN` closure), so EVERY whole-cohort Wilcoxon and LR fit
# aborted before doing any work, with:
#   "The total size of the 5 globals exported for future expression ('FUN()') is
#    2.64 GiB. This exceeds the maximum allowed size 500.00 MiB"
# The cell-type subsets stayed under the cap, which is why only the whole-cohort
# results were ever missing.
#
# This is set here rather than per-sbatch so every driver gets it regardless of
# how it was launched. R_FUTURE_GLOBALS_MAXSIZE still overrides it if set.
# Note this is a ceiling on what future WILL export, not an allocation — raising
# it does not by itself increase the job's memory use.
options(future.globals.maxSize = local({
  env <- Sys.getenv("R_FUTURE_GLOBALS_MAXSIZE", "")
  if (nzchar(env)) as.numeric(env) else 16 * 1024^3   # 16 GiB
}))

# ── Locate this file (works when sourced OR run via Rscript) ──────────────────
.de_this_file <- function() {
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (!is.null(of)) return(normalizePath(of, winslash = "/", mustWork = FALSE))
  }
  args <- commandArgs(FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m)) return(normalizePath(sub("^--file=", "", m[1]), winslash = "/", mustWork = FALSE))
  NA_character_
}

# ── Config discovery + loading ────────────────────────────────────────────────
# Resolution order: explicit path -> $DE_CONFIG env var -> search upward from
# this script's directory -> search upward from getwd(). The config's directory
# is stored so relative paths in config.yaml resolve correctly regardless of the
# HPC working directory the job was launched from.
.DE_CFG_DIR <- NULL

de_load_config <- function(explicit = NULL) {
  find_it <- function() {
    if (!is.null(explicit) && nzchar(explicit) && file.exists(explicit))
      return(normalizePath(explicit, winslash = "/"))
    env <- Sys.getenv("DE_CONFIG")
    if (nzchar(env) && file.exists(env)) return(normalizePath(env, winslash = "/"))
    starts <- character(0)
    tf <- .de_this_file()
    if (!is.na(tf)) starts <- c(starts, dirname(tf))
    starts <- c(starts, getwd())
    for (start in starts) {
      d <- normalizePath(start, winslash = "/", mustWork = FALSE)
      repeat {
        p <- file.path(d, "config.yaml")
        if (file.exists(p)) return(p)
        parent <- dirname(d)
        if (identical(parent, d)) break
        d <- parent
      }
    }
    stop("config.yaml not found. Pass --config or set the DE_CONFIG env var.")
  }
  path <- find_it()
  .DE_CFG_DIR <<- dirname(path)
  message("Loaded config: ", path)
  yaml::read_yaml(path)
}

# ── Framework selection ───────────────────────────────────────────────────────
# The config groups inputs/outputs/composition under frameworks.<name>. Resolve
# which one this run uses: explicit arg -> $DE_FRAMEWORK env -> config
# `active_framework`. Returns that framework's sub-list (with $inputs, $outputs,
# $composition, $intermediate) plus $name. Falls back to the top-level config so
# older flat (single-framework) configs keep working.
de_select_framework <- function(cfg, name = NULL) {
  if (is.null(cfg$frameworks)) {
    message("No 'frameworks' block in config; using top-level inputs/outputs/composition.")
    cfg$name <- cfg$active_framework %||% "default"
    return(cfg)
  }
  if (is.null(name) || (length(name) == 1 && (is.na(name) || !nzchar(name)))) {
    env  <- Sys.getenv("DE_FRAMEWORK")
    name <- if (nzchar(env)) env else cfg$active_framework
  }
  if (is.null(name) || !nzchar(name))
    stop("No framework selected. Pass --framework, set DE_FRAMEWORK, ",
         "or set active_framework in config.yaml.")
  if (is.null(cfg$frameworks[[name]]))
    stop(sprintf("Framework '%s' not found in config. Available: %s",
                 name, paste(names(cfg$frameworks), collapse = ", ")))
  message("Framework: ", name)
  fw <- cfg$frameworks[[name]]
  fw$name <- name
  fw
}

# Null-coalescing helper (base R has none).
`%||%` <- function(a, b) if (is.null(a) || (length(a) == 1 && is.na(a))) b else a

# Resolve a (possibly relative) config path against the config directory so the
# scripts work no matter where the scheduler sets the working directory.
de_path <- function(p) {
  if (is.null(p) || !nzchar(p)) return(p)
  if (grepl("^(/|[A-Za-z]:)", p)) return(p)      # already absolute (unix or win)
  if (is.null(.DE_CFG_DIR)) return(p)
  file.path(.DE_CFG_DIR, p)
}

# ── BiocParallel backend (workers: arg -> $SLURM_CPUS_PER_TASK -> 5) ──────────
make_bpparam <- function(workers = NULL) {
  if (is.null(workers) || is.na(workers)) {
    env <- Sys.getenv("SLURM_CPUS_PER_TASK")
    workers <- if (nzchar(env)) as.integer(env) else 5L
  }
  workers <- as.integer(workers)
  # On Linux (the HPC) use fork-based MulticoreParam: workers share the large
  # expression matrices copy-on-write, so dream()/voom don't duplicate them per
  # worker (the main RAM blow-up). Fall back to SOCK SnowParam where fork is not
  # available (e.g. Windows), which copies data to each worker instead.
  if (.Platform$OS.type == "unix") {
    message("BiocParallel workers: ", workers, " (MulticoreParam / fork, shared memory)")
    BiocParallel::MulticoreParam(workers = workers, progressbar = FALSE)
  } else {
    message("BiocParallel workers: ", workers, " (SnowParam / SOCK)")
    SnowParam(workers = workers, type = "SOCK",
              progressbar = FALSE, exportglobals = FALSE)
  }
}

# ── Minimal CLI arg parsing (no extra package dependency) ─────────────────────
# Supports  --key value  and  --key=value  ; bare  --flag  -> TRUE.
de_parse_args <- function(defaults = list()) {
  args <- commandArgs(TRUE)
  out <- defaults
  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (grepl("^--", a)) {
      key <- sub("^--", "", a)
      if (grepl("=", key, fixed = TRUE)) {
        kv <- strsplit(key, "=", fixed = TRUE)[[1]]
        out[[kv[1]]] <- if (length(kv) > 1) kv[2] else TRUE
        i <- i + 1
      } else if (i + 1 <= length(args) && !grepl("^--", args[i + 1])) {
        out[[key]] <- args[i + 1]; i <- i + 2
      } else { out[[key]] <- TRUE; i <- i + 1 }
    } else i <- i + 1
  }
  out
}

# "0-99", "0:99", or "3,7,10" -> integer vector
de_parse_int_range <- function(s) {
  s <- as.character(s)
  if (grepl(",", s)) return(as.integer(trimws(strsplit(s, ",")[[1]])))
  if (grepl("[-:]", s)) { p <- as.integer(strsplit(s, "[-:]")[[1]]); return(seq.int(p[1], p[2])) }
  as.integer(s)
}

# "1.5,1,0.5,0.25" -> numeric vector
de_parse_num_list <- function(s) as.numeric(trimws(strsplit(as.character(s), ",")[[1]]))

# Format a deviation to match Python's f"{dev:g}" so the G1_{seed}_{dev} /
# G2_{seed}_{dev} mask column names line up with those tagged in the
# Composition_Engineering step.
dev_lab <- function(d) format(d, trim = TRUE, scientific = FALSE)

# ── I/O: build a Seurat object from an exported CSV folder ─────────────────────
make_cfg <- function(root_dir,
                     seurat_rds_name    = "seurat_object.rds",
                     raw_counts_name    = "raw_counts.csv",
                     logged_counts_name = "normalized_counts.csv",
                     features_name      = "features_counts.csv",
                     metadata_name      = "cell_metadata.csv",
                     coords_name        = "coords_xy.csv",
                     quint_labels_name  = "Color_key.xlsm") {
  list(
    root_dir      = root_dir,
    seurat_rds    = file.path(root_dir, seurat_rds_name),
    raw_counts    = file.path(root_dir, raw_counts_name),
    logged_counts = file.path(root_dir, logged_counts_name),
    features      = file.path(root_dir, features_name),
    metadata      = file.path(root_dir, metadata_name),
    coords        = file.path(root_dir, coords_name),
    quint_labels  = file.path(root_dir, quint_labels_name)
  )
}

build_seurat_from_folder <- function(cfg, assay_name = "RNA") {
  message(paste0("Loading from: ", cfg$root_dir))

  # Small metadata tables (features/meta/coords): fast fread if available.
  read_tbl <- function(path) {
    if (requireNamespace("data.table", quietly = TRUE)) {
      df <- data.table::fread(path, header = TRUE, data.table = FALSE)
      rownames(df) <- df[[1]]; df[[1]] <- NULL; df
    } else read.csv(path, row.names = 1, check.names = FALSE)
  }

  # Big expression CSV (cells x genes) -> sparse (genes x cells) dgCMatrix. Keeps
  # only ONE dense copy transiently and stores the result sparse (CosMx data is
  # mostly zeros), instead of holding two ~4 GiB dense frames + a dense 'data'
  # layer. Needs data.table for the memory/speed win; falls back to read.csv.
  read_expr_sparse <- function(path) {
    if (requireNamespace("data.table", quietly = TRUE)) {
      dt  <- data.table::fread(path, header = TRUE)
      ids <- dt[[1]]; dt[[1]] <- NULL
      m   <- as.matrix(dt); rm(dt)
    } else {
      df  <- read.csv(path, row.names = 1, check.names = FALSE)
      ids <- rownames(df); m <- as.matrix(df); rm(df)
    }
    sp <- Matrix::t(Matrix::Matrix(m, sparse = TRUE)); rm(m); gc()
    colnames(sp) <- ids
    sp
  }

  features <- read_tbl(cfg$features)
  meta     <- read_tbl(cfg$metadata)
  coords   <- read_tbl(cfg$coords)

  counts <- read_expr_sparse(cfg$raw_counts)     # genes x cells (sparse)
  logged <- read_expr_sparse(cfg$logged_counts)  # genes x cells (sparse)

  if (nrow(counts) != nrow(features))
    stop(paste("Mismatch: counts has", nrow(counts), "genes, but features file has", nrow(features)))
  rownames(counts) <- rownames(features)
  rownames(logged) <- rownames(features)

  common_cells <- Reduce(intersect, list(colnames(counts), rownames(meta),
                                         rownames(coords), colnames(logged)))
  if (length(common_cells) == 0)
    stop("No common Cell IDs found. Check your CSV row names.")
  message(paste0("  Matched ", length(common_cells), " cells across all files."))

  counts <- counts[, common_cells, drop = FALSE]
  logged <- logged[, common_cells, drop = FALSE]
  meta   <- meta[common_cells, , drop = FALSE]
  coords <- coords[common_cells, , drop = FALSE]

  seu <- CreateSeuratObject(counts = counts, meta.data = meta, assay = assay_name)
  seu <- SetAssayData(seu, layer = "data", new.data = logged)  # sparse, not as.matrix
  seu <- AddMetaData(seu, metadata = coords)
  return(seu)
}

# ── Resume support ────────────────────────────────────────────────────────────
# Set by the drivers from --resume. When TRUE, any runner whose output CSV is
# already present AND looks complete returns immediately instead of refitting.
#
# This exists because a run can end with outputs missing in two ways that leave
# no trace in the Slurm exit code: a walltime kill partway through the model
# loop, and a runner whose tryCatch caught an error, logged "!! FAILED", and let
# the script continue to a clean exit. Resume treats both the same way — if the
# CSV is not there, redo it; if it is, skip.
#
# "Complete" deliberately means a header PLUS at least one data row. A job killed
# mid-write can leave a truncated or header-only file, and silently accepting one
# of those would be worse than paying to refit it.
DE_RESUME <- FALSE

de_output_done <- function(out_path) {
  if (!DE_RESUME) return(FALSE)
  if (!file.exists(out_path) || file.size(out_path) <= 0) return(FALSE)
  n <- tryCatch(length(readLines(out_path, n = 2L, warn = FALSE)),
                error = function(e) 0L)
  if (n < 2L) {
    message(paste("    Resume: refitting incomplete", basename(out_path)))
    return(FALSE)
  }
  message(paste("    Resume: skipping existing", basename(out_path)))
  TRUE
}

# ── DE model runners (require a global `param` BiocParallel backend) ───────────

# run_de_dream: region-aware Dream LMM.
#
# ct_col controls the cell-type random effect and MUST be NULL for any
# cell-type-specific run. Testing for cell-type structure inside a single cell
# type is not just redundant: the fine-grained 'cell_type' labels are themselves
# anatomical (Astrocytes.cortex.hippocampus vs Astrocytes.thalamus.hypothalamus),
# so ct_simple == "Astrocytes" still carries two levels and (1|cell_type) would
# soak up part of the very regional signal this pipeline is built to measure.
# The ct loops in LMM_all.R / LMM_dev15.R / DE_Base_Analysis.R pass ct_col = NULL.
run_de_dream <- function(seurat_obj, dataset_name, out_prefix, save_dir,
                         region_col = "napari_region", ct_col = "cell_type") {
  message(paste0("\n>>> [DREAM] STARTING FOR: ", dataset_name))
  out_path <- file.path(save_dir, paste0(out_prefix, ".csv"))
  if (de_output_done(out_path)) return(invisible(NULL))
  ct_term <- ""
  if (is.null(ct_col)) {
    message("    Cell-type-specific run - no cell-type random effect.")
  } else if (!ct_col %in% colnames(seurat_obj@meta.data)) {
    message(paste0("    Missing column ", ct_col, " - no cell-type random effect."))
  } else {
    num_cts <- length(unique(seurat_obj@meta.data[[ct_col]]))
    if (num_cts > 1) {
      message(paste0("    Detected ", num_cts, " levels in ", ct_col,
                     ". Including (1|", ct_col, ")."))
      ct_term <- paste0(" + (1|", ct_col, ")")
    } else {
      message(paste0("    Single level in ", ct_col, " - removing (1|", ct_col,
                     ") from formula."))
    }
  }
  form_str <- paste0("~ Treatment + log_depth + (1|", region_col, ")", ct_term,
                     " + (1|sample_ID)")
  form_de <- as.formula(form_str)
  message(paste("    Formula:", deparse(form_de)))
  counts <- as.matrix(GetAssayData(seurat_obj, layer = "counts"))
  info   <- seurat_obj@meta.data
  counts <- counts[rowSums(counts) > 0, ]
  dge    <- calcNormFactors(DGEList(counts))
  message("    Calculating voom weights...")
  vobj <- voomWithDreamWeights(dge, form_de, info, BPPARAM = param)
  fit  <- eBayes(dream(vobj, form_de, info, BPPARAM = param))
  target_coef <- grep("Treatment", colnames(fit$coefficients), value = TRUE)
  target_coef <- target_coef[length(target_coef)]
  if (length(target_coef) > 0) {
    de_res       <- topTable(fit, coef = target_coef, number = Inf, sort.by = "P", confint = TRUE)
    de_res$Gene  <- rownames(de_res)
    de_res$pct_1 <- rowMeans(counts[rownames(de_res), , drop = FALSE] > 0)
    write.csv(de_res, file = out_path, row.names = FALSE)
  }
}

# run_de_blind: region-blind Dream LMM with per-group expression metrics
run_de_blind <- function(seurat_obj, dataset_name, out_prefix, save_dir) {
  message(paste0("\n>>> [DREAM BLIND] STARTING FOR: ", dataset_name))
  out_path <- file.path(save_dir, paste0(out_prefix, ".csv"))
  if (de_output_done(out_path)) return(invisible(NULL))
  form_de  <- ~ Treatment + log_depth + (1|sample_ID)
  geneExpr <- as.matrix(GetAssayData(seurat_obj, layer = "data"))
  counts   <- as.matrix(GetAssayData(seurat_obj, layer = "counts"))
  info     <- seurat_obj@meta.data
  tryCatch({
    dge  <- calcNormFactors(DGEList(counts[rowSums(counts) > 0, ]))
    vobj <- voomWithDreamWeights(dge, form_de, info, BPPARAM = param)
    fit  <- eBayes(dream(vobj, form_de, info, BPPARAM = param))
    target_coef <- grep("Treatment", colnames(fit$coefficients), value = TRUE)
    target_coef <- target_coef[length(target_coef)]
    if (length(target_coef) > 0) {
      de_res        <- topTable(fit, coef = target_coef, number = Inf, sort.by = "P", confint = TRUE)
      de_res$Gene   <- rownames(de_res)
      ref_level     <- levels(seurat_obj$Treatment)[1]
      target_level  <- levels(seurat_obj$Treatment)[2]
      cells_ref     <- which(seurat_obj$Treatment == ref_level)
      cells_target  <- which(seurat_obj$Treatment == target_level)
      de_res$pct_ref    <- rowMeans(geneExpr[rownames(de_res), cells_ref,    drop = FALSE] > 0)
      de_res$pct_target <- rowMeans(geneExpr[rownames(de_res), cells_target, drop = FALSE] > 0)
      de_res$avg_ref    <- rowMeans(geneExpr[rownames(de_res), cells_ref,    drop = FALSE])
      de_res$avg_target <- rowMeans(geneExpr[rownames(de_res), cells_target, drop = FALSE])
      write.csv(de_res, file = out_path, row.names = FALSE)
      message(paste("    -> Saved:", out_path))
    }
  }, error = function(e) message(paste("    !! SKIPPING:", e$message)))
}

# run_pseudobulk_deseq2: pseudobulk DESeq2 validation
run_pseudobulk_deseq2 <- function(seurat_obj, dataset_name, out_prefix, save_dir) {
  message(paste0("    >>> [PB] Running Pseudo-bulk DESeq2: ", dataset_name))
  out_path <- file.path(save_dir, paste0("DESEQ2", out_prefix, ".csv"))
  if (de_output_done(out_path)) return(invisible(NULL))
  cts <- AggregateExpression(seurat_obj, group.by = "sample_ID", assays = "RNA", slot = "counts")$RNA
  colnames(cts) <- gsub("-", "_", gsub("^g", "", colnames(cts)))
  colData <- seurat_obj@meta.data %>%
    dplyr::select(sample_ID, Treatment) %>%
    dplyr::distinct(sample_ID, .keep_all = TRUE)
  colData <- colData[match(colnames(cts), colData$sample_ID), ]
  rownames(colData) <- colData$sample_ID
  tryCatch({
    dds <- DESeqDataSetFromMatrix(countData = cts, colData = colData, design = ~ Treatment)
    dds <- dds[rowSums(counts(dds)) >= 10, ]
    dds <- DESeq(dds, quiet = TRUE)
    res <- as.data.frame(results(dds))
    res$Gene <- rownames(res)
    write.csv(res, file = out_path, row.names = FALSE)
    message(paste("    -> Saved PB:", out_path))
  }, error = function(e) message(paste("    !! PB FAILED:", e$message)))
}

# ── Cell-level tests: Wilcoxon and logistic-regression LR (Seurat FindMarkers) ─
# These treat every cell as an independent observation — no sample_ID random
# effect and no pseudobulk aggregation — which is exactly why they are run here
# alongside the Dream LMM / DESeq2: they are the models expected to absorb a
# composition shift as if it were biology.
#
# All of Seurat's pre-filters are opened up (logfc.threshold = 0, min.pct = 0,
# min.cells.group = 1) so every gene is returned for every seed. The gene sets
# then stay identical across methods, deviations, and iterations, which is what
# lets the downstream aggregation compare them seed-for-seed.
.sc_find_markers <- function(seurat_obj, test_use, latent_vars = NULL) {
  Idents(seurat_obj) <- "Treatment"
  lv <- levels(seurat_obj$Treatment)
  if (length(lv) < 2)
    stop("need two Treatment levels for a cell-level test, found ", length(lv))
  # prepare_metadata() relevels so lv[1] is the reference; ident.1 = lv[2] makes
  # avg_log2FC read comparison-vs-reference, matching the Dream/DESeq2 sign.
  fm_args <- list(object = seurat_obj, ident.1 = lv[2], ident.2 = lv[1],
                  test.use = test_use, logfc.threshold = 0, min.pct = 0,
                  min.cells.group = 1, verbose = FALSE)
  if (!is.null(latent_vars)) fm_args$latent.vars <- latent_vars
  res <- do.call(FindMarkers, fm_args)
  res$Gene <- rownames(res)
  res
}

# run_de_wilcox: Wilcoxon rank-sum on the log-normalized 'data' layer. Takes no
# covariates at all — the deliberately naive baseline.
run_de_wilcox <- function(seurat_obj, dataset_name, out_prefix, save_dir) {
  message(paste0("\n>>> [WILCOX] STARTING FOR: ", dataset_name))
  out_path <- file.path(save_dir, paste0(out_prefix, ".csv"))
  if (de_output_done(out_path)) return(invisible(NULL))
  tryCatch({
    res <- .sc_find_markers(seurat_obj, "wilcox")
    write.csv(res, file = out_path, row.names = FALSE)
    message(paste("    -> Saved:", out_path))
  }, error = function(e) message(paste("    !! WILCOX FAILED:", e$message)))
}

# run_de_lr: logistic-regression likelihood-ratio test. Always adjusts for
# log_depth; pass region_col to additionally condition on anatomy, giving the
# blind / napari / quint trio that mirrors the Dream LMM runners.
run_de_lr <- function(seurat_obj, dataset_name, out_prefix, save_dir,
                      region_col = NULL) {
  message(paste0("\n>>> [LR] STARTING FOR: ", dataset_name))
  out_path <- file.path(save_dir, paste0(out_prefix, ".csv"))
  if (de_output_done(out_path)) return(invisible(NULL))
  latent <- "log_depth"
  if (!is.null(region_col)) {
    if (!region_col %in% colnames(seurat_obj@meta.data)) {
      message(paste0("    !! LR SKIPPED: missing column ", region_col))
      return(invisible(NULL))
    }
    if (length(unique(seurat_obj@meta.data[[region_col]])) < 2) {
      message(paste0("    Single level in ", region_col, " - dropping it from latent.vars."))
    } else {
      latent <- c(latent, region_col)
    }
  }
  message(paste("    latent.vars:", paste(latent, collapse = " + ")))
  tryCatch({
    res <- .sc_find_markers(seurat_obj, "LR", latent_vars = latent)
    write.csv(res, file = out_path, row.names = FALSE)
    message(paste("    -> Saved:", out_path))
  }, error = function(e) message(paste("    !! LR FAILED:", e$message)))
}

# prepare_metadata: use the configured contrast column (group_col) as the DE
# 'Treatment', restrict to the two contrast groups (ref_level + comparison_level,
# dropping any other level such as the FMT 'Cntrl' arm), scale sequencing depth,
# and relevel so ref_level is the reference. Restricting to exactly two levels also
# guarantees the region-blind/DESeq2 runners report comparison-vs-reference (their
# "last Treatment coefficient" / levels[2] logic is unambiguous with two groups).
# comparison_level = NULL keeps every group (legacy behaviour).
prepare_metadata <- function(seurat_obj, group_col, ref_level, comparison_level = NULL) {
  names(seurat_obj@meta.data)[names(seurat_obj@meta.data) == group_col] <- "Treatment"
  if (!is.null(comparison_level) && nzchar(comparison_level)) {
    keep <- seurat_obj$Treatment %in% c(ref_level, comparison_level)
    if (!any(keep))
      stop(sprintf("prepare_metadata: no cells in contrast groups '%s' / '%s' (column '%s').",
                   ref_level, comparison_level, group_col))
    seurat_obj <- seurat_obj[, keep]
  }
  seurat_obj$Treatment <- droplevels(as.factor(seurat_obj$Treatment))
  if (ref_level %in% levels(seurat_obj$Treatment))
    seurat_obj$Treatment <- relevel(seurat_obj$Treatment, ref = ref_level)
  seurat_obj$log_depth <- as.numeric(scale(log10(seurat_obj$nFeature_RNA + 1)))
  return(seurat_obj)
}
