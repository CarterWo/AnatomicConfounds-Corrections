#!/usr/bin/env Rscript
##
## simulate_ground_truth.R
##
## Ground-truth simulation for Reviewer 1 (iScience ISCIENCE-D-26-06541, R1
## comment 3: "How can the authors show that the aware model improves
## accuracy rather than only reducing variation?").
##
## Generates a synthetic single-cell spatial dataset with a KNOWN,
## investigator-specified treatment effect, under four scenarios:
##   A. null        -- no true treatment effect anywhere; only anatomical
##                      composition is imbalanced across treatment groups
##                      (tests false-positive/FDR control under pure
##                      confounding, R1's scenario 1)
##   B. shared       -- a uniform treatment effect present in every region
##                      (tests that region-aware correction does not erase
##                      genuine global signal, R1's scenario 2)
##   C. region_specific -- treatment effect present only in one region
##                      (Cortex) (R1's scenario 3)
##   D. interaction  -- opposite-sign treatment effects in two regions
##                      (Cortex vs. Hippocampus) (R1's scenario 4, and
##                      directly tests R1's causal-framework critique that a
##                      single random intercept for region cannot capture
##                      treatment x region interaction)
##
## In all four scenarios, anatomical (region) composition is deliberately
## imbalanced across treatment groups -- mirroring the manuscript's own
## composition-engineering design (ezy_seq::tag_region_abundance_by_FMT) --
## so the simulation reproduces the exact confounding structure the paper
## is about, but now with ground truth available for scoring.
##
## DESIGN PRINCIPLE (see conversation record / CLAUDE.md): this simulator is
## NOT fit to the lab's own CosMx data. All distributional parameters are
## set from published, citable literature values for single-cell /
## imaging-based spatial transcriptomics data. This keeps the ground-truth
## simulation fully independent of the paper's N=6 dataset, so it cannot be
## accused of "inflating" the same small sample -- it is a self-contained
## statistical testbed for the four DE pipelines (Dream LMM blind/napari/
## quint, DESeq2 pseudobulk, Seurat LR, Wilcoxon), used completely unmodified
## from DE_scripts/.
##
## Parameter sources:
##   - splatter simulation framework: Zappia, Phipson & Oshlack (2017)
##     "Splatter: simulation of single-cell RNA sequencing data."
##     Genome Biology 18:174.
##   - Gene-mean (gamma shape/rate) and biological coefficient of variation
##     (BCV) defaults are the literature-typical values validated across
##     many real scRNA-seq datasets in the Splatter paper itself
##     (BCV ~ 0.2-0.4 range), consistent with independently reported NB
##     dispersion ranges for scRNA-seq in Vieth et al. (2017) "powsimR:
##     power analysis for bulk and single cell RNA-seq experiments."
##     Bioinformatics 33(21):3486-3488.
##   - Reduced technical dropout relative to whole-transcriptome droplet
##     scRNA-seq is used to reflect targeted imaging-based panels
##     (CosMx/MERFISH, ~1000-plex), per the higher per-gene sensitivity
##     reported for targeted probe panels in Moffitt et al. (2018)
##     "High-throughput single-cell gene-expression profiling with
##     multiplexed error-robust FISH." PNAS 113(39):11046-11051, and
##     He et al. (2022) "High-plex imaging of RNA and proteins at
##     subcellular resolution in fixed tissue." Nat. Biotechnol. 40:1794-1806
##     (CosMx WTx platform description).
##   - Injected treatment log2 fold changes (|log2FC| = 1.0-1.5) are within
##     the moderate effect-size range typically detectable/reported in
##     mammalian tissue RNA-seq DE studies; see Schurch et al. (2016)
##     "How many biological replicates are needed in an RNA-seq experiment
##     and which differential expression tool should you use?" RNA 22:839-851
##     for typical achievable effect sizes and replicate-count power curves.
##
## Effect injection method: for each designated ground-truth gene and target
## cell subset, we redraw counts from a negative binomial distribution whose
## mean is the matched baseline-scenario group mean scaled by the known fold
## change (2^log2FC) and whose dispersion is the gene's own method-of-moments
## NB dispersion estimated from the (pre-injection) simulated baseline. This
## is the same "counts thinning/rescaling + resampling" approach used to
## establish ground truth in RNA-seq DE-method benchmarks such as
## compcodeR (Soneson & Delorenzi, 2013, BMC Bioinformatics) and polyester
## (Frazee et al., 2015, Bioinformatics) -- it preserves the realistic
## marginal count distribution while imposing an exactly known effect size,
## so downstream RMSE/bias/coverage against that known value are meaningful.
##
## Cell-type composition (R1's explicit simulation requirement -- "region-
## specific baseline expression, differences in region abundance, cell-type
## composition, sample-level variability, and treatment-by-region
## interactions"): three cell types are simulated -- Excitatory_Neuron,
## Astrocyte, Microglia -- matching the manuscript's own actual analyzed
## cell-type strata (not an arbitrary Neuron/Glia binary). Two components:
##   1. Cell-type COMPOSITION differs by region (ct_prob_by_region below),
##      mirroring the manuscript's own reported finding that "a higher
##      concentration of excitatory neurons in the isocortex relative to
##      other brain regions led to inversion in the relative abundance of
##      excitatory neurons between Healthy and Stroke FMT groups in G1 and
##      G2" (Results text) -- i.e. cell-type composition shift is a
##      downstream CONSEQUENCE of the already-simulated region-abundance
##      imbalance, exactly as in the real data, not a second independently
##      invented imbalance mechanism.
##   2. Cell-type MARKER GENES carry real expression signal (previously they
##      did not -- cell_type was a label with no expression consequence,
##      unable to function as a testable confound). Marker-gene fold change
##      is set to a literature-grounded, deliberately conservative log2FC =
##      4.0 (16-fold), citing Zhang et al. (2014) "An RNA-Sequencing
##      Transcriptome and Splicing Database of Glia, Neurons, and Vascular
##      Cells of the Cerebral Cortex" J Neurosci 34(36):11929-11947 -- a
##      purified-cell-type bulk RNA-seq reference atlas (not our own data)
##      in which canonical markers (e.g. Gfap/Aqp4 for astrocytes, P2ry12/
##      Cx3cr1 for microglia, Slc17a7 for excitatory neurons) are enriched
##      far more than 16-fold in their defining cell type; 16-fold is a
##      conservative floor, not a fitted value.
##
## Usage:
##   Rscript simulate_ground_truth.R <seed> <scenario> <out_root> [dev]
##   scenario in {null, shared, region_specific, interaction}
##   dev: composition-imbalance magnitude in SD units (default 1.0 = original
##   ±1 SD design). Pass a distinct out_root per dev level to avoid
##   collisions -- this script does not namespace output paths by dev itself.
##
## Output: a set of CSVs per scenario/seed under
##   <out_root>/<scenario>/seed_<seed>/{raw_counts.csv, normalized_counts.csv,
##   features_counts.csv, cell_metadata.csv, coords_xy.csv}
## matching the format consumed by DE_scripts' make_cfg()/
## build_seurat_from_folder(), plus a ground_truth.csv recording the true
## log2FC (0 for null genes) used for scoring.
##
suppressPackageStartupMessages({
  library(splatter)
  library(SingleCellExperiment)
  library(Matrix)
})

## ---------------------------------------------------------------------
## 0. Arguments
## ---------------------------------------------------------------------
args     <- commandArgs(trailingOnly = TRUE)
seed     <- as.integer(ifelse(length(args) >= 1, args[1], Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
scenario <- ifelse(length(args) >= 2, args[2], "null")
out_root <- ifelse(length(args) >= 3, args[3], "sim_out")
## dev: composition-imbalance magnitude, in units of cross-sample SD of the
## isocortex fraction (see composition_dev below). Default 1.0 reproduces
## the original ±1 SD design; exposed as a CLI arg so the imbalance
## magnitude can be swept (R1 Question 2: "test several levels of
## imbalance, not only ±1 standard deviation").
dev_arg  <- as.numeric(ifelse(length(args) >= 4, args[4], "1.0"))

stopifnot(scenario %in% c("null", "shared", "region_specific", "interaction"))
set.seed(seed)

## ---------------------------------------------------------------------
## 1. Design
## ---------------------------------------------------------------------
n_genes           <- 1000                 # matches ~1000 genes tested in the base manuscript
regions           <- c("Cortex", "Hippocampus", "Striatum", "Cerebellum", "Olfactory")
n_regions         <- length(regions)
groups            <- c("Control", "Treatment")
n_samples_per_grp <- 3                    # matches the manuscript's N=3/group design
samples           <- paste0(rep(groups, each = n_samples_per_grp), "_",
                             rep(seq_len(n_samples_per_grp), times = length(groups)))
sample_group      <- setNames(rep(groups, each = n_samples_per_grp), samples)
cell_types        <- c("Excitatory_Neuron", "Astrocyte", "Microglia")  # matches the manuscript's own analyzed cell-type strata
n_celltype_marker_genes <- 30           # per cell type (90 total), distinct from the 100 treatment true-DE genes
celltype_log2fc   <- 4.0                # conservative marker fold change (Zhang et al. 2014); see header note

cells_per_sample_region <- 800            # -> ~4000 cells/sample, ~24000 cells total
n_true_genes      <- 100                  # ground-truth DE genes (10% of panel)

roi               <- "Cortex"             # region-of-interest for composition imbalance (matches manuscript)
roi2              <- "Hippocampus"        # second region used for the interaction scenario
composition_dev   <- dev_arg               # +-dev SD shift; 1.0 matches manuscript's original design
## baseline_roi_frac/baseline_roi_sd reuse the manuscript's OWN reported,
## Napari-measured cross-sample isocortex proportion and SD ("shift the
## isocortex proportion to +1 SD (33.8%) or -1 SD (19.0%) of the cross-sample
## mean (26.4% +/- 7.4%)" -- Resubmission_Materials manuscript text). This is
## a single descriptive design statistic (how imbalanced is realistic), NOT
## a gene-expression/dispersion fit to the real data, so it does not
## reintroduce the "inflated N=6" concern -- the injected treatment effect
## and all count-level statistics remain fully synthetic (splatter). Reusing
## it here also makes this simulation's composition-imbalance design directly
## consistent with the manuscript's own already-published methodology, rather
## than an arbitrary/unrelated placeholder value.
## NOTE: the manuscript text elsewhere reports a very slightly different
## value for the same measurement (25.1% +/- 7.6%) -- a minor internal
## inconsistency worth reconciling during the STAR Methods cleanup pass, but
## immaterial to which of the two nearly-identical values is used here.
baseline_roi_frac <- 0.264                # cross-sample mean isocortex fraction (manuscript-reported)
baseline_roi_sd   <- 0.074                # cross-sample SD (manuscript-reported)
## hi/lo targets (below) are reached by SUBSAMPLING DOWN from the raw simulated
## pool, never by inventing cells -- so the raw pool must contain at least
## hi = baseline_roi_frac + composition_dev*baseline_roi_sd share of Cortex
## cells, with margin, or both the "hi" and "lo" arms silently truncate to the
## same availability ceiling and the intended composition imbalance vanishes.
## raw_roi_share sets that oversampled baseline (0.45 comfortably exceeds the
## hi target of ~0.338 implied by baseline_roi_frac + baseline_roi_sd above).
raw_roi_share     <- 0.45

## ---------------------------------------------------------------------
## 2. Baseline simulation (splatter) -- literature-parameterized, NOT
##    fit to the lab's own CosMx data.
##    batch  = sample_ID  -> sample-level variability (matches the
##             (1|sample_ID) random-intercept term in the real models)
##    group  = region     -> region-specific baseline expression (matches
##             the (1|napari_region)/(1|quint_region) terms)
## ---------------------------------------------------------------------
batch_cells <- rep(cells_per_sample_region * n_regions, length(samples))

params <- newSplatParams(
  nGenes        = n_genes,
  batchCells    = batch_cells,
  batch.facLoc  = 0.10,     # modest sample-level technical/biological shift
  batch.facScale= 0.10,
  group.prob    = c(raw_roi_share, rep((1 - raw_roi_share) / (n_regions - 1), n_regions - 1)),
  de.prob       = 0.15,     # fraction of genes with region-specific baseline expression
  de.facLoc     = 0.4,
  de.facScale   = 0.3,
  mean.shape    = 0.6,      # literature-typical gamma gene-mean shape (Zappia et al. 2017)
  mean.rate     = 0.3,
  bcv.common    = 0.3,      # literature-typical BCV (Vieth et al. 2017)
  bcv.df        = 60,
  dropout.type  = "none",   # targeted imaging panel: negligible extra technical dropout
  seed          = seed
)

message(sprintf("[seed %d | %s] Simulating baseline counts via splatter...", seed, scenario))
sim <- splatSimulate(params, method = "groups", verbose = FALSE)

counts_mat <- as.matrix(counts(sim))          # genes x cells
gene_names <- paste0("Gene", seq_len(n_genes))
rownames(counts_mat) <- gene_names

## Map splatter's internal Batch -> sample_ID, Group -> region
cd <- as.data.frame(colData(sim))
sample_of_cell <- samples[as.integer(gsub("Batch", "", cd$Batch))]
region_of_cell <- regions[as.integer(gsub("Group", "", cd$Group))]
treat_of_cell  <- sample_group[sample_of_cell]

cell_meta <- data.frame(
  Cell        = colnames(counts_mat),
  sample_ID   = sample_of_cell,
  Treatment   = factor(treat_of_cell, levels = groups),
  region      = region_of_cell,
  stringsAsFactors = FALSE
)

## Region-dependent (but not treatment-dependent) cell-type composition --
## mirrors that real anatomical regions have different resident cell-type
## mixes (R1's "cell-type composition" component of a proper simulation).
## Cortex is given the highest Excitatory_Neuron share, consistent with the
## manuscript's own qualitative finding that isocortex is enriched for
## excitatory neurons relative to other regions (see header note); the
## remaining regions' relative orderings are a plausible, but not
## individually literature-cited, qualitative gradient -- the point being
## tested is that cell-type composition covaries with region (as in the
## real data), not the exact per-region percentages.
ct_prob_by_region <- list(
  Cortex       = c(Excitatory_Neuron = 0.60, Astrocyte = 0.25, Microglia = 0.15),
  Hippocampus  = c(Excitatory_Neuron = 0.50, Astrocyte = 0.30, Microglia = 0.20),
  Olfactory    = c(Excitatory_Neuron = 0.45, Astrocyte = 0.30, Microglia = 0.25),
  Striatum     = c(Excitatory_Neuron = 0.30, Astrocyte = 0.40, Microglia = 0.30),
  Cerebellum   = c(Excitatory_Neuron = 0.20, Astrocyte = 0.35, Microglia = 0.45)
)
cell_meta$cell_type <- vapply(cell_meta$region, function(r) {
  sample(cell_types, 1, prob = ct_prob_by_region[[r]])
}, character(1))

## ---------------------------------------------------------------------
## 3. Composition-engineering: impose an anatomical (Cortex) abundance
##    imbalance across treatment groups by calling Carter's ACTUAL ezy_seq
##    balance="sample" selection engine (_select_by_sample_targets -- the
##    literal function tag_region_abundance_by_FMT uses internally) directly,
##    via a Python subprocess bridge (compose_via_ezyseq_sim.py), rather than
##    a hand-translated R reimplementation of the same algorithm. An earlier
##    version of this file DID hand-translate it, and that reimplementation
##    silently diverged from the real algorithm (kept each sample's full pool
##    and only reshuffled its internal Cortex/other split, instead of
##    subsampling every sample down to a shared feasibility-capped N) while
##    its own comment incorrectly claimed the two matched -- see CLAUDE.md,
##    2026-07-20 "we really messed up" entry. Calling the real function via
##    subprocess removes that translation-fidelity risk entirely.
##
##    hi_frac/lo_frac are computed here in R (anchored to the manuscript's
##    own reported real isocortex statistic, 26.4% +/- 7.4% -- see the
##    baseline_roi_frac/baseline_roi_sd note above) and passed to the Python
##    side as pre-computed targets, bypassing tag_region_abundance_by_FMT's
##    own hi/lo derivation step (which would otherwise compute mean +/- SD
##    from the SYNTHETIC pool's own per-sample composition -- raw_roi_share
##    is a subsampling-headroom knob, not a realistic target, so that
##    derivation is not what we want here). Everything downstream of that --
##    the actual per-sample feasibility computation and subsampling -- is
##    Carter's unmodified code.
## ---------------------------------------------------------------------
hi_frac <- min(0.9, baseline_roi_frac + composition_dev * baseline_roi_sd)
lo_frac <- max(0.05, baseline_roi_frac - composition_dev * baseline_roi_sd)

script_dir_sim <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir_sim) == 0 || script_dir_sim == "") script_dir_sim <- "."

tmp_in  <- tempfile(fileext = ".csv")
tmp_out <- tempfile(fileext = ".csv")
write.csv(cell_meta[, c("Cell", "sample_ID", "Treatment", "region")], tmp_in, row.names = FALSE)

py_cmd <- sprintf("python3 %s %s %s %.6f %.6f %d",
                   shQuote(file.path(script_dir_sim, "compose_via_ezyseq_sim.py")),
                   shQuote(tmp_in), shQuote(tmp_out), hi_frac, lo_frac, seed)
message(sprintf("[seed %d | %s] Calling ezy_seq balance='sample' selection engine: %s",
                 seed, scenario, py_cmd))
status <- system(py_cmd)
if (status != 0 || !file.exists(tmp_out)) {
  stop("compose_via_ezyseq_sim.py failed -- see message above for the Python error.")
}

selected_cells <- read.csv(tmp_out, stringsAsFactors = FALSE, colClasses = "character")$Cell
keep_idx <- cell_meta$Cell %in% selected_cells
file.remove(tmp_in, tmp_out)

cell_meta  <- cell_meta[keep_idx, , drop = FALSE]
counts_mat <- counts_mat[, keep_idx, drop = FALSE]
rownames(cell_meta) <- cell_meta$Cell

message(sprintf("[seed %d | %s] Cortex fraction after composition-engineering: Treatment=%.3f, Control=%.3f",
                 seed, scenario,
                 mean(cell_meta$region[cell_meta$Treatment == "Treatment"] == roi),
                 mean(cell_meta$region[cell_meta$Treatment == "Control"]   == roi)))

## ---------------------------------------------------------------------
## 4. Ground-truth ID: designate true DE genes and per-gene NB dispersion
##    (method-of-moments, estimated on the pre-injection baseline so
##    injection does not bias its own reference dispersion).
## ---------------------------------------------------------------------
true_gene_idx <- sample(seq_len(n_genes), n_true_genes)
true_genes    <- gene_names[true_gene_idx]

gene_mean <- rowMeans(counts_mat)
gene_var  <- apply(counts_mat, 1, var)
phi       <- pmax((gene_var - gene_mean) / pmax(gene_mean^2, 1e-6), 1e-3)  # NB dispersion (1/size)

ground_truth <- data.frame(gene = gene_names, true_log2fc = 0, region_of_effect = NA_character_)
ground_truth$cell_type_marker_of <- NA_character_

## ---------------------------------------------------------------------
## 5. Effect-injection helper (used for both cell-type markers, 4b below,
##    and the scenario treatment effect, section 6).
##    NB resampling: new_count ~ NB(mu = group_mean * 2^log2FC, size = 1/phi_g)
## ---------------------------------------------------------------------
inject_effect <- function(counts_mat, cell_meta, gene, target_mask, log2fc, phi_g) {
  target_idx <- which(target_mask)
  if (length(target_idx) == 0) return(counts_mat)
  base_mean <- mean(counts_mat[gene, target_idx])
  if (base_mean <= 0) base_mean <- mean(counts_mat[gene, ]) + 0.1
  new_mu <- base_mean * (2 ^ log2fc)
  size_g <- 1 / phi_g
  counts_mat[gene, target_idx] <- rnbinom(length(target_idx), mu = new_mu, size = size_g)
  counts_mat
}

## ---------------------------------------------------------------------
## 4b. Cell-type marker-gene injection (see header note for literature
##     grounding). Marker genes are drawn from genes NOT selected as
##     treatment true-DE genes, so cell-type signal cannot contaminate the
##     treatment ground truth used for FDR/power/bias/RMSE/CI scoring.
##     Injected BEFORE the treatment effect (section 6) so it is present
##     identically in every scenario, independent of the treatment
##     manipulation -- cell-type composition/expression is a structural
##     feature of the simulated tissue, not a scenario-specific effect.
## ---------------------------------------------------------------------
nontreatment_gene_idx <- setdiff(seq_len(n_genes), true_gene_idx)
celltype_marker_idx   <- sample(nontreatment_gene_idx, n_celltype_marker_genes * length(cell_types))
celltype_marker_assignment <- split(
  gene_names[celltype_marker_idx],
  rep(cell_types, each = n_celltype_marker_genes)
)

for (ct in cell_types) {
  ct_mask <- cell_meta$cell_type == ct
  for (g in celltype_marker_assignment[[ct]]) {
    counts_mat <- inject_effect(counts_mat, cell_meta, g, ct_mask, celltype_log2fc, phi[g])
  }
  ground_truth$cell_type_marker_of[match(celltype_marker_assignment[[ct]], ground_truth$gene)] <- ct
}

## ---------------------------------------------------------------------
## 6. Inject the scenario's known treatment effect.
##    NB resampling: new_count ~ NB(mu = group_mean * 2^log2FC, size = 1/phi_g)
## ---------------------------------------------------------------------
is_treat <- cell_meta$Treatment == "Treatment"

if (scenario == "shared") {
  log2fc_true <- 1.0
  for (g in true_genes) {
    counts_mat <- inject_effect(counts_mat, cell_meta, g, is_treat, log2fc_true, phi[g])
  }
  ground_truth$true_log2fc[match(true_genes, ground_truth$gene)] <- log2fc_true
  ground_truth$region_of_effect[match(true_genes, ground_truth$gene)] <- "all"

} else if (scenario == "region_specific") {
  log2fc_true <- 1.5
  target_mask_base <- is_treat & (cell_meta$region == roi)
  for (g in true_genes) {
    counts_mat <- inject_effect(counts_mat, cell_meta, g, target_mask_base, log2fc_true, phi[g])
  }
  ground_truth$true_log2fc[match(true_genes, ground_truth$gene)] <- log2fc_true
  ground_truth$region_of_effect[match(true_genes, ground_truth$gene)] <- roi

} else if (scenario == "interaction") {
  log2fc_roi1 <- 1.5
  log2fc_roi2 <- -1.5
  mask_roi1 <- is_treat & (cell_meta$region == roi)
  mask_roi2 <- is_treat & (cell_meta$region == roi2)
  for (g in true_genes) {
    counts_mat <- inject_effect(counts_mat, cell_meta, g, mask_roi1, log2fc_roi1, phi[g])
    counts_mat <- inject_effect(counts_mat, cell_meta, g, mask_roi2, log2fc_roi2, phi[g])
  }
  ground_truth$true_log2fc[match(true_genes, ground_truth$gene)] <- log2fc_roi1  # primary (Cortex) effect recorded
  ground_truth$region_of_effect[match(true_genes, ground_truth$gene)] <- paste0(roi, "(+)/", roi2, "(-)")

} else if (scenario == "null") {
  ## no genes injected -- composition imbalance is the only confound present
  ground_truth$true_log2fc <- 0
  ground_truth$region_of_effect <- NA_character_
}

## Derived from true_log2fc (0 for every gene under "null" -- no genes are
## injected there, so true_genes is selected but never used), NOT from
## `gene %in% true_genes` directly. Using membership alone would flag the
## 100 selected-but-never-injected genes as "true DE" under the null
## scenario even though they behave identically to every other null gene,
## catastrophically corrupting null-scenario FDR/power (verified: produced
## an apparent ~90% "FDR" under null that was pure gene-mislabeling
## artifact, not a real false-discovery problem).
ground_truth$is_true_de <- abs(ground_truth$true_log2fc) > 1e-8

## ---------------------------------------------------------------------
## 6. Normalize (log1p of CPM-like scaling to 1e4, mirrors
##    ezy_seq::filter_and_normalize) and write outputs matching
##    DE_scripts::make_cfg() expected file layout.
## ---------------------------------------------------------------------
lib_size   <- colSums(counts_mat)
norm_mat   <- log1p(sweep(counts_mat, 2, lib_size, "/") * 1e4)

out_dir <- file.path(out_root, scenario, paste0("seed_", seed))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

raw_counts_df   <- as.data.frame(t(counts_mat))
logged_counts_df<- as.data.frame(t(norm_mat))
## A zero-column data.frame(row.names=...) writes a malformed CSV via
## write.csv (header "" but data rows "Gene1", with a trailing-comma phantom
## field), so header and data field counts disagree on read-back. Giving it
## one real column keeps header/data column counts consistent.
features_df     <- data.frame(gene_id = gene_names, row.names = gene_names)
metadata_df     <- data.frame(
  row.names   = cell_meta$Cell,
  sample_ID   = cell_meta$sample_ID,
  Treatment   = cell_meta$Treatment,
  napari_region = cell_meta$region,
  quint_region  = cell_meta$region,   # single simulated ontology stands in for both label sets
  cell_type     = cell_meta$cell_type,
  ct_simple     = cell_meta$cell_type,
  log_depth     = log1p(lib_size)
)
coords_df <- data.frame(row.names = cell_meta$Cell,
                         x = runif(nrow(cell_meta)), y = runif(nrow(cell_meta)))

write.csv(raw_counts_df,    file.path(out_dir, "raw_counts.csv"))
write.csv(logged_counts_df, file.path(out_dir, "normalized_counts.csv"))
write.csv(features_df,      file.path(out_dir, "features_counts.csv"))
write.csv(metadata_df,      file.path(out_dir, "cell_metadata.csv"))
write.csv(coords_df,        file.path(out_dir, "coords_xy.csv"))
write.csv(ground_truth,     file.path(out_dir, "ground_truth.csv"), row.names = FALSE)

message(sprintf("[seed %d | %s] Wrote %d cells x %d genes to %s",
                 seed, scenario, ncol(counts_mat), nrow(counts_mat), out_dir))
