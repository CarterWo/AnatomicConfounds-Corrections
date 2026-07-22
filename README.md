# Spatial Transcriptomics Analysis Pipeline

This repository contains the full analysis pipeline for spatial transcriptomics data generated on the CosMx SMI platform. The pipeline moves from raw CosMx exports through preprocessing, differential expression (DE), and post-DE comparison.

The pipeline runs over **two independent datasets ("frameworks")** — see [Two data frameworks](#two-data-frameworks) below.

---

## Repository Structure

```
├── EzySeq_Library/          # Custom Python library (ezy_seq)
├── Pre_processing/          # Data loading, QC, annotation, region labeling,
│                            #   and Composition_Engineering.ipynb (composition step)
├── DE_scripts/              # Differential expression (R scripts for HPC + notebooks)
├── Post_DE_Processing/      # DE result comparison and meta-analysis
├── Visualization_Scripts/   # UMAP and volcano plot generation
├── Quint_Scripts/           # Quint atlas region assignment (R)
│   └── QUINT_atlas_maps/{genotype,fmt}/   # per-framework segmentation maps
├── napari_definitions/{genotype,fmt}/     # per-framework Napari sample/region polygons
├── config.yaml              # per-framework paths + composition settings
└── Final_Data/              # Processed outputs (large files on Zenodo)
```

Per-framework outputs land in labelled folders so the two datasets never
overwrite each other: `Final_Data/<fw>/`, `csvs/<fw>/`, `LMM_output/<fw>/`.

---

## Two data frameworks

The pipeline is configured for two datasets, each with its own raw CosMx export,
`sample_info` workbook, Napari polygons, QUINT maps, and **labelled** outputs.
Both are defined under `frameworks:` in `config.yaml`:

| Framework  | `sample_info`                | Contrast (`group_col`)         | Cohort filter      |
|------------|------------------------------|--------------------------------|--------------------|
| `genotype` | `sample_info_genotype.xlsx`  | `Genotype` — E4 vs E2 (ref E2) | `FMT == MCI`       |
| `fmt`      | `sample_info_FMT.xlsx`       | `FMT` — Stroke_FMT vs Healthy_FMT (ref Healthy_FMT) | none (all samples) |

**Selecting a framework per run** (default is `config.yaml → active_framework`):

- **Notebooks** (`Pre_DE_processing.ipynb`, `Composition_Engineering.ipynb`): set
  the `FRAMEWORK` variable in the first cell, or the `PIPELINE_FRAMEWORK` env var.
- **DE R scripts**: `Rscript DE_scripts/LMM_all.R --framework fmt` (or `--framework genotype`);
  also honours the `DE_FRAMEWORK` env var. The QUINT `.Rmd` uses the same `FRAMEWORK`/`DE_FRAMEWORK`.
- **Post-DE**: `python Post_DE_Processing/aggregate_seeds.py --framework fmt` (or `PIPELINE_FRAMEWORK`).

Run the *whole* pipeline once per framework — preprocess → composition → DE →
aggregate with the same framework selected each time. The `fmt` framework's `FMT`
column also carries a third `Cntrl` arm; with its `cohort_filter` empty those
cells stay in the object. Set `cohort_filter: {FMT: [Stroke_FMT, Healthy_FMT]}`
to exclude `Cntrl` from the contrast.

> **Note:** the raw stroke (`fmt`) export path in `config.yaml` is a placeholder —
> fill `frameworks.fmt.inputs.cosmx_export_dir` in for your machine.

---

## Requirements

### Python
- `scanpy`, `anndata`, `numpy`, `pandas`, `matplotlib`, `scipy`, `napari`
- `ezy_seq` (install from `EzySeq_Library/mypythonlibrary/`):
  ```bash
  pip install -e EzySeq_Library/mypythonlibrary/
  ```

### R
- `Seurat`, `variancePartition`, `BiocParallel`, `DESeq2`, `lme4`, `limma`, `edgeR`

---

## Pipeline Overview

### Step 1 — Preprocessing (`Pre_processing/Pre_DE_processing.ipynb`)

This notebook takes raw CosMx exports and produces annotated, normalized AnnData objects ready for DE.

**1a. Load raw data**
```python
import ezy_seq as ezy
adatas = ezy.load.cosmx("/path/to/cosmx/export")
meta_dictionary = ezy.read_dictionary("sample_info.xlsx")
```
`ezy.load.cosmx` reads all CosMx slide exports in a directory. `ezy.read_dictionary` reads the sample metadata Excel file mapping sample IDs to experimental groups.

**1b. Assign sample IDs via Napari polygons**

Hand-drawn polygon CSVs exported from Napari are used to assign cells to samples. Each CSV is named by sample ID and contains `axis-0`/`axis-1` vertex columns.

**1c. Filter and normalize**
```python
annotated = ezy.apply_annotation(adatas, meta_dictionary, cell_type_key="<seurat_cluster_col>")
normalized = ezy.filter_and_normalize(annotated, min_gene_cnt=20, min_t_cnt=100)
adata_full = sc.concat(normalized, join='outer', merge='first', index_unique='-')
```
Cells with fewer than 100 total transcripts or fewer than 20 detected genes are excluded. Negative control probe transcripts are removed, and cells where negative probe counts exceed 10% of total transcripts are discarded. AtoMx-derived QC flags are applied alongside log-normalization.

**1d. Assign anatomical regions**

Two region annotation strategies are used:

- **Napari** (`napari_region`): polygon CSVs drawn per brain region are used to label each cell via point-in-polygon testing.
- **Quint** (`quint_region`): atlas-based region assignments generated by the Quint pipeline (`Quint_Scripts/Quint_Pipeline.Rmd`), imported back into Python within `Pre_DE_processing.ipynb`.

**1e. Create cell type labels**
```python
adata_full.obs['ct_simple'] = ...  # e.g., "Astrocytes" from "Astrocytes.cortex.hippocampus"
adata_full.obs['ct_napari_region'] = adata_full.obs['ct_simple'] + "_" + adata_full.obs['napari_region']
adata_full.obs['ct_quint_region']  = adata_full.obs['ct_simple'] + "_" + adata_full.obs['quint_region']
```

The preprocessing notebook ends by writing the fully-annotated cohort to `outputs.adata_h5ad` (config.yaml). Composition-engineering has moved out of this notebook into its own step, below.

---

### Step 1½ — Composition-Engineering (`Pre_processing/Composition_Engineering.ipynb`)

Loads the annotated cohort, restricts it to the analysis cohort (drops the Olfactory bulb, keeps the MCI FMT group), tags the composition-engineering scenarios, and exports the single cohort (with the `G1_*`/`G2_*` masks) to `outputs.de_export_base` for the DE scripts.

Every experiment parameter lives in the selected framework's `composition` block
of **config.yaml** (`frameworks.<name>.composition` — single source of truth for
both this step and the DE scripts). For the `genotype` framework:

```yaml
frameworks:
  genotype:
    composition:
      group_col: "Genotype"    # contrasted variable (E4 vs E2)
      up_group: "E4"           # cortex-enriched group in G1 (depleted in G2)
      down_group: "E2"
      reference: "E2"          # DE reference level -> coefficients report E4 vs E2
      cohort_filter: {FMT: ["MCI"]}   # kept before the contrast ({} = no restriction)
      drop_regions: ["Olfactory"]     # regions removed from the cohort
      region_of_interest: "Cortex"
      balance: "sample"
      deviations: [1.5, 1.0, 0.5, 0.25]
      deseq2_iterations: 100   # iterations tagged per deviation; DESeq2 uses all
      dream_iterations: 3      # Dream LMM uses only the first 3 per deviation
```

The `cohort_filter` (`{obs_col: [allowed, ...]}`) and `drop_regions` are applied
in `Composition_Engineering.ipynb` before tagging; the `fmt` framework sets
`cohort_filter: {}` (keeps all samples).

`tag_region_abundance_by_FMT` shifts the `region_of_interest` proportion to +dev·SD or −dev·SD of the cross-sample mean (`dev=1` reproduces the original ±1 SD), redistributing the remaining cell budget across non-target regions proportionally. Rather than returning subsampled objects, it tags each cell's membership in the two opposite scenarios (G1/G2) as boolean columns named `G1_{seed}_{dev}` / `G2_{seed}_{dev}`. The step tags `deseq2_iterations` iterations (random_state `0..N-1`) for every deviation magnitude. With `balance="sample"` each sample is subsampled independently to the same maximised on-target cell count; `balance="fmt"` pools cells within each group instead. Only the full cohort is exported (the masks ride along in `cell_metadata.csv`); the DE script `LMM_all.ipynb` reconstructs the cortex-up / cortex-down subsets from those masks rather than reading separate CSV folders.

The original `ezy.set_region_abundance_by_FMT(adata, target_by_fmt, total_per_fmt)`, which takes explicit target dicts and returns a subsampled AnnData, remains available. (`tag_region_abundance_by_FMT` keeps its legacy `fmt_col`/`up_fmt`/`down_fmt` argument names, now driven by the config's `group_col`/`up_group`/`down_group`.)

---

### Step 2 — Visualization (`Visualization_Scripts/UMAPS.ipynb`)

```python
import ezy_seq as ezy

ezy.plot.umap(adata_full, color_by="ct_simple", groups=[("FMT", ["Healthy", "MCI"])])

de_results = ezy.rank_DE(adata_full, groupby="FMT", comparisons=[["Healthy", "MCI"]])
ezy.plot.volcano(de_results, y_axis="score", top_labels=30)
```

UMAP embeddings are generated in Scanpy (`sc.pp.neighbors` + `sc.tl.umap`). Figures are produced using matplotlib.

---

### Step 3 — Differential Expression

Two analyses run DE on different datasets. Each ships as both a notebook (for interactive use) and a standalone **`.R` script** that runs straight through on an HPC node — the scripts are the recommended entry point:

| Analysis | HPC script / notebook | Dataset | Purpose |
|----------|----------|---------|---------|
| Baseline | `DE_Base_Analysis.R` / `.ipynb` | Full cohort (no composition engineering) | Baseline DE results → feeds post-hoc comparison |
| Composition | `LMM_all.R` / `.ipynb` | Composition-engineered subsets (cortex-up / cortex-down), reconstructed from the export's `G1_*`/`G2_*` masks, per iteration | Quantify anatomical composition effects |

Both run in **R** and are configured for **two methods only** (DESeq2 and the Dream LMM):

| Model | Method | Annotation variants |
|-------|--------|-------------------|
| DREAM LMM | `variancePartition::dream` | blind, napari, quint |
| DESeq2 | Pseudobulk, Wald test | blind |

The contrast variable is the selected framework's `composition.group_col` (renamed to `Treatment` in-script): **Genotype** (E4 vs E2, ref E2) for `genotype`, **FMT** (Stroke_FMT vs Healthy_FMT, ref Healthy_FMT) for `fmt`. Raw counts are TMM-normalized via `edgeR::calcNormFactors` before DREAM.

**Iteration design (`LMM_all.ipynb`).** For each deviation magnitude in the framework's `composition.deviations`, DESeq2 runs on all `deseq2_iterations` (100) tagged iterations while the Dream LMM runs on only the first `dream_iterations` (3). Results are written to per-iteration subfolders so nothing overwrites:
```
<lmm_results_dir>/dev_<dev>/seed_<seed>/{Pseudobulk_Validation,Local_Regional_Analysis,Global_CT_Analysis}/
    DESEQ2{UP|DOWN}_{celltype}_PB.csv               # DESeq2 pseudobulk  (all 100 iters)
    {UP|DOWN}_{celltype}_dream_{blind|napari|quint}.csv   # Dream LMM     (first 3 iters)
```
`DE_Base_Analysis.ipynb` runs once (no iteration) and writes to `<lmm_results_dir>/base/{...}` with the same filename convention.

**Running on HPC.** The `.R` scripts run non-interactively via `Rscript` and share `DE_scripts/de_common.R` (libraries, config discovery, DE model functions). They locate `config.yaml` regardless of the working directory (searching upward from the script, or via the `DE_CONFIG` env var / `--config`) and resolve its relative paths against the config's directory, and they read worker counts from `$SLURM_CPUS_PER_TASK`.

```bash
# Straight through (all deviations, all seeds); pick the framework/dataset
Rscript DE_scripts/DE_Base_Analysis.R --framework genotype
Rscript DE_scripts/LMM_all.R          --framework genotype
Rscript DE_scripts/LMM_all.R          --framework fmt        # the other dataset

# One array-task slice (e.g. a single deviation, a seed range, N workers)
Rscript DE_scripts/LMM_all.R --framework fmt --devs 1 --seeds 0-49 --workers 16
```

`LMM_all.R` options (defaults from the selected `frameworks.<name>.composition`): `--config`, `--framework`, `--workers`, `--devs "1.5,1,0.5,0.25"`, `--seeds "0-99"`, `--dream-seeds N`. Omitting `--framework` uses `config.yaml → active_framework` (or the `DE_FRAMEWORK` env var). Because results go to per-`dev`/`seed` subfolders, array tasks never collide. Example SLURM templates: `DE_scripts/de_base.sbatch` (single job) and `DE_scripts/lmm_all.sbatch` (a job array, one task per deviation) — adapt the `#SBATCH` and module/env lines to your cluster.

---

### Step 4 — Post-DE Processing (`Post_DE_Processing/`)

**4a. Aggregate iterations (`aggregate_seeds.py`).** `LMM_all.ipynb` writes one result file per composition-engineering iteration (`dev_<dev>/seed_<seed>/…`). This step collapses the seeds into one summary per analysis:

```bash
python Post_DE_Processing/aggregate_seeds.py --framework genotype   # per framework
```

For each `dev_<dev>` it groups the identically named files across `seed_*` folders and writes `<lmm_results_dir>/aggregated/dev_<dev>/<subdir>/<file>.csv`. Because the seeds are re-subsamples of the *same* cells (not independent replicates), p-values are **not** pooled (Fisher/Stouffer would be invalid); instead each gene is summarized by the distribution of its estimate across seeds:

- `logFC` — across-seed mean (with `logFC_median`, `logFC_sd`, and a 2.5–97.5% band `logFC_lo`/`logFC_hi` = composition-induced spread)
- `sig_frac` — fraction of seeds calling the gene significant (`adj.P.Val < alpha`) — the selection frequency / robustness of the DE call
- `dir_consistency` — fraction of seeds agreeing on the logFC sign
- `P.Value`/`adj.P.Val` — median across seeds (a location summary, not a combined test); `n_seeds`, `present_frac`

The aggregated tree mirrors a single per-seed folder, so the comparison scripts below can be pointed at `aggregated/dev_<dev>/` unchanged.

**4b. Comparison scripts.** Compare DE results across models and annotation strategies, plus a shared utility module:

| Script | Purpose |
|--------|---------|
| `aggregate_seeds.py` | Collapse per-seed iterations into per-analysis summaries (Step 4a) |
| `consolidate_summary_table.py` | Aggregate DE results into a single summary table |
| `visualize_anatomic_inclusion.py` | Compare blind vs. anatomically-aware models (MAB, Bias Ratio) |
| `visualize_g1_g2_comparison.py` | Compare composition-engineered datasets (cortex-up vs. cortex-down) |
| `visualize_wilcoxon_deseq2.py` | Visualize DESeq2 DE results |
| `plot_utils.py` | Shared plotting utilities **+ the framework resolver** (`resolve_results_root`) |

All of these are **framework-aware** via `plot_utils.resolve_results_root()`: select the dataset with `PIPELINE_FRAMEWORK=fmt` (or `--framework fmt`), and each script reads that framework's `lmm_results_dir` and writes figures to `Post_DE_Processing/figures/<framework>/`. The G1/G2 scripts read one aggregated deviation folder (default `aggregated/dev_1`; override with the `LMM_SUBPATH` env var), and `visualize_anatomic_inclusion.py` reads the framework's `base/` results.

Mean Absolute Bias (MAB) and Bias Ratio metrics are computed via `ezy.impact_metrics` to quantify the contribution of anatomical labels to DE effect sizes.

---

## ezy_seq Library Reference

| Function | Description |
|----------|-------------|
| `ezy.load.cosmx(path)` | Load all CosMx slide exports from a directory |
| `ezy.read_dictionary(xlsx)` | Read sample metadata from Excel |
| `ezy.apply_annotation(adatas, meta, cell_type_key)` | Map cluster + sample metadata onto AnnData list |
| `ezy.filter_and_normalize(adatas, min_gene_cnt, min_t_cnt)` | Filter by gene/transcript counts, log-normalize |
| `ezy.combine_adatas(out_path, adatas)` | Concatenate and save AnnData list |
| `ezy.set_region_abundance_by_FMT(adata, target_by_fmt, total_per_fmt)` | Composition-engineering: subsample to target regional proportions per treatment group |
| `ezy.tag_region_abundance_by_FMT(adata, dev, random_state, balance)` | Composition-engineering: tag G1/G2 boolean columns for ±dev·SD cortex-shift scenarios (per-sample or pooled) |
| `ezy.rank_DE(adata, groupby, comparisons)` | Wilcoxon DE across one or more group pairs |
| `ezy.impact_metrics(naive_df, anat_df, region_markers)` | MAB and Bias Ratio: quantify anatomical label contribution to DE |
| `ezy.plot.volcano(de_df, ...)` | Volcano plot with optional gene labeling |
| `ezy.plot.umap(adata, ...)` | UMAP plots with group filtering |
| `ezy.plot.rank_and_plot(adata, ...)` | Rank genes and plot top features |
| `ezy.plot.feats_bar(de_df, ...)` | Horizontal bar chart of top DE features |
| `ezy.lmm.*` | DE result loading, correlation, overlap, and concordance utilities |
| `ezy.lmm.aggregate_seed_files(files, model_type, alpha)` | Collapse per-seed DE files into one per-gene summary (mean effect, selection frequency, spread) |
| `ezy.lmm.aggregate_seed_directory(results_root, ...)` | Walk `dev_*/seed_*/` and aggregate every analysis across iterations |
