# Variance Components

Per-gene variance components for the two real datasets (CosMx and the Zhuang-ABCA-1-4
MERFISH data), added for the second revision. Each gene is fit with a linear mixed model
using the same covariates as the Dream models (log_depth, plus Treatment for CosMx, as
fixed effects; cell_type, region and sample_ID as random intercepts), and each term's
share of the total variance is reported. Summarized in Supplementary Table 3.

CosMx uses 48,000 cells from the six Stroke-FMT and Healthy-FMT samples: olfactory bulb
cells are removed first, as in `Pre_processing/Pre_DE_processing.ipynb`, and then 8,000 cells
are sampled per sample (seed 0). MERFISH uses the 20,000-cell subset from
the 1.0x external-validation run.

## Files

- `export_cosmx_for_variance.py` - builds the CosMx input (drops Cntrl and olfactory bulb
  cells, then samples 8,000 cells per sample). `export_cosmx.sbatch` runs it on the cluster.
- `variance_components.R` - fits every gene and writes one CSV per dataset/label set.
- `run_variance_components.sbatch` - SLURM array for the four runs (CosMx and MERFISH,
  Napari and QUINT labels).
- `check_marker_genes.R` - compares the reported region share for a few marker genes
  against an ML fit, a region-only model and one-way ANOVA.
- `make_supp_table.py` - builds Supplementary Table 3 from `results/`
  (`python3 make_supp_table.py results Supplementary_Table_3.xlsx`).
- `results/` - per-gene output (`vc_<dataset>_<labels>.csv`); values are fractions of
  total variance.

Note: `variancePartition::fitExtractVarPartModel` gave zero region variance for every gene
with the lme4 version on our cluster (1.1-35.5), so the fractions are computed directly from
`lme4::VarCorr`.
