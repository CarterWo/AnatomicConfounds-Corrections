# Variance Components

Per-gene variance components for the two real datasets (CosMx and the Zhuang-ABCA-1-4
MERFISH data), added for the second revision. Each gene is fit with a linear mixed model
using the same covariates as the Dream models (log_depth, plus Treatment for CosMx, as
fixed effects; cell_type, region and sample_ID as random intercepts), and each term's
share of the total variance is reported. Summarized in Supplementary Table 3.

CosMx uses the 6-sample export (8,000 cells per sample) with olfactory bulb cells removed,
as in `Pre_processing/Pre_DE_processing.ipynb`. MERFISH uses the 20,000-cell subset from
the 1.0x external-validation run.

## Files

- `variance_components.R` - fits every gene and writes one CSV per dataset/label set.
- `run_variance_components.sbatch` - SLURM array for the four runs (CosMx and MERFISH,
  Napari and QUINT labels).
- `check_marker_genes.R` - compares the reported region share for a few marker genes
  against an ML fit, a region-only model and one-way ANOVA.
- `make_supp_table.py` - builds Supplementary Table 3 from `results/`
  (`python3 make_supp_table.py results Supplementary_Table_3.xlsx 46271`).
- `results/` - per-gene output (`vc_<dataset>_<labels>.csv`); values are fractions of
  total variance.

Note: `variancePartition::fitExtractVarPartModel` gave zero region variance for every gene
with the lme4 version on our cluster (1.1-35.5), so the fractions are computed directly from
`lme4::VarCorr`.
