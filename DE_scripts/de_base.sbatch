#!/bin/bash
# =============================================================================
# de_base.sbatch — baseline DE (DE_Base_Analysis.R) as a single SLURM job.
# Submit from the repo root:   sbatch DE_scripts/de_base.sbatch
# Pick a framework:   sbatch --export=ALL,FRAMEWORK=fmt DE_scripts/de_base.sbatch
#   (unset -> config.yaml active_framework)
# Adapt the #SBATCH lines (partition/account/time/mem) to your cluster.
# =============================================================================
#SBATCH --job-name=de_base
#SBATCH --output=logs/de_base_%j.out
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G

set -euo pipefail
mkdir -p logs

# Make R available (edit for your cluster: an environment module or a conda env)
RSCRIPT="${RSCRIPT:-/home/cdw5fz/.conda/envs/spatial_env/bin/Rscript}"

# Repo root = directory containing config.yaml (defaults to the submit dir).
REPO="${REPO:-$SLURM_SUBMIT_DIR}"

# Framework (dataset/contrast); empty -> config.yaml active_framework.
FRAMEWORK="${FRAMEWORK:-}"

# Trailing "$@" forwards any extra sbatch arguments through to R, so
#   sbatch DE_scripts/de_base.sbatch --resume
# actually reaches DE_Base_Analysis.R. Without it the flag is silently dropped.
# --models deseq2,dream: the baseline runs the sample-aware families only. The
# cell-level Wilcoxon/LR tests are deliberately excluded here. Override on the
# command line if you want them back:
#   sbatch DE_scripts/de_base.sbatch --models deseq2,dream,wilcox,lr
"$RSCRIPT" "$REPO/DE_scripts/DE_Base_Analysis.R" \
    ${FRAMEWORK:+--framework "$FRAMEWORK"} \
    --models deseq2,dream \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
