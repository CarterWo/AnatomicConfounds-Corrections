#!/bin/bash
## SLURM array job: runs ONLY the two DE steps (seurat_quint, wilcoxon)
## that never completed in run_independent_de.R across two prior attempts
## (job 14880818, --time=02:00:00, TIMEOUT; job 14882353, --time=04:00:00,
## TIMEOUT) -- dream_blind/napari/quint, DESeq2 PB, and seurat_blind/napari
## already completed and saved successfully for both g1 and g2 in those
## runs and are NOT re-run here.
##
## seurat_quint uses run_seurat_lr_parallel (gene-chunked across
## MulticoreParam workers) instead of the serial run_seurat_lr -- same
## FindMarkers(test.use="LR") statistical test, just parallelized across
## the 8 requested CPUs, since the serial version confirmed genuinely too
## slow (4h+, zero completions) for quint_region's 306-level factor at
## 20K-cell scale.
#SBATCH --job-name=indep_de_remaining
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/indep_de_remaining_%A_%a.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/indep_de_remaining_%A_%a.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SCENARIOS=(g1 g2)
SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR=/home/nbhtd/AnatomicalRegions_iScience/IndependentValidation
SEURAT_INPUT_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/seurat_input
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/de_results

echo "Task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO"

Rscript "$SCRIPT_DIR/run_independent_de_remaining.R" "$SCENARIO" "$SEURAT_INPUT_ROOT" "$RESULTS_ROOT"
