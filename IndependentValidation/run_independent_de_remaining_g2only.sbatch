#!/bin/bash
## g1 already completed successfully (job 14888972, task 0) -- this reruns
## ONLY g2's remaining steps (seurat_quint, wilcoxon), which timed out at
## exactly 3h. Bumping to 4h since g1 finished at 2:57:38, close to the
## wall, suggesting g2 needs only modestly more time, not a fundamentally
## different approach.
#SBATCH --job-name=indep_de_remaining_g2
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/indep_de_remaining_g2only_%j.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/indep_de_remaining_g2only_%j.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SCRIPT_DIR=/home/nbhtd/AnatomicalRegions_iScience/IndependentValidation
SEURAT_INPUT_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/seurat_input
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/de_results

Rscript "$SCRIPT_DIR/run_independent_de_remaining.R" g2 "$SEURAT_INPUT_ROOT" "$RESULTS_ROOT"
