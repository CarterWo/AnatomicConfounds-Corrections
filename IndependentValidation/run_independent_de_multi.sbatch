#!/bin/bash
## SLURM array job: runs the manuscript's unmodified DE suite (via
## run_independent_de_multi.R, with seurat_quint parallelized from the
## start -- see that script's header) on both composition-engineered
## scenarios (g1, g2) of the 4-animal-pooled Zhuang-ABCA-1/2/3/4 dataset.
#SBATCH --job-name=indep_de_multi
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/indep_de_multi_%A_%a.out
#SBATCH --error=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/indep_de_multi_%A_%a.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SCENARIOS=(g1 g2)
SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR=/home/nbhtd/AnatomicalRegions_iScience/IndependentValidation
SEURAT_INPUT_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/seurat_input_multi
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/de_results_multi

echo "Task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO"

Rscript "$SCRIPT_DIR/run_independent_de_multi.R" "$SCENARIO" "$SEURAT_INPUT_ROOT" "$RESULTS_ROOT"
