#!/bin/bash
## Runs the full DE suite (run_independent_de_multi.R, seurat_quint
## parallelized from the start) for one dev level's g1+g2 scenarios.
## Usage: sbatch --export=DEV=1.5 run_independent_de_devsweep.sbatch
#SBATCH --job-name=indep_de_dev
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/indep_de_dev_%A_%a.out
#SBATCH --error=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/indep_de_dev_%A_%a.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SCENARIOS=(g1 g2)
SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR=/home/nbhtd/AnatomicalRegions_iScience/IndependentValidation
SEURAT_INPUT_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/seurat_input_multi_dev${DEV}
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/de_results_multi_dev${DEV}

echo "dev=$DEV task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO"

Rscript "$SCRIPT_DIR/run_independent_de_multi.R" "$SCENARIO" "$SEURAT_INPUT_ROOT" "$RESULTS_ROOT"
