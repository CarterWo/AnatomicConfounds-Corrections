#!/bin/bash
## DE suite for all 5 dev magnitudes x {g1,g2} using the NEW ezy_seq-based
## composition tagging + Seurat export. Array indices match
## build_seurat_ezyseq_array.sbatch EXACTLY (dev slowest-varying pairs of
## g1/g2) so this can be submitted with --dependency=aftercorr:<export_jobid>
## and each task will wait only for its OWN corresponding export task,
## not the whole export array.
#SBATCH --job-name=de_ezyseq
#SBATCH --partition=requeue
#SBATCH --array=0-9%4
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/de_ezyseq_%A_%a.out
#SBATCH --error=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/de_ezyseq_%A_%a.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

DEVS=(0.5 0.5 1.0 1.0 1.5 1.5 2.0 2.0 3.0 3.0)
SCENS=(g1 g2 g1 g2 g1 g2 g1 g2 g1 g2)
DEV=${DEVS[$SLURM_ARRAY_TASK_ID]}
SCEN=${SCENS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR=/home/nbhtd/AnatomicalRegions_iScience/IndependentValidation
SEURAT_INPUT_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/seurat_input_multi_ezyseq_dev${DEV}
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/de_results_multi_ezyseq_dev${DEV}

echo "Task $SLURM_ARRAY_TASK_ID -> dev=$DEV scenario=$SCEN"
Rscript "$SCRIPT_DIR/run_independent_de_multi_ezyseq.R" "$SCEN" "$SEURAT_INPUT_ROOT" "$RESULTS_ROOT"
