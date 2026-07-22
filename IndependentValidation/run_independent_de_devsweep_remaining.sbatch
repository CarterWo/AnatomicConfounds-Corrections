#!/bin/bash
## Resumes ONLY seurat_quint + wilcoxon for one (dev, scenario) pair that
## timed out at 4h. Everything else (PB DESeq2, dream blind/napari/quint,
## seurat blind/napari) already saved and is NOT re-run.
## Usage: sbatch --export=DEV=2.0,SCEN=g1 run_independent_de_devsweep_remaining.sbatch
## For dev=1.0 (original paths, no _dev<X> suffix), pass DEV=1.0_orig.
#SBATCH --job-name=indep_de_remain
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/indep_de_remain_%j.out
#SBATCH --error=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/indep_de_remain_%j.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SCRIPT_DIR=/home/nbhtd/AnatomicalRegions_iScience/IndependentValidation

if [ "$DEV" == "1.0_orig" ]; then
  SEURAT_INPUT_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/seurat_input_multi
  RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/de_results_multi
else
  SEURAT_INPUT_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/seurat_input_multi_dev${DEV}
  RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/de_results_multi_dev${DEV}
fi

echo "dev=$DEV scenario=$SCEN -> $RESULTS_ROOT"
Rscript "$SCRIPT_DIR/run_independent_de_multi_remaining.R" "$SCEN" "$SEURAT_INPUT_ROOT" "$RESULTS_ROOT" 12
