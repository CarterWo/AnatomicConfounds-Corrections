#!/bin/bash
## SLURM array job: runs the manuscript's unmodified DE suite (via
## run_independent_de.R) on both composition-engineered scenarios (g1, g2)
## of the independent Zhuang-ABCA-1 MERFISH dataset.
##
## Submitted via sbatch (not srun) so the job survives local session/SSH
## drops -- an earlier interactive srun smoke test was orphaned when the
## local driving process was killed mid-queue-wait (2026-07-09). sbatch
## jobs run fully independently of any client connection, same as the
## ground-truth simulation array.
##
## quint_region has ~306 distinct CCF `structure` categories in this
## dataset (vs. ~36 in the manuscript's real QUINT taxonomy) -- the
## dummy-coded per-gene logistic regression (seurat_quint) is genuinely
## slow, not stuck (confirmed via direct compute-node process check:
## R at 69% CPU, still actively computing). A first srun attempt with
## --time=00:40:00 was killed by the time limit mid-seurat_quint, before
## Wilcoxon even ran. A second attempt (sbatch, --time=02:00:00, job
## 14880818) ALSO timed out at exactly 2:00:07 for both g1 and g2, still
## stuck in seurat_quint (dream blind/napari/quint, DESeq2 PB, and
## seurat_blind/napari had all already saved by then) -- confirms
## seurat_quint alone needs well over 1hr on top of everything before it.
## --time=04:00:00 here to give real headroom past both seurat_quint and
## Wilcoxon.
#SBATCH --job-name=indep_de
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/indep_de_%A_%a.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/indep_de_%A_%a.err

set -e   # propagate R script failures instead of silently "completing" the array task

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

Rscript "$SCRIPT_DIR/run_independent_de.R" "$SCENARIO" "$SEURAT_INPUT_ROOT" "$RESULTS_ROOT"
