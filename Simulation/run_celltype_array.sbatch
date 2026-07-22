#!/bin/bash
## Cell-type-aware ground-truth simulation array (R1's explicit simulation
## requirement: "region-specific baseline expression, differences in region
## abundance, cell-type composition, sample-level variability, and
## treatment-by-region interactions"). simulate_ground_truth.R now gives
## cell type (Excitatory_Neuron/Astrocyte/Microglia, matching the
## manuscript's own analyzed strata) real marker-gene expression signal
## (Zhang et al. 2014, literature-grounded, not fit to our data), in
## addition to the region-conditional composition it already had.
##
## 4 scenarios x 20 seeds = 80 tasks, separate output roots from the
## original 200-task array so neither set of results is disturbed.
#SBATCH --job-name=sim_celltype
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-79%20
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/celltype_%A_%a.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/celltype_%A_%a.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SCENARIOS=(null shared region_specific interaction)
N_SEEDS=20

SCEN_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS + 1 ))
SCENARIO=${SCENARIOS[$SCEN_IDX]}

SIM_DIR=/home/nbhtd/AnatomicalRegions_iScience/Simulation
SIM_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_out_celltype
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_results_celltype

echo "Task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO seed=$SEED"

Rscript "$SIM_DIR/simulate_ground_truth.R" "$SEED" "$SCENARIO" "$SIM_ROOT"
Rscript "$SIM_DIR/run_simulation_de.R" "$SEED" "$SCENARIO" "$SIM_ROOT" "$RESULTS_ROOT"

## Free disk -- same reasoning as run_simulation_array.sbatch.
rm -f "$SIM_ROOT/$SCENARIO/seed_$SEED/raw_counts.csv" "$SIM_ROOT/$SCENARIO/seed_$SEED/normalized_counts.csv"
