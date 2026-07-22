#!/bin/bash
## Targeted rerun of the 6 seed/scenario pairs that hit a transient
## BiocParallel::SnowParam port-collision race in job 14894261 (1-2 of 5
## regions silently skipped per affected task, while the task still logged
## overall "DONE"). Reruns the FULL seed/scenario pair (all 5 regions,
## cheap/fast) rather than just the missing region, matching the project's
## established fix for this exact failure mode (see CLAUDE.md).
#SBATCH --job-name=sim_stratified_rerun
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/stratified_rerun_%A_%a.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/stratified_rerun_%A_%a.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SEEDS=(8 14 14 17 5 9)
SCENARIOS=(shared shared region_specific shared null null)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

SIM_DIR=/home/nbhtd/AnatomicalRegions_iScience/Simulation
SIM_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_out
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_results

echo "Rerun task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO seed=$SEED"

Rscript "$SIM_DIR/run_stratified_de.R" "$SEED" "$SCENARIO" "$SIM_ROOT" "$RESULTS_ROOT"
