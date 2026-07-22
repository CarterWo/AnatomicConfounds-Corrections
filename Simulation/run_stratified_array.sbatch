#!/bin/bash
## SLURM array job: region-stratified DE (R2 "no comparison with
## stratification"). For each of 4 scenarios x 20 seeds, subsets simulated
## cells by region and runs run_de_blind() within each region separately
## (de_functions.R, unmodified), so stratified power/FDR/FPR can be compared
## directly against the already-computed whole-dataset blind/aware/
## interaction models on the same ground truth.
##
## Does NOT re-run the manuscript's seven standard models or the interaction
## model -- those already exist on disk from the original array
## (run_simulation_array.sbatch) and run_interaction_array.sbatch.
##
## Submit only from the login node (sbatch is a submission command, not
## compute); actual work runs on compute nodes. Uses the requeue partition
## per the established fix for general-partition contention (see CLAUDE.md).
#SBATCH --job-name=sim_stratified
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-79%20
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/stratified_%A_%a.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/stratified_%A_%a.err

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
SIM_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_out
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_results

echo "Task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO seed=$SEED"

Rscript "$SIM_DIR/run_stratified_de.R" "$SEED" "$SCENARIO" "$SIM_ROOT" "$RESULTS_ROOT"
