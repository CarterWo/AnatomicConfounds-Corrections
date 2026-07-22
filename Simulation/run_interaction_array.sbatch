#!/bin/bash
## SLURM array job: EXPLORATORY Treatment x region interaction model
## (de_functions.R::run_de_dream_interaction), added to test R1 comment 5's
## point that a random intercept for region cannot represent a treatment
## effect that itself varies by region. Only meaningful for the two
## scenarios where the ground truth places the effect within a specific
## region: region_specific and interaction (50 seeds each = 100 tasks).
##
## Does NOT re-run the manuscript's seven standard models -- those results
## already exist on disk from the original 200-task run
## (run_simulation_array.sbatch / job 14860147+14861495). Uses
## run_simulation_interaction_only.R, which regenerates only the raw counts
## (deleted after the original run to save disk) then fits the new model.
##
## Submit only from the login node (sbatch is a submission command, not
## compute); actual work runs on compute nodes.
#SBATCH --job-name=sim_interaction
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-99%20
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/interaction_%A_%a.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/interaction_%A_%a.err

set -e

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SCENARIOS=(region_specific interaction)
N_SEEDS=50

SCEN_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS + 1 ))
SCENARIO=${SCENARIOS[$SCEN_IDX]}

SIM_DIR=/home/nbhtd/AnatomicalRegions_iScience/Simulation
SIM_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_out
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_results

echo "Task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO seed=$SEED"

Rscript "$SIM_DIR/run_simulation_interaction_only.R" "$SEED" "$SCENARIO" "$SIM_ROOT" "$RESULTS_ROOT"
