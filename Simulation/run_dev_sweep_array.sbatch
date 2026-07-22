#!/bin/bash
## SLURM array job: composition-imbalance MAGNITUDE sweep (R1 Question 2:
## "test several levels of imbalance, not only ±1 standard deviation...
## show whether the observed bias increases smoothly with regional imbalance
## and whether the proposed adjustment remains stable under mild and severe
## perturbations"). This is a SYNTHETIC/simulation-side complement to
## Carter's real-data composition-engineering resampling work (Weakness 2 /
## Question 2) -- not a duplicate of it.
##
## 3 dev levels (0.5 mild, 1.5 moderate-severe, 2.0 severe -- 1.0 original
## already covered by the main 200-task array) x 2 scenarios (null: does FPR
## stay controlled as imbalance grows; region_specific: does power stay
## stable) x 20 seeds = 120 tasks.
##
## Submit only from the login node (sbatch is a submission command, not
## compute); actual work runs on compute nodes.
#SBATCH --job-name=sim_devsweep
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-119%20
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/devsweep_%A_%a.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/devsweep_%A_%a.err

set -e   # propagate R script failures instead of silently "completing" the array task

module load r/4.4.0
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

DEVS=(0.5 1.5 2.0)
SCENARIOS=(null region_specific)
N_SEEDS=20

## Task index decomposition: dev is the slowest-varying, then scenario, then seed.
N_SCEN=${#SCENARIOS[@]}
TASKS_PER_DEV=$(( N_SCEN * N_SEEDS ))

DEV_IDX=$(( SLURM_ARRAY_TASK_ID / TASKS_PER_DEV ))
REM=$(( SLURM_ARRAY_TASK_ID % TASKS_PER_DEV ))
SCEN_IDX=$(( REM / N_SEEDS ))
SEED=$(( REM % N_SEEDS + 1 ))

DEV=${DEVS[$DEV_IDX]}
SCENARIO=${SCENARIOS[$SCEN_IDX]}

SIM_DIR=/home/nbhtd/AnatomicalRegions_iScience/Simulation
## Distinct root per dev level -- simulate_ground_truth.R does not
## namespace by dev itself, so this sbatch script handles the separation.
SIM_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_out_devsweep/dev_${DEV}
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_results_devsweep/dev_${DEV}

echo "Task $SLURM_ARRAY_TASK_ID -> dev=$DEV scenario=$SCENARIO seed=$SEED"

Rscript "$SIM_DIR/simulate_ground_truth.R" "$SEED" "$SCENARIO" "$SIM_ROOT" "$DEV"
Rscript "$SIM_DIR/run_simulation_de.R"    "$SEED" "$SCENARIO" "$SIM_ROOT" "$RESULTS_ROOT"

## Free disk: raw/normalized count CSVs are large and no longer needed once
## DE has run and ground_truth.csv has been copied into sim_results/.
rm -f "$SIM_ROOT/$SCENARIO/seed_$SEED/raw_counts.csv" "$SIM_ROOT/$SCENARIO/seed_$SEED/normalized_counts.csv"
