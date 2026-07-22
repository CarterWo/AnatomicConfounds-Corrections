#!/bin/bash
## SLURM array job: ground-truth simulation for Reviewer 1 (iScience revision).
## 4 scenarios x 50 seeds = 200 tasks. Each task simulates one dataset and runs
## the full existing DE pipeline (Dream blind/napari/quint, DESeq2 pseudobulk,
## Seurat LR blind/napari/quint, Wilcoxon) against it, unmodified.
##
## Submit only from the login node (sbatch is a submission command, not
## compute); actual work runs on compute nodes.
#SBATCH --job-name=sim_gt
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-199%20
#SBATCH --output=/home/nbhtd/AnatomicalRegions_iScience/logs/sim_%A_%a.out
#SBATCH --error=/home/nbhtd/AnatomicalRegions_iScience/logs/sim_%A_%a.err

set -e   # propagate R script failures instead of silently "completing" the array task

module load r/4.4.0
## See install_packages.sbatch: newer libstdc++ needed by farver/Bioconductor deps.
## gcc module only sets build-time paths, so LD_LIBRARY_PATH must be set explicitly.
module load gcc/12.3.0
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

SCENARIOS=(null shared region_specific interaction)
N_SEEDS=50

SCEN_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS + 1 ))
SCENARIO=${SCENARIOS[$SCEN_IDX]}

SIM_DIR=/home/nbhtd/AnatomicalRegions_iScience/Simulation
## Raw/normalized count CSVs are large (~20K cells x 1000 genes each) and up to
## 20 tasks run concurrently -- the home filesystem quota is small and shared
## with other lab members' data (hit "Disk quota exceeded" writing here on the
## first full-array attempt, 2026-07-07). ~/data (-> /mnt/pixstor/data/nbhtd,
## 37T free) is the large-data mount; use it instead.
SIM_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_out
RESULTS_ROOT=/home/nbhtd/data/AnatomicalRegions_iScience/sim_results

echo "Task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO seed=$SEED"

Rscript "$SIM_DIR/simulate_ground_truth.R" "$SEED" "$SCENARIO" "$SIM_ROOT"
Rscript "$SIM_DIR/run_simulation_de.R"    "$SEED" "$SCENARIO" "$SIM_ROOT" "$RESULTS_ROOT"

## Free disk: raw/normalized count CSVs are large and no longer needed once
## DE has run and ground_truth.csv has been copied into sim_results/. If a
## later analysis needs them again (e.g. adding a new model), regenerate via
## simulate_ground_truth.R with the same seed -- deterministic/reproducible.
rm -f "$SIM_ROOT/$SCENARIO/seed_$SEED/raw_counts.csv" "$SIM_ROOT/$SCENARIO/seed_$SEED/normalized_counts.csv"
