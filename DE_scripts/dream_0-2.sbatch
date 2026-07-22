#!/bin/bash
# =============================================================================
# dream_0-2.sbatch — DREAM LMM ONLY, seeds 0-2, via LMM_all.R as a SLURM job
# array, ONE ARRAY TASK PER (deviation x seed) PAIR.
#
#   4 deviations (1.5, 1, 0.5, 0.25) x 3 seeds (0, 1, 2) = 12 independent tasks.
#
# Only the first 3 seeds are run: those are the Dream LMM seeds
# (config.yaml -> composition.dream_iterations = 3) and they dominate runtime —
# ~4h40 per whole-cohort scenario, ~10-14h per task.
#
# Division of labour — the two scripts are strictly DISJOINT and may run
# concurrently. This one owns Dream and nothing else; DESeq2 was handed over
# because the companion already covers all 100 seeds of it:
#
#   deseq2_0-99_wilcoxlr_0-9.sbatch   deseq2 0-99, wilcox 0-9, lr 0-9
#   dream_0-2.sbatch                  dream 0-2                    <- this file
#
# Wilcoxon/LR are NOT here: they need the future.globals ceiling raised, which
# only the companion script sets.
#
# Resuming: --resume skips any output CSV that already exists and is complete,
# so only missing/failed fits are redone. It is forwarded via "$@" at the bottom:
#   sbatch DE_scripts/dream_0-2.sbatch --resume
#
# Submit from the repo root:   sbatch DE_scripts/dream_0-2.sbatch
# Pick a framework:   sbatch --export=ALL,FRAMEWORK=fmt DE_scripts/dream_0-2.sbatch
#   (unset -> config.yaml active_framework)
#
# Task index -> (dev, seed) mapping:
#   dev  = DEVS[ task_id / 3 ]      seed = SEEDS[ task_id % 3 ]
#     0,1,2  -> dev 1.5   seeds 0,1,2
#     3,4,5  -> dev 1     seeds 0,1,2
#     6,7,8  -> dev 0.5   seeds 0,1,2
#     9,10,11-> dev 0.25  seeds 0,1,2
#
# Re-running a subset: pass an explicit index list, e.g. only dev 1 / seed 0
# (the task that timed out in job 15033684):
#   sbatch --array=3 DE_scripts/dream_0-2.sbatch --resume
# To cap how many run at once (12 x 256G is a large simultaneous ask):
#   sbatch --array=0-11%4 DE_scripts/dream_0-2.sbatch
#
# LMM_all.R only creates output dirs it does not already have and never deletes,
# so re-running a task overwrites that seed's CSVs in place and leaves the rest
# of LMM_output alone.
# =============================================================================
#SBATCH --job-name=dream_0-2
#SBATCH --output=logs/dream_0-2_%A_%a.out
#SBATCH --array=0-11
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

# Sizing note: at 6 workers a Dream seed took ~14 h and peaked ~139 G; at 16
# workers it ran ~4.5 h but hit OOM against a 200 G limit (~200 G+ RSS).
# 8 workers / 256 G sits between the two: ~150 G expected peak, ~10-11 h per seed.

set -euo pipefail
mkdir -p logs

# Make R available (edit for your cluster: an environment module or a conda env)
RSCRIPT="${RSCRIPT:-/home/cdw5fz/.conda/envs/spatial_env/bin/Rscript}"

# Repo root = directory containing config.yaml (defaults to the submit dir).
REPO="${REPO:-$SLURM_SUBMIT_DIR}"

# Framework (dataset/contrast); empty -> config.yaml active_framework.
FRAMEWORK="${FRAMEWORK:-}"

# Deviation magnitudes and the seeds to run. Keep DEVS aligned with
# config.yaml -> composition.deviations. If you change the length of either
# list, update --array above to (nDEVS * nSEEDS - 1).
DEVS=(1.5 1 0.5 0.25)
SEEDS=(0 1 2)

N_SEEDS=${#SEEDS[@]}
DEV="${DEVS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}"
SEED="${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}"

echo "Array task $SLURM_ARRAY_TASK_ID -> deviation $DEV, seed $SEED (dream)  (framework: ${FRAMEWORK:-<config default>})"
echo "Host: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK:-?}  Mem: ${SLURM_MEM_PER_NODE:-?}M  Start: $(date)"

# Dream ONLY. DESeq2 and the cell-level tests belong to
# deseq2_0-99_wilcoxlr_0-9.sbatch, which keeps the two scripts fully disjoint so
# they can run at the same time. --dream-seeds 3 pins the threshold regardless
# of config drift.
#
# Trailing "$@" forwards any extra sbatch arguments through to R, so
#   sbatch DE_scripts/dream_0-2.sbatch --resume
# actually reaches LMM_all.R. Without it the flag is silently dropped.
"$RSCRIPT" "$REPO/DE_scripts/LMM_all.R" \
    ${FRAMEWORK:+--framework "$FRAMEWORK"} \
    --devs "$DEV" \
    --seeds "$SEED" \
    --models dream \
    --dream-seeds 3 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"

echo "Done: $(date)"
