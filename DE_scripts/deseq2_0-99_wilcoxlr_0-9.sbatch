#!/bin/bash
# =============================================================================
# deseq2_0-99_wilcoxlr_0-9.sbatch — every non-Dream model, at its full configured seed
# coverage, in ONE array job:
#
#   DESeq2    seeds 0-99   (config.yaml -> composition.deseq2_iterations)
#   Wilcoxon  seeds 0-9    (config.yaml -> composition.sc_iterations)
#   LR        seeds 0-9    (config.yaml -> composition.sc_iterations)
#
# Dream is deliberately excluded — it is 3 seeds only and costs ~4h40 per
# scenario on the whole cohort, so it stays in dream_0-2.sbatch.
#
# Two task shapes in one array
# ----------------------------
# The seed space is split because the two halves have completely different
# cost profiles:
#
#   tasks 0-39   one task per (dev x seed) for seeds 0-9, models deseq2,wilcox,lr
#                Wilcoxon/LR are slow and single-threaded, so each seed gets its
#                own task and they run 40-wide.
#   tasks 40-43  one task per dev covering seeds 10-99, models deseq2 only
#                DESeq2 is ~1 min per fit; the dominant cost is the one-time
#                cohort load (~2 min, ~100 G). Looping 90 seeds inside a single
#                task pays that load once instead of 90 times.
#
# The future.globals ceiling
# --------------------------
# Seurat's FindMarkers dispatches through *future*, not the BiocParallel backend
# that Dream and DESeq2 use, and future refuses by default to export any global
# over 500 MiB. The whole cohort is ~1.66 GiB of globals, which is why every
# whole-cohort Wilcoxon/LR call in job 15033684 died with
#   "The total size of the 5 globals ... exceeds the maximum allowed size
#    500.00 MiB per by R option future.globals.maxSize"
# while the cell-type subsets (Astrocytes ~155 MiB, Microglia ~17 MiB) would
# have succeeded. R_FUTURE_GLOBALS_MAXSIZE below lifts the ceiling; future reads
# R_FUTURE_* env vars at load time, so no R source change is required.
#
# Concurrency
# -----------
# Output paths are a pure function of (framework, dev, seed, label, file_tag).
# The job ID is NOT in the path and nothing locks, so two jobs given the same
# dev+seed write the same filenames and can interleave into corrupt CSVs.
#
# This script and dream_0-2.sbatch are DISJOINT — that one runs --models dream
# and nothing else, while this one never runs Dream. They may be submitted at
# the same time, and the full 0-43 array needs no exclusions.
#
# It DOES collide with the superseded deseq_seeds.sbatch, sc_tests.sbatch, and
# pb_wilcox.sbatch, all of which claim seeds this script already owns. Do not
# run any of those alongside it.
#
# Submit from the repo root:   sbatch DE_scripts/deseq2_0-99_wilcoxlr_0-9.sbatch
# Pick a framework:   sbatch --export=ALL,FRAMEWORK=fmt DE_scripts/deseq2_0-99_wilcoxlr_0-9.sbatch
# Cap concurrency:    sbatch --array=0-43%12 DE_scripts/deseq2_0-99_wilcoxlr_0-9.sbatch
# =============================================================================
#SBATCH --job-name=deseq2_wilcoxlr
#SBATCH --output=logs/deseq2_0-99_wilcoxlr_0-9_%A_%a.out
#SBATCH --array=0-43
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G

set -euo pipefail
mkdir -p logs

# Lift future's global-export ceiling to 16 GiB (default 500 MiB). Without this
# every whole-cohort FindMarkers call aborts before doing any work.
export R_FUTURE_GLOBALS_MAXSIZE=17179869184

# Make R available (edit for your cluster: an environment module or a conda env)
RSCRIPT="${RSCRIPT:-/home/cdw5fz/.conda/envs/spatial_env/bin/Rscript}"

# Repo root = directory containing config.yaml (defaults to the submit dir).
REPO="${REPO:-$SLURM_SUBMIT_DIR}"

# Framework (dataset/contrast); empty -> config.yaml active_framework.
FRAMEWORK="${FRAMEWORK:-}"

# Keep DEVS aligned with config.yaml -> composition.deviations. SC_SEEDS is the
# per-(dev,seed) half; PB_RANGE is the DESeq2-only remainder. If you change
# either, update --array to (nDEVS * nSC_SEEDS + nDEVS - 1).
DEVS=(1.5 1 0.5 0.25)
SC_SEEDS=(0 1 2 3 4 5 6 7 8 9)
PB_RANGE="${PB_RANGE:-10-99}"

N_DEVS=${#DEVS[@]}
N_SC=${#SC_SEEDS[@]}
SPLIT=$(( N_DEVS * N_SC ))          # first task index of the DESeq2-only half

if [ "$SLURM_ARRAY_TASK_ID" -lt "$SPLIT" ]; then
    # ── Cell-level half: one (dev, seed) pair, all three non-Dream models ──
    DEV="${DEVS[$(( SLURM_ARRAY_TASK_ID / N_SC ))]}"
    SEEDS="${SC_SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SC ))]}"
    MODELS="deseq2,wilcox,lr"
else
    # ── DESeq2-only half: one dev, the whole remaining seed range ──
    DEV="${DEVS[$(( SLURM_ARRAY_TASK_ID - SPLIT ))]}"
    SEEDS="$PB_RANGE"
    MODELS="deseq2"
fi

echo "Array task $SLURM_ARRAY_TASK_ID -> deviation $DEV, seeds $SEEDS, models $MODELS  (framework: ${FRAMEWORK:-<config default>})"
echo "Host: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK:-?}  future.globals.maxSize: $R_FUTURE_GLOBALS_MAXSIZE  Start: $(date)"

# --sc-seeds 10 pins the Wilcoxon/LR threshold so seeds 0-9 always qualify and
# the DESeq2-only tasks never trip it, regardless of config drift.
#
# Trailing "$@" forwards any extra sbatch arguments through to R, so
#   sbatch DE_scripts/deseq2_0-99_wilcoxlr_0-9.sbatch --resume
# actually reaches LMM_all.R. Without it the flag is silently dropped.
"$RSCRIPT" "$REPO/DE_scripts/LMM_all.R" \
    ${FRAMEWORK:+--framework "$FRAMEWORK"} \
    --devs "$DEV" \
    --seeds "$SEEDS" \
    --models "$MODELS" \
    --sc-seeds 10 \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    "$@"

echo "Done: $(date)"
