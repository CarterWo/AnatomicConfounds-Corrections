#!/bin/bash
#SBATCH --job-name=var_comp
#SBATCH --partition=requeue
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/var_comp_%A_%a.out
#SBATCH --error=logs/var_comp_%A_%a.err
#
# 0-1: CosMx, olfactory bulb removed before sampling (export_cosmx_for_variance.py),
#      Napari and QUINT labels
# 2-3: Zhuang-ABCA MERFISH 20,000-cell subset (no treatment term), Napari- and QUINT-equivalent labels

set -e
module load r/4.4.0
module load gcc/12.3.0
export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${LD_LIBRARY_PATH}"

ROOT=~/data/AnatomicalRegions_iScience
COSMX=$ROOT/variance_components_r21/cosmx_export
MERFISH=$ROOT/independent_validation/seurat_input_multi_ezyseq_realct_dev1.0/g1
SCRIPT=$(pwd)/variance_components.R
OUT=$ROOT/variance_components_r21
mkdir -p "$OUT"

# The CosMx export already excludes the olfactory bulb; passing Olfactory again is a
# check that should report 0 cells dropped.
case $SLURM_ARRAY_TASK_ID in
  0) Rscript "$SCRIPT" "$COSMX"   "$OUT/vc_cosmx_napari.csv"   napari_region 1 16 Olfactory ;;
  1) Rscript "$SCRIPT" "$COSMX"   "$OUT/vc_cosmx_quint.csv"    quint_region  1 16 Olfactory ;;
  2) Rscript "$SCRIPT" "$MERFISH" "$OUT/vc_merfish_napari.csv" napari_region 0 16 ;;
  3) Rscript "$SCRIPT" "$MERFISH" "$OUT/vc_merfish_quint.csv"  quint_region  0 16 ;;
esac
