#!/bin/bash
#SBATCH --job-name=vc_export
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vc_export_%j.out
#SBATCH --error=logs/vc_export_%j.err
set -e
ROOT=~/data/AnatomicalRegions_iScience
$ROOT/pyenv_scvi/bin/python export_cosmx_for_variance.py $ROOT/real_data/fully_updated.h5ad $ROOT/variance_components_r21/cosmx_export 8000 0
