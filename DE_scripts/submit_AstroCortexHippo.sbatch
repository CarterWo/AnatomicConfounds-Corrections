#!/bin/bash
#SBATCH --job-name=de_astroCH
#SBATCH --partition=Lewis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=de_astroCH_%j.out
#SBATCH --error=de_astroCH_%j.err

set -euo pipefail

# --- Modules / env ---
module purge
module load r/4.3 || module load R/4.3 || true
# If you use a conda env instead, swap the module load for:
# source ~/.bashrc && conda activate variance_partition

# --- Run ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Host: $(hostname)"
echo "Start: $(date)"

Rscript Base_Comparison_AstroCortexHippo.R

echo "End: $(date)"
