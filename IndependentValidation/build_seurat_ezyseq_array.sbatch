#!/bin/bash
## Seurat-input export for all 5 dev magnitudes x {g1,g2} using the NEW
## ezy_seq-based composition tagging (cell_metadata_tagged_multi_ezyseq_dev*.csv),
## replacing the old real_animal_composition_engineering.py results.
## 10 tasks: dev in {0.5,1.0,1.5,2.0,3.0} x scenario in {g1,g2}.
#SBATCH --job-name=build_seurat_ezyseq
#SBATCH --partition=requeue
#SBATCH --array=0-9%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/build_seurat_ezyseq_%A_%a.out
#SBATCH --error=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/build_seurat_ezyseq_%A_%a.err

set -e

DEVS=(0.5 0.5 1.0 1.0 1.5 1.5 2.0 2.0 3.0 3.0)
SCENS=(g1 g2 g1 g2 g1 g2 g1 g2 g1 g2)
DEV=${DEVS[$SLURM_ARRAY_TASK_ID]}
SCEN=${SCENS[$SLURM_ARRAY_TASK_ID]}

cd /home/nbhtd/AnatomicalRegions_iScience/IndependentValidation
echo "dev=$DEV scenario=$SCEN"
python3 build_seurat_input_multi.py \
  /home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/cell_metadata_tagged_multi_ezyseq_dev${DEV}.csv \
  /home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/raw \
  /home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/seurat_input_multi_ezyseq_dev${DEV} \
  20000 0 ${SCEN}
