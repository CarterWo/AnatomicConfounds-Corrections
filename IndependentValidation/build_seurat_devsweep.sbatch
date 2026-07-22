#!/bin/bash
## Builds Seurat input CSVs for one (dev, scenario) pair of the composition
## dev-sweep, so the full DE pipeline can be run at every dev level, not
## just the cheap composition-tagging diagnostic. Usage:
##   sbatch --export=DEV=1.5,SCEN=g1 build_seurat_devsweep.sbatch
#SBATCH --job-name=build_seurat_dev
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=03:30:00
#SBATCH --output=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/build_seurat_dev_%j.out
#SBATCH --error=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/build_seurat_dev_%j.err

cd ~/AnatomicalRegions_iScience/IndependentValidation
python3 build_seurat_input_multi.py \
  ~/data/AnatomicalRegions_iScience/independent_validation/dev_sweep_tagged_dev${DEV}.csv \
  ~/data/AnatomicalRegions_iScience/independent_validation/raw \
  ~/data/AnatomicalRegions_iScience/independent_validation/seurat_input_multi_dev${DEV} \
  20000 0 ${SCEN}
