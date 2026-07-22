#!/bin/bash
## Resumes ONLY the g2 scenario of build_seurat_input_multi.py -- g1
## already completed successfully (20,000 cells, balanced 5,000/animal,
## verified sane) in the earlier srun run that timed out mid-g2 at the
## 1-hour limit. Submitted via sbatch (not srun) so it survives
## independently of any local session, with a longer time budget and
## output captured to a log file instead of an ephemeral srun stream.
#SBATCH --job-name=build_seurat_g2
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=03:30:00
#SBATCH --output=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/build_seurat_g2_%j.out
#SBATCH --error=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/build_seurat_g2_%j.err

cd ~/AnatomicalRegions_iScience/IndependentValidation
python3 build_seurat_input_multi.py \
  ~/data/AnatomicalRegions_iScience/independent_validation/cell_metadata_tagged_multi.csv \
  ~/data/AnatomicalRegions_iScience/independent_validation/raw \
  ~/data/AnatomicalRegions_iScience/independent_validation/seurat_input_multi \
  20000 0 g2
