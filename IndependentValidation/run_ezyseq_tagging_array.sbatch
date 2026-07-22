#!/bin/bash
## Composition-tagging via the real ezy_seq.tag_region_abundance_by_FMT
## (snapshot), replacing the old hand-rolled real_animal_composition_engineering.py.
## 5 tasks: dev in {0.5, 1.0, 1.5, 2.0, 3.0}, seed=0 (matches the original
## design: one seed per magnitude, not a multi-seed sweep).
#SBATCH --job-name=ezyseq_tag
#SBATCH --partition=requeue
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/ezyseq_tag_%A_%a.out
#SBATCH --error=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/logs/ezyseq_tag_%A_%a.err

set -e

DEVS=(0.5 1.0 1.5 2.0 3.0)
DEV=${DEVS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR=/home/nbhtd/AnatomicalRegions_iScience/IndependentValidation
MAPPED_CSV=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/cell_metadata_with_regions_multi.csv
OUT_CSV=/home/nbhtd/data/AnatomicalRegions_iScience/independent_validation/cell_metadata_tagged_multi_ezyseq_dev${DEV}.csv

echo "dev=$DEV -> $OUT_CSV"
python3 "$SCRIPT_DIR/compose_via_ezyseq_multi.py" "$MAPPED_CSV" "$OUT_CSV" "$DEV" 0
