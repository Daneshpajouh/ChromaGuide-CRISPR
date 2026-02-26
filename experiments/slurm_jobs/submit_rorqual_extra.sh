#!/bin/bash
# Submit redistributed jobs on rorqual
echo 'Submitting 3 redistributed jobs on rorqual...'
cd ~/scratch/chromaguide_v2

echo "Submitting: dnabert2 | Split A | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/dnabert2_splitA_seed123_rorqual.sh

echo "Submitting: dnabert2 | Split B | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/dnabert2_splitB_seed123_rorqual.sh

echo "Submitting: dnabert2 | Split C | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/dnabert2_splitC_seed123_rorqual.sh

echo ""
echo "All 3 redistributed jobs submitted on rorqual."
