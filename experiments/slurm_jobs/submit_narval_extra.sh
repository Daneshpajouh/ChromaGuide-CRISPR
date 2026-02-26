#!/bin/bash
# Submit redistributed jobs on narval
echo 'Submitting 6 redistributed jobs on narval...'
cd ~/scratch/chromaguide_v2

echo "Submitting: caduceus | Split A | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/caduceus_splitA_seed123_narval.sh

echo "Submitting: caduceus | Split B | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/caduceus_splitB_seed123_narval.sh

echo "Submitting: caduceus | Split C | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/caduceus_splitC_seed123_narval.sh

echo "Submitting: caduceus | Split A | Seed 456 (from beluga)"
sbatch experiments/slurm_jobs/caduceus_splitA_seed456_narval.sh

echo "Submitting: caduceus | Split B | Seed 456 (from beluga)"
sbatch experiments/slurm_jobs/caduceus_splitB_seed456_narval.sh

echo "Submitting: caduceus | Split C | Seed 456 (from beluga)"
sbatch experiments/slurm_jobs/caduceus_splitC_seed456_narval.sh

echo ""
echo "All 6 redistributed jobs submitted on narval."
