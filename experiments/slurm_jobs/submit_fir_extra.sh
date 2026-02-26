#!/bin/bash
# Submit redistributed jobs on fir
echo 'Submitting 3 redistributed jobs on fir...'
cd ~/scratch/chromaguide_v2

echo "Submitting: evo | Split A | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/evo_splitA_seed123_fir.sh

echo "Submitting: evo | Split B | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/evo_splitB_seed123_fir.sh

echo "Submitting: evo | Split C | Seed 123 (from killarney)"
sbatch experiments/slurm_jobs/evo_splitC_seed123_fir.sh

echo ""
echo "All 3 redistributed jobs submitted on fir."
