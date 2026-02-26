#!/bin/bash
# Submit all ChromaGuide jobs on killarney
# Generated automatically

echo "Submitting 9 jobs on killarney..."
cd ~/scratch/chromaguide_v2

echo "Submitting: caduceus | Split A | Seed 123"
sbatch experiments/slurm_jobs/caduceus_splitA_seed123_killarney.sh

echo "Submitting: caduceus | Split B | Seed 123"
sbatch experiments/slurm_jobs/caduceus_splitB_seed123_killarney.sh

echo "Submitting: caduceus | Split C | Seed 123"
sbatch experiments/slurm_jobs/caduceus_splitC_seed123_killarney.sh

echo "Submitting: dnabert2 | Split A | Seed 123"
sbatch experiments/slurm_jobs/dnabert2_splitA_seed123_killarney.sh

echo "Submitting: dnabert2 | Split B | Seed 123"
sbatch experiments/slurm_jobs/dnabert2_splitB_seed123_killarney.sh

echo "Submitting: dnabert2 | Split C | Seed 123"
sbatch experiments/slurm_jobs/dnabert2_splitC_seed123_killarney.sh

echo "Submitting: evo | Split A | Seed 123"
sbatch experiments/slurm_jobs/evo_splitA_seed123_killarney.sh

echo "Submitting: evo | Split B | Seed 123"
sbatch experiments/slurm_jobs/evo_splitB_seed123_killarney.sh

echo "Submitting: evo | Split C | Seed 123"
sbatch experiments/slurm_jobs/evo_splitC_seed123_killarney.sh


echo ""
echo "All 9 jobs submitted on killarney."
echo "Check status: squeue -u $USER"
