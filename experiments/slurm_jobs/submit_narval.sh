#!/bin/bash
# Submit all ChromaGuide jobs on narval
# Generated automatically

echo "Submitting 6 jobs on narval..."
cd ~/scratch/chromaguide_v2

echo "Submitting: cnn_gru | Split A | Seed 42"
sbatch experiments/slurm_jobs/cnn_gru_splitA_seed42_narval.sh

echo "Submitting: cnn_gru | Split B | Seed 42"
sbatch experiments/slurm_jobs/cnn_gru_splitB_seed42_narval.sh

echo "Submitting: cnn_gru | Split C | Seed 42"
sbatch experiments/slurm_jobs/cnn_gru_splitC_seed42_narval.sh

echo "Submitting: caduceus | Split A | Seed 42"
sbatch experiments/slurm_jobs/caduceus_splitA_seed42_narval.sh

echo "Submitting: caduceus | Split B | Seed 42"
sbatch experiments/slurm_jobs/caduceus_splitB_seed42_narval.sh

echo "Submitting: caduceus | Split C | Seed 42"
sbatch experiments/slurm_jobs/caduceus_splitC_seed42_narval.sh


echo ""
echo "All 6 jobs submitted on narval."
echo "Check status: squeue -u $USER"
