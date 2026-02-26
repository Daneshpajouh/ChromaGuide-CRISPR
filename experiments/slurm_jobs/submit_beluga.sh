#!/bin/bash
# Submit all ChromaGuide jobs on beluga
# Generated automatically

echo "Submitting 6 jobs on beluga..."
cd ~/scratch/chromaguide_v2

echo "Submitting: cnn_gru | Split A | Seed 123"
sbatch experiments/slurm_jobs/cnn_gru_splitA_seed123_beluga.sh

echo "Submitting: cnn_gru | Split B | Seed 123"
sbatch experiments/slurm_jobs/cnn_gru_splitB_seed123_beluga.sh

echo "Submitting: cnn_gru | Split C | Seed 123"
sbatch experiments/slurm_jobs/cnn_gru_splitC_seed123_beluga.sh

echo "Submitting: caduceus | Split A | Seed 456"
sbatch experiments/slurm_jobs/caduceus_splitA_seed456_beluga.sh

echo "Submitting: caduceus | Split B | Seed 456"
sbatch experiments/slurm_jobs/caduceus_splitB_seed456_beluga.sh

echo "Submitting: caduceus | Split C | Seed 456"
sbatch experiments/slurm_jobs/caduceus_splitC_seed456_beluga.sh


echo ""
echo "All 6 jobs submitted on beluga."
echo "Check status: squeue -u $USER"
