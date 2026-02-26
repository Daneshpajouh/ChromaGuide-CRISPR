#!/bin/bash
# Submit redistributed jobs on nibi
echo 'Submitting 3 redistributed jobs on nibi...'
cd ~/scratch/chromaguide_v2

echo "Submitting: cnn_gru | Split A | Seed 123 (from beluga)"
sbatch experiments/slurm_jobs/cnn_gru_splitA_seed123_nibi.sh

echo "Submitting: cnn_gru | Split B | Seed 123 (from beluga)"
sbatch experiments/slurm_jobs/cnn_gru_splitB_seed123_nibi.sh

echo "Submitting: cnn_gru | Split C | Seed 123 (from beluga)"
sbatch experiments/slurm_jobs/cnn_gru_splitC_seed123_nibi.sh

echo ""
echo "All 3 redistributed jobs submitted on nibi."
