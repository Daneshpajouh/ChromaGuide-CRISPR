#!/bin/bash
# Submit all ChromaGuide jobs on nibi
# Generated automatically

echo "Submitting 3 jobs on nibi..."
cd ~/scratch/chromaguide_v2

echo "Submitting: cnn_gru | Split A | Seed 456"
sbatch experiments/slurm_jobs/cnn_gru_splitA_seed456_nibi.sh

echo "Submitting: cnn_gru | Split B | Seed 456"
sbatch experiments/slurm_jobs/cnn_gru_splitB_seed456_nibi.sh

echo "Submitting: cnn_gru | Split C | Seed 456"
sbatch experiments/slurm_jobs/cnn_gru_splitC_seed456_nibi.sh


echo ""
echo "All 3 jobs submitted on nibi."
echo "Check status: squeue -u $USER"
