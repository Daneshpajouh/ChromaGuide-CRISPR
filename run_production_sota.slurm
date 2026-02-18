#!/bin/bash
#SBATCH --job-name=crispro_sota
#SBATCH --output=logs/sota_%j.out
#SBATCH --error=logs/sota_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

echo 'Starting Production SOTA Run: Ensemble + VAE + Diffusion + Benchmark'
echo "Started at: $(date)"

cd /home/amird/projects/def-kwiese/amird/Proposal
source /home/amird/projects/def-kwiese/amird/mamba_h100/bin/activate

# Primary Production Execution
python3 run_production_sota.py

echo "Completed at: $(date)"
