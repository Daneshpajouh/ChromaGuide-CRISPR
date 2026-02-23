#!/bin/bash
# SLURM job script for FIR - V10 Conformal Prediction Calibration
# Purpose: Calibrate conformal prediction intervals on validation set
# Target: Coverage 0.90 ± 0.02

#SBATCH --job-name=v10_conformal
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/v10_conformal_%j.out
#SBATCH --error=logs/v10_conformal_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G

set -euo pipefail

echo "=== Conformal Prediction Calibration ==="
cd ~/chromaguide

python3 -u scripts/calibrate_conformal.py \
    --model_path models/v10_multimodal_ensemble.pt \
    --val_data data/processed/split_a_gene_held_out_val.npz \
    --coverage 0.90 \
    --output_path models/conformal_intervals.pkl

echo "✓ Conformal calibration complete"
