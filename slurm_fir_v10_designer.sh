#!/bin/bash
# SLURM job script for FIR - V10 Designer Score Computation
# Purpose: Compute designer guide efficiency score
# S = w_e*mu - w_r*R - w_u*sigma (weights: w_e=1.0, w_r=0.5, w_u=0.2)

#SBATCH --job-name=v10_designer
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/v10_designer_%j.out
#SBATCH --error=logs/v10_designer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G

set -euo pipefail

echo "=== Designer Score Computation ==="
cd ~/chromaguide

python3 -u scripts/run_designer.py \
    --v10_predictions models/v10_multimodal_predictions.pkl \
    --ground_truth data/processed/split_a_gene_held_out_test.npz \
    --w_efficacy 1.0 \
    --w_resistance 0.5 \
    --w_uncertainty 0.2 \
    --output_path results/designer_scores.json

echo "✓ Designer scores computed"
