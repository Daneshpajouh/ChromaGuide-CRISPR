#!/bin/bash
# SLURM job script for FIR - V10 Fusion Module Ablation
# Purpose: Compare gated vs concatenation vs cross-attention fusion strategies

#SBATCH --job-name=v10_ablation_f
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/v10_ablation_fusion_%j.out
#SBATCH --error=logs/v10_ablation_fusion_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G

set -euo pipefail

echo "=== Fusion Module Ablation Study ==="
cd ~/chromaguide

python3 -u scripts/run_ablation_fusion.py \
    --train_data data/processed/split_a_train.npz \
    --val_data data/processed/split_a_gene_held_out_val.npz \
    --test_data data/processed/split_a_gene_held_out_test.npz \
    --output_path results/ablation_fusion_results.json \
    --methods "gating,concatenation,cross_attention"

echo "✓ Fusion ablation complete"
