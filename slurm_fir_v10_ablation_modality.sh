#!/bin/bash
# SLURM job script for FIR - V10 Modality Ablation
# Purpose: Compare sequence-only vs epigenetic-only vs multimodal

#SBATCH --job-name=v10_ablation_m
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/v10_ablation_modality_%j.out
#SBATCH --error=logs/v10_ablation_modality_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G

set -euo pipefail

echo "=== Modality Ablation Study ==="
cd ~/chromaguide

python3 -u scripts/run_ablation_modality.py \
    --train_data data/processed/split_a_train.npz \
    --val_data data/processed/split_a_gene_held_out_val.npz \
    --test_data data/processed/split_a_gene_held_out_test.npz \
    --output_path results/ablation_modality_results.json \
    --modalities "sequence_only,epigenetic_only,multimodal"

echo "✓ Modality ablation complete"
