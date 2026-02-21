#!/bin/bash
#SBATCH --job-name=chromaguide_ablations
#SBATCH --account=def-kwiese_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablations_%j.out
#SBATCH --error=logs/ablations_%j.err

echo "Starting Ablation batch on $(hostname) at $(date)"

# Load modules
module load cuda/12.2
module load python/3.10

source /home/amird/env_chromaguide/bin/activate
export PYTHONPATH=/home/amird/chromaguide_experiments/src:/home/amird/chromaguide_experiments:$PYTHONPATH

cd /home/amird/chromaguide_experiments
mkdir -p logs results/ablations

# 1. Baseline: CNN-GRU Seq-only
echo "Running Baseline: CNN-GRU Sequence-only..."
python -u scripts/train_on_real_data_v2.py \
    --backbone cnn_gru \
    --no_epi \
    --epochs 10 \
    --batch_size 1024 \
    --output_name results/ablations/cnn_gru_seq_only.json

# 2. Sequential Improvement: CNN-GRU + Epigenomics
echo "Running Improvement: CNN-GRU + Epigenomics..."
python -u scripts/train_on_real_data_v2.py \
    --backbone cnn_gru \
    --epochs 10 \
    --batch_size 1024 \
    --output_name results/ablations/cnn_gru_full.json

# 3. Backbone Comparison: DNABERT-2 (Transformer) Seq-only
echo "Running Comparison: DNABERT-2 Sequence-only..."
python -u scripts/train_on_real_data_v2.py \
    --backbone dnabert2 \
    --no_epi \
    --epochs 10 \
    --batch_size 128 \
    --output_name results/ablations/dnabert2_seq_only.json

echo "Ablation batch complete."
