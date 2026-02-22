#!/bin/bash
#SBATCH --job-name=chromaguide-multimodal-dnabert2-fixed
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/multimodal_dnabert2_fixed_%j.out
#SBATCH --error=slurm_logs/multimodal_dnabert2_fixed_%j.err

module load cuda/12.2 python/3.11
source ~/env_chromaguide/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

# Force offline mode - must not try to download from network
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HOME/.cache/huggingface"

cd ~/chromaguide_experiments
mkdir -p slurm_logs

echo "========================================="
echo "DNABERT-2 FIXED MULTIMODAL TRAINING"
echo "========================================="
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run with DNABERT-2 backbone - NEW FIX
python -u scripts/train_on_real_data_v2.py \
    --backbone dnabert2 \
    --epochs 50 \
    --batch_size 250 \
    --learning_rate 5e-4 \
    --patience 15 \
    --device cuda \
    --split A

echo ""
echo "End time: $(date)"
echo "Training completed"
