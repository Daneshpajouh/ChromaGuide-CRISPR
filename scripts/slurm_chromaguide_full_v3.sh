#!/bin/bash
#SBATCH --job-name=chromaguide_full_v3
#SBATCH --account=def-kwiese_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

echo "Job started on $(hostname) at $(date)"

# Load modules
module load cuda/12.2
module load python/3.10

source /home/amird/env_chromaguide/bin/activate
export PYTHONPATH=/home/amird/chromaguide_experiments/src:/home/amird/chromaguide_experiments:$PYTHONPATH

# Setup HuggingFace to use cached models
export HF_HOME="/home/amird/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

cd /home/amird/chromaguide_experiments
mkdir -p logs results/chromaguide_full

# Run the production script
python -u scripts/train_on_real_data_v2.py \
    --backbone dnabert2 \
    --fusion gate \
    --epochs 20 \
    --batch_size 128 \
    --patience 5 \
    --learning_rate 2e-5
