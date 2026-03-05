#!/bin/bash
#SBATCH --job-name=chromaguide_real
#SBATCH --account=def-kwiese
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/real_data_%j.log
#SBATCH --error=slurm_logs/real_data_%j.err

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"

module load cuda/12.2
module load python/3.11

source ~/env_chromaguide/bin/activate

export PYTHONPATH=/home/amird/chromaguide_experiments/src:$PYTHONPATH

# Use HuggingFace offline if possible
export HF_HUB_OFFLINE=1
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"

mkdir -p slurm_logs

echo "Starting Python training script..."
python3 -u scripts/train_on_real_data_v2.py --backbone dnabert2 --split A --epochs 50 --batch_size 250 --device cuda --seed 42

echo "Job finished at $(date)"
