#!/bin/bash
#SBATCH --job-name=chromaguide_offtarget_v3
#SBATCH --account=def-kwiese_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/offtarget_%j.out
#SBATCH --error=logs/offtarget_%j.err

echo "Job started on $(hostname) at $(date)"
export PYTHONPATH=/home/amird/chromaguide_experiments/src:/home/amird/chromaguide_experiments:$PYTHONPATH

source /home/amird/env_chromaguide/bin/activate
cd /home/amird/chromaguide_experiments

mkdir -p logs

# Use -u for unbuffered output
python -u scripts/train_off_target_v2.py --epochs 20 --batch_size 1024
