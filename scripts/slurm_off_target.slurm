#!/bin/bash
#SBATCH --job-name=chromaguide_offtarget
#SBATCH --account=def-kwiese_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/amird/chromaguide_experiments/slurm_logs/offtarget_%j.log
#SBATCH --error=/home/amird/chromaguide_experiments/slurm_logs/offtarget_%j.err

echo "Job started on $(hostname) at $(date)"
export PYTHONPATH=/home/amird/chromaguide_experiments/src:$PYTHONPATH

source /home/amird/env_chromaguide/bin/activate

cd /home/amird/chromaguide_experiments

python scripts/train_off_target.py
