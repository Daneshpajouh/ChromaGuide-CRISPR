#!/bin/bash
#SBATCH --job-name=off_target_v4_enhanced
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/off_target_v4_enhanced_%j.out
#SBATCH --error=slurm_logs/off_target_v4_enhanced_%j.err

module load cuda/12.2 python/3.11
source ~/env_chromaguide/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

cd ~/chromaguide_experiments
mkdir -p slurm_logs

# Run enhanced off-target training with class weights and improved architecture
echo "Starting off-target v4 ENHANCED training..."
echo "Features: Weighted BCE loss, batch norm, increased capacity, class imbalance handling"
python -u scripts/train_off_target_v4.py \
    --data_path data/raw/crisprofft/CRISPRoffT_all_targets.txt \
    --epochs 100 \
    --batch_size 512 \
    --lr 0.0005

echo "Training completed"
