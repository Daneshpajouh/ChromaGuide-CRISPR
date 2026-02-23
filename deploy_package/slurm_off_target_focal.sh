#!/bin/bash
#SBATCH --job-name=off_target_focal_loss
#SBATCH --account=def-kwiese_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/off_target_focal_%j.out
#SBATCH --error=slurm_logs/off_target_focal_%j.err

module load cuda/12.2 python/3.11
source ~/env_chromaguide/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

cd ~/chromaguide_experiments
mkdir -p slurm_logs

echo "========================================="
echo "OFF-TARGET WITH FOCAL LOSS"
echo "========================================="
echo "Features: Focal loss (gamma=2.0, alpha=0.25) for extreme class imbalance"
echo "Data: CRISPRoffT (245,846 samples, 99.54% OFF-target)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run off-target with focal loss
python -u scripts/train_off_target_focal.py \
    --data_path data/raw/crisprofft/CRISPRoffT_all_targets.txt \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.0005

echo ""
echo "End time: $(date)"
echo "Training completed"
