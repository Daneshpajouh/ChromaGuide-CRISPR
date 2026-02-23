#!/bin/bash

#SBATCH --job-name=v10_offtarget
#SBATCH --time=08:00:00
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --output=logs/slurm_offtarget_v10_%j.log
#SBATCH --error=logs/slurm_offtarget_v10_%j.err

# Load modules
module load python/3.11 cuda/12.2 pytorch

# Activate environment
source ~/venv/bin/activate || true

cd ~/chromaguide

echo "=========================================="
echo "V10 OFF-TARGET CLASSIFICATION"
echo "Fir Cluster - 2x H100 GPU"
echo "=========================================="
echo ""

# Check devices
python3 << 'EOF'
import torch
print(f"GPU Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
EOF

echo ""

# Run training with DataParallel across 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1
python3 -u scripts/train_off_target_v10.py 2>&1 | tee logs/off_target_v10_fir.log

echo ""
echo "✅ V10 Off-target training complete on Fir"
