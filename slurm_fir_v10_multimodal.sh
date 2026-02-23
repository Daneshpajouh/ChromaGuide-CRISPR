#!/bin/bash

#SBATCH --job-name=v10_multimodal
#SBATCH --time=06:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=logs/slurm_multimodal_v10_%j.log
#SBATCH --error=logs/slurm_multimodal_v10_%j.err

# Load modules
module load python/3.11 cuda/12.2 pytorch

# Activate environment or use system Python
source ~/venv/bin/activate || true

cd ~/chromaguide

echo "=========================================="
echo "V10 MULTIMODAL (ON-TARGET)"
echo "Fir Cluster - H100 GPU"
echo "=========================================="
echo ""

# Check device
python3 << 'EOF'
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""

# Run training
python3 -u scripts/train_on_real_data_v10.py 2>&1 | tee logs/multimodal_v10_fir.log

echo ""
echo "✅ V10 Multimodal training complete on Fir"
