#!/bin/bash
#SBATCH --job-name=CRISPRO_Test_Mini
#SBATCH --output=logs/test_mini_%j.log
#SBATCH --error=logs/test_mini_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

echo "=== CRISPRO TEST RUN (Mini Dataset) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load Modules
module load python/3.11
module load cuda/12

# Activate Environment
cd /scratch/amird/CRISPRO-MAMBA-X
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Test Run on Mini Dataset
python -m src.train \
    --data_path data/mini/crisprofft/mini_crisprofft.txt \
    --batch_size 4 \
    --epochs 3 \
    --causal \
    --quantum \
    --topo \
    --thermo \
    --lr 1e-4

echo "Test Job Complete."
