#!/bin/bash
#SBATCH --job-name=CRISPRO_Pipeline_Test
#SBATCH --output=logs/pipeline_test_%j.log
#SBATCH --error=logs/pipeline_test_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00

echo "=== CRISPRO Pipeline Verification Test ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load Modules
module load python/3.11
module load cuda/12

# Navigate to project directory
cd /scratch/amird/CRISPRO-MAMBA-X

# Activate Environment
source venv/bin/activate
export PYTHONPATH=/scratch/amird/CRISPRO-MAMBA-X:$PYTHONPATH

# Run test
python test_cluster_pipeline.py

echo "=== Test Complete ==="
