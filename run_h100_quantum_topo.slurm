#!/bin/bash
#SBATCH --job-name=CRISPRO_Quantum_Grand_Unified
#SBATCH --output=logs/h100_quantum_%j.log
#SBATCH --error=logs/h100_quantum_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

echo "=== H100 CLUSTER JOB: Grand Unification (Causal + Quantum + Topo) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Load Modules (Adapt to cluster environment)
# module load cuda/12.1
# module load python/3.10

# Activate Environment
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Run Training
# Full Production Run (238k samples) with all Theoretical Modules
python src/train.py \
    --data_path data/raw/crisprofft/CRISPRoffT_all_targets.txt \
    --production \
    --batch_size 32 \
    --causal \
    --quantum \
    --topo \
    --epochs 100 \
    --lr 5e-4

echo "Job Complete."
