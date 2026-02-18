#!/bin/bash
#SBATCH --job-name=CRISPRO_NAS_HPTuning
#SBATCH --output=logs/hptuning_%j.log
#SBATCH --error=logs/hptuning_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00

echo "=== CRISPRO: Neural Architecture Search + Hyperparameter Optimization ==="
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

# Run comprehensive HP tuning (100 trials over 48 hours)
python src/optimization/cluster_hptuning.py \
    --n_trials 100 \
    --study_name crispro_nas_h100_fixed

echo "=== NAS + HP Tuning Complete ==="
echo "Check results in: results_crispro_nas_h100.json"
