#!/bin/bash
#SBATCH --job-name=CRISPRO_NAS_Test
#SBATCH --output=logs/nas_test_%j.log
#SBATCH --error=logs/nas_test_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

echo "=== CRISPRO: NAS Test (3 trials with MoM) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load Modules
module load python/3.11
module load cuda/12

# Activate Environment
cd /scratch/amird/CRISPRO-MAMBA-X
source venv/bin/activate
export PYTHONPATH=/scratch/amird/CRISPRO-MAMBA-X:$PYTHONPATH

# Run quick test (3 trials only)
python src/optimization/cluster_hptuning.py \
    --n_trials 3 \
    --study_name crispro_nas_test

echo "=== NAS Test Complete ==="
