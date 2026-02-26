#!/bin/bash
#SBATCH --job-name=cg-{BACKBONE}-{SPLIT}-s{SEED}
#SBATCH --account={ACCOUNT}
#SBATCH --time={TIME}
#SBATCH --mem={MEM}
#SBATCH --gres=gpu:{GPU_TYPE}:1
#SBATCH --cpus-per-task=4
#SBATCH --output={OUTPUT_DIR}/slurm-%j.out
#SBATCH --error={OUTPUT_DIR}/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amir@mystorax.com

# ============================================================
# ChromaGuide Training: {BACKBONE} | Split {SPLIT} | Seed {SEED}
# Cluster: {CLUSTER}
# ============================================================

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: {BACKBONE}"
echo "Split: {SPLIT}"
echo "Seed: {SEED}"
echo "==================="

# Load modules
{MODULE_LOADS}

# Activate virtual environment
source ~/scratch/chromaguide_v2_env/bin/activate

# Set environment
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=~/scratch/.cache/huggingface
export HF_HOME=~/scratch/.cache/huggingface
export TORCH_HOME=~/scratch/.cache/torch
export WANDB_MODE=offline
export OMP_NUM_THREADS=4

# Navigate to project
cd ~/scratch/chromaguide_v2

# Create output directory
mkdir -p results/{BACKBONE}_split{SPLIT}_seed{SEED}

echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "VRAM: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\")')"
fi
echo ""

# Run training
python experiments/train_experiment.py \
    --backbone {BACKBONE} \
    --split {SPLIT} \
    --split-fold {SPLIT_FOLD} \
    --seed {SEED} \
    --data-dir ~/scratch/chromaguide_v2/data \
    --output-dir ~/scratch/chromaguide_v2/results \
    --no-wandb \
    --patience {PATIENCE} \
    --gradient-clip 1.0 \
    {EXTRA_ARGS}

EXIT_CODE=$?

echo ""
echo "===== JOB COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# Copy results summary
if [ -f results/{BACKBONE}_split{SPLIT}_seed{SEED}/results.json ]; then
    echo "Results:"
    cat results/{BACKBONE}_split{SPLIT}_seed{SEED}/results.json
fi

exit $EXIT_CODE
