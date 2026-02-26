#!/bin/bash
#SBATCH --job-name=cg-data-prep
#SBATCH --account=def-kwiese_cpu
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/data_prep_slurm-%j.out
#SBATCH --error=results/data_prep_slurm-%j.err

# ============================================================
# ChromaGuide: Data Preparation
# Downloads, preprocesses, and builds splits
# ============================================================

echo "===== DATA PREPARATION ====="
echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"

# Load modules
module load StdEnv/2023 python/3.11.5 scipy-stack 2>/dev/null || true

# Activate venv
source ~/scratch/chromaguide_v2_env/bin/activate

export PYTHONUNBUFFERED=1

cd ~/scratch/chromaguide_v2

# Create data directory
mkdir -p data/raw data/processed/splits results

echo ""
echo "Python: $(which python)"
echo ""

# Run data preparation with synthetic data first
# (Real data download requires internet access which may be limited on compute nodes)
# For production: run on login node with --no-slurm flag

python experiments/prepare_data.py \
    --data-dir ~/scratch/chromaguide_v2/data \
    --synthetic

EXIT_CODE=$?

echo ""
echo "===== DATA PREP COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# Verify
if [ -f data/processed/sequences.parquet ]; then
    echo "✓ sequences.parquet exists"
    python -c "import pandas as pd; df=pd.read_parquet('data/processed/sequences.parquet'); print(f'  {len(df)} samples')"
fi
if [ -f data/processed/efficacy.npy ]; then
    echo "✓ efficacy.npy exists"
    python -c "import numpy as np; a=np.load('data/processed/efficacy.npy'); print(f'  shape: {a.shape}')"
fi
if [ -f data/processed/epigenomic.npy ]; then
    echo "✓ epigenomic.npy exists"
    python -c "import numpy as np; a=np.load('data/processed/epigenomic.npy'); print(f'  shape: {a.shape}')"
fi
echo ""
echo "Split files:"
ls -la data/processed/splits/ 2>/dev/null || echo "No splits found"

exit $EXIT_CODE
