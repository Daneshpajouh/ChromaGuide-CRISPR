#!/bin/bash
#SBATCH --job-name=cg-data-prep
#SBATCH --account=def-kwiese_cpu
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=data_prep_slurm-%j.out

# ============================================================
# ChromaGuide: Data Preparation with REAL CRISPR Datasets
# Run ONCE on login node before submitting training jobs
# ============================================================

echo "===== DATA PREPARATION ====="
echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================="

# Load modules (if on Alliance clusters)
module load StdEnv/2023 python/3.11.5 scipy-stack 2>/dev/null || true

# Activate virtual environment
source ~/scratch/chromaguide_v2_env/bin/activate

# Navigate to project
cd ~/scratch/chromaguide_v2

echo ""
echo "Step 1: Downloading REAL CRISPR datasets..."
echo "  Source: CRISPR-FMC benchmark (9 datasets, 291K+ sgRNAs)"
echo "  URL: https://github.com/xx0220/CRISPR-FMC"
echo ""

# Download all 9 benchmark datasets
mkdir -p data/raw/crispr_fmc/datasets

for dataset in WT ESP HF HCT116 HELA HL60; do
    echo "  Downloading ${dataset}.csv..."
    curl -sL "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets/${dataset}.csv" \
        -o "data/raw/crispr_fmc/datasets/${dataset}.csv"
done

# These have different filenames
echo "  Downloading xCas.csv (→ xCas9)..."
curl -sL "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets/xCas.csv" \
    -o "data/raw/crispr_fmc/datasets/xCas.csv"

echo "  Downloading SpCas9-NG.csv..."
curl -sL "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets/SpCas9-NG.csv" \
    -o "data/raw/crispr_fmc/datasets/SpCas9-NG.csv"

echo "  Downloading Sniper-Cas9.csv..."
curl -sL "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets/Sniper-Cas9.csv" \
    -o "data/raw/crispr_fmc/datasets/Sniper-Cas9.csv"

echo ""
echo "Verifying downloads..."
for f in data/raw/crispr_fmc/datasets/*.csv; do
    lines=$(($(wc -l < "$f") - 1))
    echo "  $(basename $f): ${lines} samples"
done

echo ""
echo "Step 2: Running preprocessing pipeline..."
python experiments/prepare_data.py --data-dir data/ --skip-download

echo ""
echo "Step 3: Verifying processed data..."
echo "  Sequences:"
wc -l data/processed/sequences.csv
echo "  Efficacy:"
python -c "import numpy as np; e=np.load('data/processed/efficacy.npy'); print(f'  shape={e.shape}, mean={e.mean():.4f}, var={e.var():.6f}')"
echo "  Epigenomic:"
python -c "import numpy as np; e=np.load('data/processed/epigenomic.npy'); print(f'  shape={e.shape}')"
echo "  Splits:"
ls -la data/processed/splits/

echo ""
echo "===== DATA PREPARATION COMPLETE ====="
echo "End: $(date)"
