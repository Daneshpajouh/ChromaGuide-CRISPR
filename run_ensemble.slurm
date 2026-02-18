#!/bin/bash
#SBATCH --job-name=CRISPR_Ensemble
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/ensemble_%j.log
#SBATCH --error=logs/ensemble_%j.err

echo "=========================================="
echo "CRISPR ENSEMBLE SOTA AGGREGATION"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
module load python/3.11

# Activate environment
cd /scratch/amird/CRISPRO-MAMBA-X
source venv/bin/activate

# Define Input Paths (Assuming Production Output Locations)
RAG_PREDS="results_rag_tta/predictions_test.csv"
GEO_PREDS="results_geometric/predictions_test.csv" # Need to ensure this exists!
NAS_PREDS="results_nas/predictions_test.csv"       # Need to ensure this exists!
GT_FILE="data/crispr_test_ground_truth.csv"        # Need to ensure this exists!

# Run Ensemble
echo "Running Ensemble..."
python src/model/ensemble.py \
    --rag $RAG_PREDS \
    --geo $GEO_PREDS \
    --nas $NAS_PREDS \
    --gt $GT_FILE

if [ $? -eq 0 ]; then
    echo "✅ Ensemble SOTA calculation complete!"
else
    echo "❌ Ensemble failed"
    exit 1
fi
