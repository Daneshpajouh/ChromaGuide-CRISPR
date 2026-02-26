#!/bin/bash
#SBATCH --job-name=chromaguide-thesis
#SBATCH --account=def-kwiese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/thesis_%j.out
#SBATCH --error=results/logs/thesis_%j.err

# ============================================================
# ChromaGuide: Generate Thesis Outputs
# Run AFTER all training/evaluation jobs complete
# ============================================================

echo "============================================"
echo "ChromaGuide Thesis Figure/Table Generation"
echo "Start: $(date)"
echo "============================================"

module load python/3.11

source $HOME/chromaguide_env/bin/activate
cd $HOME/chromaguide

python -m chromaguide.cli thesis --results-dir results/ 2>&1 | tee results/logs/thesis.log

echo "Thesis outputs complete: $(date)"
echo "Figures: results/figures/"
echo "Tables: results/tables/"
