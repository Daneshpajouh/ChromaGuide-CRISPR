#!/bin/bash
#SBATCH --job-name=chromaguide_v2_deephf
#SBATCH --account=def-kalegg
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_deephf_%j.out
#SBATCH --error=logs/train_deephf_%j.err
#SBATCH --mail-user=daneshpajouh@uottawa.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# ChromaGuide V2 Training - DeepHF Dataset
# Narval Supercomputer (CCDB-4 H100 Nodes)
# 
# Integrates all 9 critical modules:
#   1. LeakageControlledSplits (gene-held-out split)
#   2. BetaRegressionHead (bounded [0,1] predictions)
#   3. ConformalPrediction (uncertainty quantification)
#   4. MINEClubRegularizer (feature independence)
#   5. OffTargetModule (off-target risk)
#   6. DesignScoreAggregator (multi-objective scoring)
#   7. StatisticalTests (significance testing)
#   8. SOTAComparison (baseline comparison)
#   9. BackboneAblation (architecture exploration)
#
# Target: rho >= 0.911 on DeepHF (vs CCL/MoFF baseline)
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "ChromaGuide V2 Training - DeepHF"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="

# Setup module environment
module load python/3.10
module load cuda/12.2
module load cudnn/8.9.5.29_cuda12

echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version)"

# Navigate to project root
cd ~/chromaguide_experiments

# Sync latest code from GitHub
echo "Syncing code from GitHub..."
git fetch origin main
git reset --hard origin/main

# Create logs directory
mkdir -p logs
mkdir -p checkpoints

# Activate virtual environment
echo "Activating Python environment..."
source ~/chromaguide_env/bin/activate

# Verify PyTorch with CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Echo GPU info
echo "GPU Info:"
nvidia-smi

# ============================================================================
# PHASE 1: Data Preparation
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Data Preparation"
echo "=========================================="

if [ ! -f "data/deephf_processed.pkl" ]; then
    echo "DeepHF data not found. Downloading and preprocessing..."
    python scripts/download_deepHF_data.py
    python scripts/prepare_real_data.py --dataset deephf --output data/deephf_processed.pkl
else
    echo "DeepHF data already prepared."
fi

# ============================================================================
# PHASE 2: Training with ChromaGuide V2
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: ChromaGuide V2 Training"
echo "=========================================="

# Create config if not exists
cat > configs/train_v2_deephf.json << 'EOF'
{
  "model": {
    "backbone_type": "dnabert2",
    "beta_hidden_dims": [256, 128],
    "dropout": 0.1
  },
  "data": {
    "dataset": "deephf",
    "split_type": "gene_held_out",
    "batch_size": 32,
    "val_ratio": 0.1,
    "test_ratio": 0.2
  },
  "training": {
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "early_stopping_patience": 10,
    "checkpoint_dir": "checkpoints/deephf_v2"
  },
  "regularization": {
    "use_mi_regularization": true,
    "lambda_mi": 0.01,
    "use_off_target": false
  },
  "evaluation": {
    "use_conformal": true,
    "run_statistical_tests": true,
    "run_sota_comparison": true
  }
}
EOF

echo "Config created at configs/train_v2_deephf.json"
cat configs/train_v2_deephf.json

# Run training with GPU parallelization
echo "Starting training..."
python -u src/training/train_chromaguide_v2.py \
  --config configs/train_v2_deephf.json \
  --data data/deephf_processed.pkl \
  --device cuda \
  --seed 42

# ============================================================================
# PHASE 3: Evaluation
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 3: Statistical Evaluation"
echo "=========================================="

# Run statistical tests
python -u src/evaluation/statistical_tests.py \
  --predictions checkpoints/deephf_v2/predictions.pkl \
  --ground_truth data/deephf_processed.pkl \
  --output results/statistical_eval_deephf.json

# Run SOTA comparison
python -u src/evaluation/sota_comparison.py \
  --our_predictions checkpoints/deephf_v2/predictions.pkl \
  --baseline_db src/evaluation/sota_baselines.json \
  --output results/sota_comparison_deephf.json

# ============================================================================
# PHASE 4: Results Aggregation & Reporting
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 4: Results Aggregation"
echo "=========================================="

# Aggregate results
python -u scripts/aggregate_training_results.py \
  --training_results checkpoints/deephf_v2/training_results.json \
  --statistical_results results/statistical_eval_deephf.json \
  --sota_results results/sota_comparison_deephf.json \
  --output results/deephf_v2_complete_report.json

# Create summary report
echo ""
echo "=========================================="
echo "Training Results Summary"
echo "=========================================="
cat results/deephf_v2_complete_report.json | python -m json.tool

# ============================================================================
# Commit & Push Results
# ============================================================================
echo ""
echo "Committing results to GitHub..."
git add checkpoints/deephf_v2/ results/ logs/
git commit -m "ChromaGuide V2 DeepHF Results: Job $SLURM_JOB_ID"
git push origin main

echo ""
echo "=========================================="
echo "Training completed successfully!"
echo "End Time: $(date)"
echo "=========================================="
