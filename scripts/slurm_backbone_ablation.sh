#!/bin/bash
#SBATCH --job-name=chromaguide_backbone_ablation
#SBATCH --account=def-kalegg
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/backbone_ablation_%j.out
#SBATCH --error=logs/backbone_ablation_%j.err
#SBATCH --mail-user=daneshpajouh@uottawa.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# ChromaGuide Backbone Ablation Study
# Compares 5 DNA sequence encoding architectures:
#   1. CNN-GRU (classical baseline, no pre-training)
#   2. DNABERT-2 (117M transformer, default)
#   3. Nucleotide Transformer (500M, long-range)
#   4. Caduceus-PS (CNN-Mamba hybrid, efficient)
#   5. Evo (7B foundation model, max transfer)
#
# Controlled experiment: same training budget, same hyperparams
# ============================================================================

set -e

echo "=========================================="
echo "ChromaGuide Backbone Ablation Study"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="

# Setup modules
module load python/3.10
module load cuda/12.2
module load cudnn/8.9.5.29_cuda12

cd ~/chromaguide_experiments
git fetch origin main && git reset --hard origin/main

mkdir -p logs
mkdir -p checkpoints/ablation
source ~/chromaguide_env/bin/activate

echo "Python: $(python --version)"
echo "CUDA: $(nvcc --version)"

# ============================================================================
# Create base config
# ============================================================================
cat > configs/ablation_base.json << 'EOF'
{
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
    "early_stopping_patience": 10
  },
  "regularization": {
    "use_mi_regularization": true,
    "lambda_mi": 0.01,
    "use_off_target": false
  },
  "evaluation": {
    "use_conformal": true,
    "run_statistical_tests": true
  }
}
EOF

# Prepare data once
if [ ! -f "data/deephf_processed.pkl" ]; then
    echo "Downloading DeepHF data..."
    python scripts/download_deepHF_data.py
    python scripts/prepare_real_data.py --dataset deephf --output data/deephf_processed.pkl
fi

# ============================================================================
# Define backbone architectures to test
# ============================================================================
BACKBONES=("cnn_gru" "dnabert2" "nucleotide_transformer" "caduceus_ps" "evo")

# ============================================================================
# Run ablation experiments
# ============================================================================
for backbone in "${BACKBONES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training with backbone: $backbone"
    echo "=========================================="

    # Create config for this backbone
    cat > configs/ablation_${backbone}.json << EOFCONFIG
{
  "model": {
    "backbone_type": "$backbone",
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
    "checkpoint_dir": "checkpoints/ablation/${backbone}"
  },
  "regularization": {
    "use_mi_regularization": true,
    "lambda_mi": 0.01,
    "use_off_target": false
  },
  "evaluation": {
    "use_conformal": true,
    "run_statistical_tests": true,
    "run_sota_comparison": false
  }
}
EOFCONFIG

    # Run training
    python -u src/training/train_chromaguide_v2.py \
      --config configs/ablation_${backbone}.json \
      --data data/deephf_processed.pkl \
      --device cuda \
      --seed 42 || echo "Warning: Training for $backbone encountered an error but continuing..."

    echo "Completed training for backbone: $backbone"
done

# ============================================================================
# Run formal ablation analysis
# ============================================================================
echo ""
echo "=========================================="
echo "Running Formal Ablation Analysis"
echo "=========================================="

python -u src/models/backbone_ablation.py \
  --experiment_dir checkpoints/ablation \
  --output results/ablation_study.json \
  --analyze_efficiency \
  --generate_figures

# ============================================================================
# Aggregate results
# ============================================================================
echo ""
echo "=========================================="
echo "Aggregating Ablation Results"
echo "=========================================="

python << 'PYEOF'
import json
import os
from pathlib import Path

results_dir = Path('checkpoints/ablation')
ablation_results = {}

for backbone_dir in results_dir.iterdir():
    if backbone_dir.is_dir():
        backbone_name = backbone_dir.name
        results_file = backbone_dir / 'training_results.json'

        if results_file.exists():
            with open(results_file, 'r') as f:
                ablation_results[backbone_name] = json.load(f)
                print(f"\n{backbone_name}:")
                if 'test_spearman' in ablation_results[backbone_name]:
                    print(f"  Spearman: {ablation_results[backbone_name]['test_spearman']:.4f}")
                if 'test_ndcg20' in ablation_results[backbone_name]:
                    print(f"  NDCG@20: {ablation_results[backbone_name]['test_ndcg20']:.4f}")

# Find best backbone
best_backbone = max(ablation_results.items(),
                   key=lambda x: x[1].get('test_spearman', 0))
print(f"\nBest backbone: {best_backbone[0]} with Spearman={best_backbone[1].get('test_spearman', 0):.4f}")

# Save summary
with open('results/ablation_summary.json', 'w') as f:
    json.dump({
        'all_results': ablation_results,
        'best_backbone': best_backbone[0],
        'best_spearman': best_backbone[1].get('test_spearman'),
    }, f, indent=2)

PYEOF

echo ""
echo "=========================================="
echo "Ablation Study Summary"
echo "=========================================="
cat results/ablation_summary.json | python -m json.tool

# ============================================================================
# Commit and push
# ============================================================================
echo ""
echo "Committing ablation results..."
git add checkpoints/ablation/ results/ablation* results/ablation_study.json
git commit -m "Backbone Ablation Study Results: Job $SLURM_JOB_ID"
git push origin main

echo ""
echo "=========================================="
echo "Ablation study completed successfully!"
echo "End Time: $(date)"
echo "=========================================="
