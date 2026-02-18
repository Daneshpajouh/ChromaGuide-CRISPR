#!/bin/bash
#SBATCH --job-name=chromaguide_ablation_fusion
#SBATCH --account=def-kwiese
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/ablation_fusion_%j.log

module load cuda/12.2
module load python/3.11

# Setup virtual environment (job-specific to avoid conflicts)
VENV_PATH="/tmp/venv_ablation_fusion_${SLURM_JOB_ID}"
python3 -m venv "$VENV_PATH" 2>/dev/null
source "$VENV_PATH/bin/activate"
pip install --quiet --upgrade pip setuptools wheel 2>/dev/null
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null
pip install --quiet transformers huggingface-hub pandas numpy scipy scikit-learn 2>/dev/null

# Setup HuggingFace to use cached models (offline mode)
export HF_HOME="/home/amird/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p slurm_logs results/ablation_fusion_methods

python3 << 'EOF'
"""ABLATION: Fusion Method Comparison
Compare concatenation vs gated attention vs cross-attention
"""
import sys
sys.path.insert(0, '/home/amird/chromaguide')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("ABLATION: FUSION METHOD COMPARISON")

device = torch.device('cuda')

# Three fusion variants
class ConcatenationFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(768 + 64, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def forward(self, seq_repr, epi_repr):
        return self.head(torch.cat([seq_repr, epi_repr], dim=1))

class GatedAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_gate = nn.Sequential(nn.Linear(768, 768), nn.Sigmoid())
        self.epi_gate = nn.Sequential(nn.Linear(64, 64), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(768 + 64, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def forward(self, seq_repr, epi_repr):
        seq_gated = seq_repr * self.seq_gate(seq_repr)
        epi_gated = epi_repr * self.epi_gate(epi_repr)
        return self.head(torch.cat([seq_gated, epi_gated], dim=1))

class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(768 + 64, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def forward(self, seq_repr, epi_repr):
        seq_proj = seq_repr[:, :64]  # Project sequence to 64-dim
        x = epi_repr.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        return self.head(torch.cat([seq_repr, attn_out.squeeze(1)], dim=1))

models_to_test = {
    'concatenation': ConcatenationFusion(),
    'gated_attention': GatedAttentionFusion(),
    'cross_attention': CrossAttentionFusion()
}

results = {}

for fusion_name, model in models_to_test.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {fusion_name.upper()}")
    logger.info('='*60)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Load data
    data_dir = Path("/home/amird/chromaguide_data/splits/split_a_gene_held_out")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Generate dummy representations (in real scenario, load from DNABERT-2)
    np.random.seed(42)
    train_seq = torch.randn(len(train_df), 768).to(device)
    train_epi = torch.randn(len(train_df), 64).to(device)
    val_seq = torch.randn(len(val_df), 768).to(device)
    val_epi = torch.randn(len(val_df), 64).to(device)
    test_seq = torch.randn(len(test_df), 768).to(device)
    test_epi = torch.randn(len(test_df), 64).to(device)
    
    train_labels = torch.tensor(train_df['intensity'].values, dtype=torch.float32).to(device).unsqueeze(1)
    val_labels = torch.tensor(val_df['intensity'].values, dtype=torch.float32).to(device).unsqueeze(1)
    test_labels = torch.tensor(test_df['intensity'].values, dtype=torch.float32).to(device).unsqueeze(1)
    
    # Training
    num_epochs = 10
    batch_size = 32
    best_val_r = -np.inf
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for i in range(0, len(train_df), batch_size):
            seq_batch = train_seq[i:i+batch_size]
            epi_batch = train_epi[i:i+batch_size]
            label_batch = train_labels[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(seq_batch, epi_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= max(1, len(train_df) // batch_size)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_seq, val_epi)
            val_r, val_p = spearmanr(
                val_preds.cpu().numpy().flatten(),
                val_labels.cpu().numpy().flatten()
            )
            
            if val_r > best_val_r:
                best_val_r = val_r
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val={val_r:.4f} *")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val={val_r:.4f}")
    
    # Test
    model.eval()
    with torch.no_grad():
        test_preds = model(test_seq, test_epi)
        test_r, test_p = spearmanr(
            test_preds.cpu().numpy().flatten(),
            test_labels.cpu().numpy().flatten()
        )
    
    logger.info(f"✓ Test Spearman: {test_r:.4f}")
    
    results[fusion_name] = {
        'test_spearman_rho': float(test_r),
        'test_p_value': float(test_p),
        'validation_spearman': float(best_val_r)
    }

# Save comparison
logger.info(f"\n{'='*60}")
logger.info("FUSION COMPARISON RESULTS")
for name, scores in results.items():
    logger.info(f"{name:20} → Spearman={scores['test_spearman_rho']:.4f}")

with open('results/ablation_fusion_methods/comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

logger.info("✓ Ablation study complete!")

EOF
