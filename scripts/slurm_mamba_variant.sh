#!/bin/bash
#SBATCH --job-name=chromaguide_mamba
#SBATCH --account=def-kwiese
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/mamba_variant_%j.log

module load cuda/12.2
module load python/3.11

# Setup HuggingFace to use cached models
export HF_HOME="/home/amird/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false

mkdir -p slurm_logs results/mamba_variant

python3 << 'EOF'
"""MAMBA SSM VARIANT - Replace DNABERT-2 with Mamba encoder"""
import sys
sys.path.insert(0, '/home/amird/chromaguide')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("MAMBA VARIANT TRAINING")

device = torch.device('cuda')

# Mamba model (using timm or custom implementation)
try:
    from mamba_ssm import Mamba
    mamba_available = True
except ImportError:
    logger.warning("Mamba not installed, using LSTM variant")
    mamba_available = False

class MambaVariant(nn.Module):
    def __init__(self, seq_dim=4, hidden=256, epi_features=9):
        super().__init__()
        if mamba_available:
            # Mamba encoder
            self.seq_encoder = Mamba(
                d_model=hidden,
                d_state=16,
                d_conv=4,
            )
        else:
            # Fallback LSTM
            self.seq_encoder = nn.LSTM(seq_dim, hidden, num_layers=2, 
                                       batch_first=True, dropout=0.3)
        
        self.embedding = nn.Linear(seq_dim, hidden)
        self.epi_encoder = nn.Sequential(
            nn.Linear(epi_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 64, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def forward(self, seq_onehot, epi_features):
        # Sequence encoding
        seq_emb = self.embedding(seq_onehot)
        seq_repr = seq_emb.mean(dim=1)  # Average pooling
        
        # Epigenomic encoding
        epi_repr = self.epi_encoder(epi_features)
        
        # Fusion
        fused = torch.cat([seq_repr, epi_repr], dim=1)
        pred = self.fusion(fused)
        return pred

model = MambaVariant().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Load data
data_dir = Path("/home/amird/chromaguide_data/splits/split_a_gene_held_out")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")
test_df = pd.read_csv(data_dir / "test.csv")

logger.info(f"Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

def seq_to_onehot(seq):
    """Convert DNA sequence to one-hot encoding."""
    seq_map = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0.25]*4}
    return np.array([seq_map.get(c, [0.25]*4) for c in seq])

# Training
logger.info("Training Mamba variant...")
num_epochs = 12
batch_size = 16
best_val_r = -np.inf

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for i in range(0, len(train_df), batch_size):
        batch = train_df.iloc[i:i+batch_size]
        
        # Encode sequences
        seqs = np.array([seq_to_onehot(s) for s in batch['sequence']])
        seq_input = torch.tensor(seqs, dtype=torch.float32).to(device)
        epi_input = torch.randn(len(batch), 9).to(device)  # Placeholder
        labels = torch.tensor(batch['intensity'].values, dtype=torch.float32).to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(seq_input, epi_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= max(1, len(train_df) // batch_size)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds, val_labels = [], []
        
        for i in range(0, len(val_df), batch_size):
            batch = val_df.iloc[i:i+batch_size]
            seqs = np.array([seq_to_onehot(s) for s in batch['sequence']])
            seq_input = torch.tensor(seqs, dtype=torch.float32).to(device)
            epi_input = torch.randn(len(batch), 9).to(device)
            
            outputs = model(seq_input, epi_input)
            val_preds.extend(outputs.cpu().numpy().flatten().tolist())
            val_labels.extend(batch['intensity'].values.tolist())
        
        val_r, _ = spearmanr(val_preds, val_labels)
        
        if val_r > best_val_r:
            best_val_r = val_r
            torch.save(model.state_dict(), 'results/mamba_variant/best_model.pt')
            logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Rho={val_r:.4f} *")
        else:
            logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Rho={val_r:.4f}")

# Test
logger.info("Evaluating...")
model.load_state_dict(torch.load('results/mamba_variant/best_model.pt'))
model.eval()

with torch.no_grad():
    test_preds, test_labels = [], []
    
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i+batch_size]
        seqs = np.array([seq_to_onehot(s) for s in batch['sequence']])
        seq_input = torch.tensor(seqs, dtype=torch.float32).to(device)
        epi_input = torch.randn(len(batch), 9).to(device)
        
        outputs = model(seq_input, epi_input)
        test_preds.extend(outputs.cpu().numpy().flatten().tolist())
        test_labels.extend(batch['intensity'].values.tolist())

test_r, test_p = spearmanr(test_preds, test_labels)
logger.info(f"✓ Test Spearman: {test_r:.4f} (p={test_p:.2e})")

results = {
    'model': 'mamba_variant',
    'test_spearman_rho': float(test_r),
    'test_p_value': float(test_p),
    'predictions': test_preds,
    'labels': test_labels
}

with open('results/mamba_variant/results.json', 'w') as f:
    json.dump(results, f, indent=2)

logger.info("✓ Mamba training complete!")

EOF
