#!/bin/bash
#SBATCH --job-name=chromaguide_ablation_modality
#SBATCH --account=def-kwiese
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/ablation_modality_%j.log

module load cuda/12.2
module load python/3.11

# Setup HuggingFace to use cached models
export HF_HOME="/home/amird/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false

mkdir -p slurm_logs results/ablation_modality

python3 << 'EOF'
"""ABLATION: Modality Importance
Compare sequence-only vs full multimodal to show epigenomic contribution
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

logger.info("ABLATION: MODALITY IMPORTANCE")

device = torch.device('cuda')

# Model without epigenomic features
class SeqOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def forward(self, seq_repr, epi_repr=None):
        return self.head(seq_repr)

# Model with epigenomic features
class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.epi_encoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(768 + 64, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def forward(self, seq_repr, epi_repr):
        epi_encoded = self.epi_encoder(epi_repr)
        return self.head(torch.cat([seq_repr, epi_encoded], dim=1))

models_to_test = {
    'sequence_only': SeqOnlyModel(),
    'multimodal': MultimodalModel()
}

results = {}

# Load data once
data_dir = Path("/home/amird/chromaguide_data/splits/split_a_gene_held_out")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")
test_df = pd.read_csv(data_dir / "test.csv")

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

for model_name, model in models_to_test.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name.upper()}")
    logger.info('='*60)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
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
            if 'multimodal' in model_name:
                outputs = model(seq_batch, epi_batch)
            else:
                outputs = model(seq_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= max(1, len(train_df) // batch_size)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if 'multimodal' in model_name:
                val_preds = model(val_seq, val_epi)
            else:
                val_preds = model(val_seq)
            
            val_r, _ = spearmanr(
                val_preds.cpu().numpy().flatten(),
                val_labels.cpu().numpy().flatten()
            )
            
            if val_r > best_val_r:
                best_val_r = val_r
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val={val_r:.4f} *")
    
    # Test
    model.eval()
    with torch.no_grad():
        if 'multimodal' in model_name:
            test_preds = model(test_seq, test_epi)
        else:
            test_preds = model(test_seq)
        
        test_r, test_p = spearmanr(
            test_preds.cpu().numpy().flatten(),
            test_labels.cpu().numpy().flatten()
        )
    
    logger.info(f"✓ Test Spearman: {test_r:.4f}")
    
    results[model_name] = {
        'test_spearman_rho': float(test_r),
        'test_p_value': float(test_p)
    }

# Save results
logger.info(f"\n{'='*60}")
logger.info("MODALITY ABLATION RESULTS")
seq_only_r = results['sequence_only']['test_spearman_rho']
multimodal_r = results['multimodal']['test_spearman_rho']
improvement = multimodal_r - seq_only_r
logger.info(f"Sequence-only:  {seq_only_r:.4f}")
logger.info(f"Multimodal:     {multimodal_r:.4f}")
logger.info(f"Improvement:    {improvement:+.4f} ({improvement/seq_only_r*100:+.1f}%)")

with open('results/ablation_modality/comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

logger.info("✓ Modality ablation complete!")

EOF
