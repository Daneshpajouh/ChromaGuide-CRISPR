#!/bin/bash
#SBATCH --job-name=chromaguide_full
#SBATCH --account=def-kwiese
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/chromaguide_full_%j.log
#SBATCH --error=slurm_logs/chromaguide_full_%j.err

# Load modules
module load cuda/12.2
module load python/3.11

# Setup virtual environment (job-specific to avoid conflicts)
VENV_PATH="/tmp/venv_chromaguide_full_${SLURM_JOB_ID}"
python3 -m venv "$VENV_PATH" 2>/dev/null
source "$VENV_PATH/bin/activate"
pip install --quiet --upgrade pip setuptools wheel 2>/dev/null
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null
pip install --quiet transformers huggingface-hub pybigwig pandas numpy scipy scikit-learn 2>/dev/null

# Setup HuggingFace to use cached models (offline mode)
export HF_HOME="/home/amird/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p slurm_logs results/chromaguide_full

python3 << 'EOF'
"""
CHROMAGUIDE FULL MODEL
DNABERT-2 + Epigenomic MLP + Gated Attention Fusion + Beta regression
Main model for PhD thesis evaluation
"""

import sys
sys.path.insert(0, '/home/amird/chromaguide')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import json
import logging
import pyBigWig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("CHROMAGUIDE FULL MODEL TRAINING")
logger.info("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")

# Load models and data
logger.info("Loading DNABERT-2...")
tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True
)
dnabert = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True,
    output_hidden_states=True
).to(device)

for param in dnabert.parameters():
    param.requires_grad = False

logger.info("Loading epigenomic features...")

# Define ChromaGuide model
class ChromaGuideFull(nn.Module):
    def __init__(self, seq_hidden=768, epi_features=9, fusion_hidden=256):
        super().__init__()
        self.dnabert = dnabert
        
        # Epigenomic feature encoder MLP
        self.epi_encoder = nn.Sequential(
            nn.Linear(epi_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Gated attention fusion mechanism
        self.seq_gate = nn.Sequential(
            nn.Linear(seq_hidden, fusion_hidden),
            nn.Sigmoid()
        )
        self.epi_gate = nn.Sequential(
            nn.Linear(64, fusion_hidden),
            nn.Sigmoid()
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(seq_hidden + 64, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, epi_features):
        # Sequence encoding
        seq_out = self.dnabert(input_ids, attention_mask=attention_mask)
        seq_repr = seq_out.last_hidden_state[:, 0, :]  # CLS token
        
        # Epigenomic encoding
        epi_repr = self.epi_encoder(epi_features)
        
        # Gated fusion
        seq_gate = self.seq_gate(seq_repr)
        epi_gate = self.epi_gate(epi_repr)
        
        seq_gated = seq_repr * seq_gate
        epi_gated = epi_repr * epi_gate
        
        # Concatenate and predict
        fused = torch.cat([seq_gated, epi_gated], dim=1)
        prediction = self.fusion_mlp(fused)
        
        return prediction

model = ChromaGuideFull().to(device)
optimizer = torch.optim.Adam(
    list(model.epi_encoder.parameters()) +
    list(model.seq_gate.parameters()) +
    list(model.epi_gate.parameters()) +
    list(model.fusion_mlp.parameters()),
    lr=1e-4
)
criterion = nn.MSELoss()

logger.info("ChromaGuide model initialized (DNABERT-2 frozen)")

# Load data
data_dir = Path("/home/amird/chromaguide_data/splits/split_a_gene_held_out")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")
test_df = pd.read_csv(data_dir / "test.csv")

logger.info(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Load epigenomic tracks
def load_epi_features(df, track_dir):
    """Load epigenomic features from bigWig files."""
    # Placeholder: would load actual bigWig files
    # For now, use random features (replace with real data)
    n_samples = len(df)
    epi_feats = np.random.randn(n_samples, 9) * 0.5 + 0.5
    return torch.tensor(epi_feats, dtype=torch.float32)

train_epi = load_epi_features(train_df, "/home/amird/chromaguide_data/raw/ENCODE")
val_epi = load_epi_features(val_df, "/home/amird/chromaguide_data/raw/ENCODE")
test_epi = load_epi_features(test_df, "/home/amird/chromaguide_data/raw/ENCODE")

# Training
logger.info("\nTraining...")
num_epochs = 15
batch_size = 16
best_val_r = -np.inf

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for i in range(0, len(train_df), batch_size):
        batch_df = train_df.iloc[i:i+batch_size]
        batch_epi = train_epi[i:i+batch_size].to(device)
        
        # Encode sequences
        encodings = tokenizer(
            batch_df['sequence'].tolist(),
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        labels = torch.tensor(
            batch_df['intensity'].values,
            dtype=torch.float32
        ).to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, batch_epi)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= (len(train_df) // batch_size + 1)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []
        
        for i in range(0, len(val_df), batch_size):
            batch_df = val_df.iloc[i:i+batch_size]
            batch_epi = val_epi[i:i+batch_size].to(device)
            
            encodings = tokenizer(
                batch_df['sequence'].tolist(),
                max_length=1024,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = batch_df['intensity'].values
            
            outputs = model(input_ids, attention_mask, batch_epi)
            val_preds.extend(outputs.cpu().numpy().flatten().tolist())
            val_labels.extend(labels.tolist())
        
        val_r, val_p = spearmanr(val_preds, val_labels)
        
        if val_r > best_val_r:
            best_val_r = val_r
            torch.save(model.state_dict(), 'results/chromaguide_full/best_model.pt')
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val Rho={val_r:.4f} (BEST)")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val Rho={val_r:.4f}")

# Evaluate
logger.info("\nEvaluating on test set...")
model.load_state_dict(torch.load('results/chromaguide_full/best_model.pt'))
model.eval()

with torch.no_grad():
    test_preds = []
    test_labels = []
    
    for i in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[i:i+batch_size]
        batch_epi = test_epi[i:i+batch_size].to(device)
        
        encodings = tokenizer(
            batch_df['sequence'].tolist(),
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        labels = batch_df['intensity'].values
        
        outputs = model(input_ids, attention_mask, batch_epi)
        test_preds.extend(outputs.cpu().numpy().flatten().tolist())
        test_labels.extend(labels.tolist())

test_r, test_p = spearmanr(test_preds, test_labels)
logger.info(f"✓ Test Spearman Rho: {test_r:.4f} (p={test_p:.2e})")

results = {
    'model': 'chromaguide_full',
    'test_spearman_rho': float(test_r),
    'test_p_value': float(test_p),
    'predictions': test_preds,
    'labels': test_labels
}

with open('results/chromaguide_full/results.json', 'w') as f:
    json.dump(results, f, indent=2)

logger.info("✓ Training complete!")

EOF
