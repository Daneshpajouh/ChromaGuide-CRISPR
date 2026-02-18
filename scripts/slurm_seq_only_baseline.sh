#!/bin/bash
#SBATCH --job-name=chromaguide_seq_baseline
#SBATCH --account=def-kwiese
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/seq_baseline_%j.log
#SBATCH --error=slurm_logs/seq_baseline_%j.err

# Load modules
module load cuda/12.2
module load python/3.11

# Setup virtual environment (job-specific to avoid conflicts)
VENV_PATH="/tmp/venv_seq_only_${SLURM_JOB_ID}"
python3 -m venv "$VENV_PATH" 2>/dev/null
source "$VENV_PATH/bin/activate"
pip install --quiet --upgrade pip setuptools wheel 2>/dev/null
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null
pip install --quiet transformers huggingface-hub pandas numpy scipy scikit-learn 2>/dev/null

# Setup HuggingFace to use cached models and offline mode
export HF_HOME="/home/amird/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

# Create output directory
mkdir -p slurm_logs
mkdir -p results/seq_only_baseline

# Run training
python3 << 'EOF'
"""
SEQUENCE-ONLY BASELINE
DNABERT-2 + Beta regression head
Establishes lower bound on performance
"""

import sys
sys.path.insert(0, '/home/amird/chromaguide')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("SEQUENCE-ONLY BASELINE TRAINING")
logger.info("=" * 80)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")
logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

# Load DNABERT-2
logger.info("Loading DNABERT-2 model...")
tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True,
    output_hidden_states=True
)
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False  # Freeze DNABERT-2

logger.info(f"DNABERT-2 loaded ({model.config.hidden_size} hidden dims)")

# Add regression head
class SeqOnlyModel(torch.nn.Module):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()  # Efficacy is [0, 1]
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # Use CLS token (first token)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        prediction = self.regression_head(cls_hidden)
        return prediction

model_ft = SeqOnlyModel(model, model.config.hidden_size).to(device)
optimizer = torch.optim.Adam(model_ft.regression_head.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

logger.info("Model created (DNABERT-2 frozen + trainable regression head)")

# Load training data (Split A, gene-held-out)
data_dir = Path("/home/amird/chromaguide_data/splits/split_a_gene_held_out")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")
test_df = pd.read_csv(data_dir / "test.csv")

logger.info(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Batch training function
def encode_sequences(sequences, tokenizer, device, max_length=1024):
    """Tokenize and encode sequences."""
    encodings = tokenizer(
        sequences.tolist(),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    return encodings['input_ids'].to(device), encodings['attention_mask'].to(device)

# Train
logger.info("\nTraining...")
num_epochs = 10
batch_size = 16
best_val_r = -np.inf

for epoch in range(num_epochs):
    # Training loop
    model_ft.train()
    train_loss = 0.0
    
    for i in range(0, len(train_df), batch_size):
        batch = train_df.iloc[i:i+batch_size]
        
        input_ids, attention_mask = encode_sequences(
            batch['sequence'].values, tokenizer, device
        )
        labels = torch.tensor(batch['intensity'].values, dtype=torch.float32).to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model_ft(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= (len(train_df) // batch_size + 1)
    
    # Validation
    model_ft.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []
        
        for i in range(0, len(val_df), batch_size):
            batch = val_df.iloc[i:i+batch_size]
            input_ids, attention_mask = encode_sequences(
                batch['sequence'].values, tokenizer, device
            )
            labels = batch['intensity'].values
            
            outputs = model_ft(input_ids, attention_mask)
            val_preds.extend(outputs.cpu().numpy().flatten().tolist())
            val_labels.extend(labels.tolist())
        
        val_r, val_p = spearmanr(val_preds, val_labels)
        
        if val_r > best_val_r:
            best_val_r = val_r
            torch.save(model_ft.state_dict(), 'results/seq_only_baseline/best_model.pt')
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val Rho={val_r:.4f} (BEST)")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Val Rho={val_r:.4f}")

# Evaluate on test
logger.info("\nEvaluating on test set...")
model_ft.load_state_dict(torch.load('results/seq_only_baseline/best_model.pt'))
model_ft.eval()

with torch.no_grad():
    test_preds = []
    test_labels = []
    
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i+batch_size]
        input_ids, attention_mask = encode_sequences(
            batch['sequence'].values, tokenizer, device
        )
        labels = batch['intensity'].values
        
        outputs = model_ft(input_ids, attention_mask)
        test_preds.extend(outputs.cpu().numpy().flatten().tolist())
        test_labels.extend(labels.tolist())

test_r, test_p = spearmanr(test_preds, test_labels)
logger.info(f"✓ Test Spearman Rho: {test_r:.4f} (p={test_p:.2e})")

# Save predictions
results = {
    'model': 'seq_only_baseline',
    'test_spearman_rho': float(test_r),
    'test_p_value': float(test_p),
    'predictions': test_preds,
    'labels': test_labels
}

with open('results/seq_only_baseline/results.json', 'w') as f:
    json.dump(results, f, indent=2)

logger.info("✓ Training complete!")

EOF

logger.info "SLURM job completed"
