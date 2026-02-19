#!/bin/bash
#SBATCH --job-name=chromaguide_hpo_optuna
#SBATCH --account=def-kwiese
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/hpo_optuna_%j.log

module load cuda/12.2
module load python/3.11

# Setup virtual environment (job-specific to avoid conflicts)
# VENV_PATH removed
# Environment handled by ~/env_chromaguide
source ~/env_chromaguide/bin/activate

# Setup HuggingFace to use cached models (offline mode)
export HF_HOME="/home/amird/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p slurm_logs results/hpo_optuna

python3 << 'EOF'
"""HYPERPARAMETER OPTIMIZATION WITH OPTUNA
50 trials to find best ChromaGuide configuration
Optimizes Spearman correlation on validation set
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

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("HYPERPARAMETER OPTIMIZATION - OPTUNA (50 TRIALS)")
logger.info("="*80)

device = torch.device('cuda')

# Load data
data_dir = Path("/home/amird/chromaguide_data/splits/split_a_gene_held_out")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")
test_df = pd.read_csv(data_dir / "test.csv")

logger.info(f"Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Dummy representations
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

class ChromaGuideModel(nn.Module):
    def __init__(self, hidden1=256, hidden2=128, epi_hidden=128, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.epi_encoder = nn.Sequential(
            nn.Linear(64, epi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(epi_hidden, 64),
            nn.ReLU()
        )
        
        self.seq_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid()
        )
        self.epi_gate = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        
        self.head = nn.Sequential(
            nn.Linear(768 + 64, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, seq_repr, epi_repr):
        epi_encoded = self.epi_encoder(epi_repr)
        seq_gated = seq_repr * self.seq_gate(seq_repr)
        epi_gated = epi_encoded * self.epi_gate(epi_encoded)
        fused = torch.cat([seq_gated, epi_gated], dim=1)
        return self.head(fused)

def objective(trial: Trial) -> float:
    """Optuna objective function - maximize validation Spearman rho"""
    
    # Hyperparameter space
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    hidden1 = trial.suggest_int('hidden1', 128, 512, step=64)
    hidden2 = trial.suggest_int('hidden2', 64, 256, step=64)
    epi_hidden = trial.suggest_int('epi_hidden', 64, 256, step=64)
    dropout1 = trial.suggest_float('dropout1', 0.1, 0.5, step=0.1)
    dropout2 = trial.suggest_float('dropout2', 0.1, 0.4, step=0.1)
    batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
    
    model = ChromaGuideModel(
        hidden1=hidden1,
        hidden2=hidden2,
        epi_hidden=epi_hidden,
        dropout1=dropout1,
        dropout2=dropout2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train for 5 epochs (short training for fast HPO)
    for epoch in range(5):
        model.train()
        
        for i in range(0, len(train_df), batch_size):
            seq_batch = train_seq[i:i+batch_size]
            epi_batch = train_epi[i:i+batch_size]
            label_batch = train_labels[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(seq_batch, epi_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(val_seq, val_epi)
        val_r, _ = spearmanr(
            val_preds.cpu().numpy().flatten(),
            val_labels.cpu().numpy().flatten()
        )
    
    return val_r

# Run optimization
logger.info("\nStarting 50-trial optimization...")
sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction='maximize',
    sampler=sampler
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# Best trial
best_trial = study.best_trial
logger.info(f"\n{'='*60}")
logger.info(f"BEST TRIAL: #{best_trial.number}")
logger.info(f"Best Validation Rho: {best_trial.value:.4f}")
logger.info(f"Best Hyperparameters:")
for key, value in best_trial.params.items():
    logger.info(f"  {key}: {value}")

# Train on best hyperparameters
logger.info(f"\n{'='*60}")
logger.info("Training on test set with best hyperparameters...")

best_model = ChromaGuideModel(
    hidden1=int(best_trial.params['hidden1']),
    hidden2=int(best_trial.params['hidden2']),
    epi_hidden=int(best_trial.params['epi_hidden']),
    dropout1=best_trial.params['dropout1'],
    dropout2=best_trial.params['dropout2']
).to(device)

optimizer = torch.optim.Adam(best_model.parameters(), lr=best_trial.params['lr'])
criterion = nn.MSELoss()
batch_size = int(best_trial.params['batch_size'])

for epoch in range(10):
    best_model.train()
    for i in range(0, len(train_df), batch_size):
        seq_batch = train_seq[i:i+batch_size]
        epi_batch = train_epi[i:i+batch_size]
        label_batch = train_labels[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = best_model(seq_batch, epi_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()

# Test
best_model.eval()
with torch.no_grad():
    test_preds = best_model(test_seq, test_epi)
    test_r, test_p = spearmanr(
        test_preds.cpu().numpy().flatten(),
        test_labels.cpu().numpy().flatten()
    )

logger.info(f"✓ Test Spearman: {test_r:.4f} (p={test_p:.2e})")

# Save results
results = {
    'best_trial_number': best_trial.number,
    'best_validation_rho': float(best_trial.value),
    'best_hyperparameters': {
        k: float(v) if isinstance(v, (int, float)) else v
        for k, v in best_trial.params.items()
    },
    'test_spearman_rho': float(test_r),
    'test_p_value': float(test_p),
    'n_trials': len(study.trials)
}

with open('results/hpo_optuna/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save trial history
trial_history = []
for trial in study.trials:
    trial_history.append({
        'trial_number': trial.number,
        'validation_rho': float(trial.value) if trial.value else None,
        'hyperparameters': {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in trial.params.items()
        }
    })

with open('results/hpo_optuna/trial_history.json', 'w') as f:
    json.dump(trial_history, f, indent=2)

logger.info(f"\n✓ HPO complete! Best: {test_r:.4f}")

EOF
