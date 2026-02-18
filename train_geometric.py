#!/usr/bin/env python3
"""
Geometric Biothermodynamics Training Script
Target: Train CRISPR model using Natural Gradient Descent on Fisher Manifold
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from sklearn.model_selection import train_test_split
import argparse
from torch.utils.data import DataLoader, Dataset

# Add project to path
sys.path.insert(0, '/scratch/amird/CRISPRO-MAMBA-X')

from src.model.geometric_optimizer import GeometricCRISPROptimizer

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model (Using DNABERT-2 as base)
    foundation_model = "zhihan1996/DNABERT-2-117M"
    max_seq_len = 512

    # Training
    batch_size = 32
    epochs = 20
    learning_rate = 1e-2  # Higher LR often works better with Natural Gradient
    damping = 1e-3

    # Data
    data_path = "/scratch/amird/CRISPRO-MAMBA-X/data/merged_crispr_data.csv"

    # Output
    output_dir = "./results_geometric"

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATASET & MODEL
# ============================================================================

class SimpleCRISPRDataset(Dataset):
    def __init__(self, sequences, efficiencies, tokenizer, max_len=512):
        self.sequences = sequences
        self.efficiencies = efficiencies
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx]).upper()
        eff = float(self.efficiencies[idx])

        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'val': torch.tensor(eff, dtype=torch.float32)
        }

class CRISPRRegressionModel(nn.Module):
    def __init__(self, foundation_model):
        super().__init__()
        self.bert = AutoModel.from_pretrained(foundation_model, trust_remote_code=True)
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids_or_dict, attention_mask=None):
        # Handle dict input (from DataLoader)
        if isinstance(input_ids_or_dict, dict):
            input_ids = input_ids_or_dict['input_ids']
            attention_mask = input_ids_or_dict['attention_mask']
        else:
            input_ids = input_ids_or_dict

        # Move to device if needed (handled by caller ideally, but safety check)
        if hasattr(input_ids, 'device') and input_ids.device != self.bert.device:
             input_ids = input_ids.to(self.bert.device)
             if attention_mask is not None:
                 attention_mask = attention_mask.to(self.bert.device)

        utils = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # DNABERT-2 uses mean pooling often, or CLS. Let's use CLS or first token.
        # Check config, but usually [:, 0, :] is safe for BERT-likes.
        # Actually DNABERT-2 might be EOS pooling. Let's start with CLS equivalent.
        if hasattr(utils, 'last_hidden_state'):
            last_hidden_state = utils.last_hidden_state
        else:
            last_hidden_state = utils[0]
        pooler_output = last_hidden_state[:, 0, :]
        return self.head(pooler_output)

# ============================================================================
# TRAINING LOOP (NATURAL GRADIENT)
# ============================================================================

def train_geometric():
    print("="*80)
    print("GEOMETRIC BIOTHERMODYNAMICS TRAINING (Natural Gradient)")
    print("="*80)

    # 1. Load Data
    print("\nLoading data...")
    if not os.path.exists(Config.data_path):
        raise FileNotFoundError(f"CRITICAL: Real data not found at {Config.data_path}. Synthetic data fallback is DISABLED.")

    df = pd.read_csv(Config.data_path)
    print(f"✓ Loaded {len(df)} samples")

    tokenizer = AutoTokenizer.from_pretrained(Config.foundation_model, trust_remote_code=True)

    dataset = SimpleCRISPRDataset(
        df['sequence'].tolist(),
        df['efficiency'].tolist(),
        tokenizer
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False)

    # 2. Initialize Model & Geometric Optimizer
    print("\nInitializing Geometry...")
    model = CRISPRRegressionModel(Config.foundation_model).to(Config.device)

    geo_opt = GeometricCRISPROptimizer(
        model,
        lr=Config.learning_rate,
        damping=Config.damping
    )

    # 3. Training Loop
    print(f"\nStarting training on {Config.device}...")
    best_rho = -1.0
    history = []

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            targets = batch['val'].to(Config.device).view(-1, 1)

            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            # Natural Gradient Step (The Magic)
            loss = geo_opt.step(inputs, targets)
            total_loss += loss

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        preds, acts = [], []
        all_outputs_list, all_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(Config.device)
                attention_mask = batch['attention_mask'].to(Config.device)
                targets = batch['val'].to(Config.device).view(-1, 1)

                out = model({'input_ids': input_ids, 'attention_mask': attention_mask})

                all_outputs_list.append(out.cpu())
                all_targets_list.append(targets.cpu())

        # Diagnostics
        all_outputs = torch.cat(all_outputs_list).flatten()
        all_targets = torch.cat(all_targets_list).flatten()
        pred_std = all_outputs.std().item()

        # Compute Spearman Rho
        if len(all_outputs) > 1:
             rho, _ = spearmanr(all_outputs.cpu().numpy(), all_targets.cpu().numpy())
        else:
             rho = 0.0

        print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {avg_loss:.4f} | Spearman ρ: {rho:.4f} | Pred Std: {pred_std:.4f}")

        if rho > best_rho:
            best_rho = rho
            # Save checkpoint
            torch.save(model.state_dict(), f"{Config.output_dir}/best_geometric_model.pt")

        # Save logs
        history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'rho': float(rho) if rho is not None else 0.0
        })
        pd.DataFrame(history).to_csv(f"{Config.output_dir}/training_log.csv", index=False)

    print(f"\n🏆 Best Result: ρ = {best_rho:.4f}")
    print("Geometric optimization complete.")

if __name__ == "__main__":
    os.makedirs(Config.output_dir, exist_ok=True)
    train_geometric()
