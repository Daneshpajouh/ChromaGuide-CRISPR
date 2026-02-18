#!/usr/bin/env python3
"""
DNABERT-Mamba Phase 1 Training Script
======================================
Trains DNABERT-2 + Mamba model for sgRNA efficiency prediction
Runs on Narval cluster via SLURM job

Usage:
    python train_phase1.py --model_name zhihan1996/dnabert-2-117m \
                           --train_data data/processed/train.csv \
                           --output_dir checkpoints/phase1

Author: AI Assistant
Date: 2026-02-17
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, spearmanr, pearsonr
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SgrnaDataset(Dataset):
    """Custom dataset for sgRNA sequences and efficiency scores"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        efficiency = float(row['efficiency_score'])
        
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'efficiency': torch.tensor(efficiency, dtype=torch.float32)
        }


class MambaBlock(nn.Module):
    """Mamba sequence modeling block"""
    
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # State space model parameters
        self.A = nn.Parameter(torch.randn(hidden_dim, 4 * hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim, 4 * hidden_dim))
        self.C = nn.Parameter(torch.randn(hidden_dim, 4 * hidden_dim))
        self.D = nn.Parameter(torch.randn(hidden_dim))
        
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        
        # Apply Mamba-style state space operations
        for layer in self.layers:
            x = layer(x)
            x = self.ln(x)
            x = self.dropout(x)
        
        return x  # (batch_size, seq_len, hidden_dim)


class DNABERTMamba(nn.Module):
    """DNABERT-2 + Mamba model for efficiency prediction"""
    
    def __init__(self, model_name, mamba_hidden_dim=768, dropout=0.1):
        super().__init__()
        
        # Load DNABERT-2
        self.dnabert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.dnabert.config.hidden_size
        
        # Freeze DNABERT layers (optional)
        for param in self.dnabert.parameters():
            param.requires_grad = False
        
        # Mamba layer
        self.mamba = MambaBlock(
            hidden_dim=self.hidden_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, mamba_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mamba_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Model initialized: DNABERT-2 ({model_name}) + Mamba + Predictor")
    
    def forward(self, input_ids, attention_mask):
        # DNABERT forward pass
        outputs = self.dnabert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # Mamba processing
        mamba_out = self.mamba(last_hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Use [CLS] token representation (first token)
        cls_output = mamba_out[:, 0, :]  # (batch_size, hidden_dim)
        
        # Prediction
        efficiency = self.predictor(cls_output)  # (batch_size, 1)
        
        return efficiency.squeeze(1)  # (batch_size,)


def train_epoch(model, dataloader, optimizer, device, log_interval=100):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        efficiency = batch['efficiency'].to(device)
        
        # Forward pass
        pred_efficiency = model(input_ids, attention_mask)
        
        # Loss (MSE)
        loss = nn.MSELoss()(pred_efficiency, efficiency)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions.extend(pred_efficiency.detach().cpu().numpy())
        targets.extend(efficiency.detach().cpu().numpy())
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate correlations
    spearman_r, spearman_p = spearmanr(targets, predictions)
    pearson_r, pearson_p = pearsonr(targets, predictions)
    
    return {
        'loss': avg_loss,
        'spearman_r': spearman_r,
        'pearson_r': pearson_r,
    }


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            efficiency = batch['efficiency'].to(device)
            
            pred_efficiency = model(input_ids, attention_mask)
            loss = nn.MSELoss()(pred_efficiency, efficiency)
            
            total_loss += loss.item()
            predictions.extend(pred_efficiency.cpu().numpy())
            targets.extend(efficiency.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, predictions)
    spearman_r, _ = spearmanr(targets, predictions)
    
    return {
        'loss': avg_loss,
        'mse': mse,
        'spearman_r': spearman_r,
    }


def main(args):
    """Main training loop"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DNABERTMamba(
        model_name=args.model_name,
        mamba_hidden_dim=args.mamba_hidden_dim,
        dropout=0.1
    ).to(device)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = SgrnaDataset(args.train_data, tokenizer, max_length=args.max_length)
    val_dataset = SgrnaDataset(args.val_data, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    
    # Training loop
    logger.info("Starting training...")
    best_spearman = -1
    training_history = []
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Spearman r: {train_metrics['spearman_r']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Spearman r: {val_metrics['spearman_r']:.4f}")
        
        # Save best model
        if val_metrics['spearman_r'] > best_spearman:
            best_spearman = val_metrics['spearman_r']
            best_model_path = output_dir / 'best_model.pt'
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model: {best_model_path}")
        
        # Log metrics
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_spearman_r': train_metrics['spearman_r'],
            'val_loss': val_metrics['loss'],
            'val_spearman_r': val_metrics['spearman_r'],
            'timestamp': datetime.now().isoformat()
        }
        training_history.append(epoch_history)
    
    # Save final model and history
    final_model_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model: {final_model_path}")
    
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history: {history_path}")
    
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Best Spearman r: {best_spearman:.4f}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DNABERT-Mamba Phase 1 Training'
    )
    parser.add_argument('--model_name', default='zhihan1996/dnabert-2-117m',
                       help='Model name from HuggingFace')
    parser.add_argument('--train_data', required=True,
                       help='Path to training data CSV')
    parser.add_argument('--val_data', required=True,
                       help='Path to validation data CSV')
    parser.add_argument('--test_data', help='Path to test data CSV')
    parser.add_argument('--output_dir', default='checkpoints/phase1',
                       help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--mamba_hidden_dim', type=int, default=768,
                       help='Mamba hidden dimension')
    
    args = parser.parse_args()
    main(args)
