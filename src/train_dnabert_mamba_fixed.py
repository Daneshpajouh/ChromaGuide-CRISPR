"""Training script for DNABERT-2 -> Adapter -> Mamba-2 pipeline.

This script provides a minimal training implementation for CRISPR sgRNA
efficacy prediction using DNABERT-2 embeddings and Mamba architecture.
"""

import warnings
warnings.filterwarnings("once")

from __future__ import annotations
import argparse
import logging
import os
import random
import time
from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    raise ImportError("PyTorch is required to run DNABERT-Mamba training!")

import numpy as np

from src.utils.device import DeviceManager, warn_if_pythonpath
from src.evaluation.metrics import spearman_correlation, compute_regression_metrics

LOG = logging.getLogger("train_dnabert_mamba")


class SeqDataset(Dataset):
    def __init__(self, seqs: List[str], targets: np.ndarray):
        assert len(seqs) == len(targets)
        self.seqs = seqs
        self.targets = targets

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], float(self.targets[idx])


def train_epoch(model, train_loader, optimizer, criterion, device, dnabert_encoder=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_seen = 0
    train_preds = []
    train_ys = []
    
    for seqs, targets in train_loader:
        # Safe tensor conversion for targets
        if isinstance(targets, torch.Tensor):
            targets = targets.clone().detach().float()
        else:
            targets = torch.as_tensor(targets, dtype=torch.float32)
        targets = device.to_device(targets)
        
        # Get DNABERT token-level embeddings where possible
        try:
            emb = dnabert_encoder.embed(seqs, pooled='none') if dnabert_encoder else None
        except Exception:
            emb = None
        
        # Forward pass through model
        if emb is not None:
            pred = model(emb, seqs, pooled='none').squeeze()
        else:
            pred = model(seqs).squeeze()
        
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        n_seen += len(seqs)
        train_preds.extend(pred.detach().cpu().numpy().tolist())
        train_ys.extend(targets.detach().cpu().numpy().tolist())
    
    return total_loss / len(train_loader), train_preds, train_ys


def validate(model, val_loader, criterion, device, dnabert_encoder=None):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    val_preds = []
    val_ys = []
    
    with torch.no_grad():
        for seqs, targets in val_loader:
            if isinstance(targets, torch.Tensor):
                targets = targets.clone().detach().float()
            else:
                targets = torch.as_tensor(targets, dtype=torch.float32)
            targets = device.to_device(targets)
            
            try:
                emb = dnabert_encoder.embed(seqs, pooled='none') if dnabert_encoder else None
            except Exception:
                emb = None
            
            if emb is not None:
                pred = model(emb, seqs, pooled='none').squeeze()
            else:
                pred = model(seqs).squeeze()
            
            loss = criterion(pred, targets)
            total_loss += loss.item()
            val_preds.extend(pred.cpu().numpy().tolist())
            val_ys.extend(targets.cpu().numpy().tolist())
    
    return total_loss / len(val_loader), val_preds, val_ys


def train(args):
    """Main training function."""
    LOG.info(f"Starting training with {args.epochs} epochs, batch_size={args.batch_size}")
    
    # Initialize device manager
    device = DeviceManager()
    LOG.info(f"Using device: {device.device}")
    
    # Load data (minimal for now)
    # In production, load from data/processed/
    LOG.info("Loading dataset...")
    
    # Synthetic data for testing
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    n_samples = 1000 if not args.use_mini else 100
    seq_len = 23  # Standard sgRNA length
    
    seqs = [''.join(random.choices('ACGT', k=seq_len)) for _ in range(n_samples)]
    targets = np.random.rand(n_samples).astype(np.float32)
    
    # Train/val split
    split_idx = int(0.8 * n_samples)
    train_seqs, val_seqs = seqs[:split_idx], seqs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    train_ds = SeqDataset(train_seqs, train_targets)
    val_ds = SeqDataset(val_seqs, val_targets)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    LOG.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Initialize DNABERT encoder if available
    dnabert_encoder = None
    if not args.trust_remote_code:
        LOG.warning("--trust_remote_code not set; DNABERT-2 may not load properly")
    
    try:
        if args.hf_token:
            os.environ["HF_TOKEN"] = args.hf_token
        
        from src.models.dnabert_encoder import DNABERTEncoder
        dnabert_encoder = DNABERTEncoder(
            model_name=args.dnabert_name,
            device=device.device,
            trust_remote_code=args.trust_remote_code
        )
        LOG.info(f"Loaded DNABERT encoder: {args.dnabert_name}")
    except Exception as e:
        LOG.warning(f"Could not load DNABERT encoder: {e}")
        LOG.info("Proceeding without DNABERT embeddings")
    
    # Initialize Mamba model
    try:
        from src.models.mamba_model import MambaEncoder
        mamba = MambaEncoder(
            d_model=args.adapter_out_dim,
            n_layers=args.n_layers,
            d_inner=args.d_inner,
            vocab_size=5,  # ACGT + pad
            dropout=0.1
        )
        model = device.to_device(mamba)
        LOG.info("Initialized Mamba model")
    except Exception as e:
        LOG.error(f"Failed to initialize Mamba model: {e}")
        # Fallback to simple MLP
        LOG.info("Using fallback MLP model")
        
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim=23*4, hidden_dim=256, output_dim=1):
                super().__init__()
                self.embed = nn.Embedding(5, 64)  # ACGT + pad
                self.fc1 = nn.Linear(23 * 64, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
                self.relu = nn.ReLU()
                
            def forward(self, seqs, *args, **kwargs):
                # Convert sequences to indices
                char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
                if isinstance(seqs[0], str):
                    batch_indices = []
                    for seq in seqs:
                        indices = [char_to_idx.get(c, 4) for c in seq[:23]]
                        # Pad if needed
                        indices += [4] * (23 - len(indices))
                        batch_indices.append(indices)
                    x = torch.tensor(batch_indices, device=self.embed.weight.device)
                else:
                    x = seqs
                x = self.embed(x)
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                return self.fc2(x)
        
        model = device.to_device(SimpleMLP())
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val = float('inf')
    epochs_no_improve = 0
    
    LOG.info("Starting training loop...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_preds, train_ys = train_epoch(
            model, train_loader, optimizer, criterion, device, dnabert_encoder
        )
        val_loss, val_preds, val_ys = validate(
            model, val_loader, criterion, device, dnabert_encoder
        )
        
        scheduler.step(val_loss)
        
        # Compute metrics
        train_corr = spearman_correlation(train_preds, train_ys)
        val_corr = spearman_correlation(val_preds, val_ys)
        
        LOG.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Corr: {train_corr:.4f} | Val Corr: {val_corr:.4f}"
        )
        
        # Early stopping check
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            # Save checkpoint
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, f"{model.__class__.__name__}.pt")
            torch.save(model.state_dict(), path)
            LOG.info(f"Saved best model to {path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                LOG.info(f"Early stopping at epoch {epoch}")
                break
    
    LOG.info("Training complete!")
    return model


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dnabert_name", type=str, default="dna_bert_2")
    parser.add_argument("--adapter_kind", type=str, default="lora")
    parser.add_argument("--adapter_r", type=int, default=16)
    parser.add_argument("--adapter_out_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_inner", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use_mini", action="store_true", help="Use mini dataset")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--output_dir", type=str, default="models/mamba")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = cli()
    train(args)
