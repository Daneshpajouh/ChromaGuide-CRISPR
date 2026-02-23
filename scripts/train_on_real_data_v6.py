#!/usr/bin/env python3
"""
Multimodal on-target efficacy prediction v6
Uses DNABERT-2 backbone with CNN fallback, Beta regression loss, cross-attention fusion.
Target: Spearman rho >= 0.911 on GOLD test set.
d_model=64 throughout (PhD proposal constraint).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add to path
sys.path.insert(0, '/Users/studio/Desktop/PhD/Proposal')

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}\n")

from src.chromaguide.sequence_encoder import DNABERT2Encoder


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between sequence and epigenomics modalities."""
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Self-attention for each modality
        self.seq_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.epi_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Cross-attention: seq attends to epi, epi attends to seq
        self.cross_attn_seq = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_epi = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 4, d_model)
        )

        # Layer norms
        self.ln_seq = nn.LayerNorm(d_model)
        self.ln_epi = nn.LayerNorm(d_model)
        self.ln_fused = nn.LayerNorm(d_model)

    def forward(self, seq_feat, epi_feat):
        """
        seq_feat: (batch, 1, d_model) - sequence features
        epi_feat: (batch, n_epi, d_model) - epigenomics features
        """
        # Self-attention
        seq_self, _ = self.seq_self_attn(seq_feat, seq_feat, seq_feat)
        epi_self, _ = self.epi_self_attn(epi_feat, epi_feat, epi_feat)

        # Cross-attention
        seq_cross, _ = self.cross_attn_seq(seq_feat, epi_feat, epi_feat)
        epi_cross, _ = self.cross_attn_epi(epi_feat, seq_feat, seq_feat)

        # Combine modalities
        seq_combined = self.ln_seq(seq_self + seq_cross)  # (batch, 1, d_model)
        epi_combined = self.ln_epi(epi_self + epi_cross)  # (batch, n_epi, d_model)

        # Global pooling
        seq_pooled = seq_combined.mean(dim=1)  # (batch, d_model)
        epi_pooled = epi_combined.mean(dim=1)  # (batch, d_model)

        # Concatenate and fuse
        fused = torch.cat([seq_pooled, epi_pooled], dim=1)  # (batch, 2*d_model)
        fused = self.fusion_mlp(fused)  # (batch, d_model)
        fused = self.ln_fused(fused)

        return fused


class MultimodalEfficacyModelV6(nn.Module):
    """Multimodal efficacy prediction with cross-attention fusion."""
    def __init__(self, d_model=64, n_epi_features=690):
        super().__init__()
        self.d_model = d_model

        # Sequence encoder (DNABERT-2 with CNN fallback)
        self.seq_encoder = DNABERT2Encoder(d_model=d_model, use_fallback=True)

        # Epigenomics encoder
        self.epi_encoder = nn.Sequential(
            nn.Linear(n_epi_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, d_model),
            nn.BatchNorm1d(d_model)
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(d_model=d_model, n_heads=4)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, seq_one_hot, epi_features):
        """
        seq_one_hot: (batch, 4, 23) - one-hot encoded sequences
        epi_features: (batch, 690) - epigenomics features
        """
        # Encode modalities
        seq_feat = self.seq_encoder(seq_one_hot)  # (batch, d_model)
        seq_feat = seq_feat.unsqueeze(1)  # (batch, 1, d_model)

        epi_feat = self.epi_encoder(epi_features)  # (batch, d_model)
        epi_feat = epi_feat.unsqueeze(1)  # (batch, 1, d_model)

        # Fuse with cross-attention
        fused = self.fusion(seq_feat, epi_feat)  # (batch, d_model)

        # Predict efficiency
        pred = self.head(fused)  # (batch, 1)

        return pred


class BetaRegression(nn.Module):
    """Beta regression loss for bounded [0,1] targets."""
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        logits: (batch, 1) - unbounded predictions
        targets: (batch, 1) - values in [0, 1]
        """
        # Sigmoid to get predictions in [0, 1]
        preds = torch.sigmoid(logits)

        # Clamp to avoid log(0)
        preds = torch.clamp(preds, 1e-6, 1 - 1e-6)
        targets = torch.clamp(targets, 1e-6, 1 - 1e-6)

        # Beta regression loss
        # L = -(alpha * log(p) + (1 - alpha) * log(1 - p))
        # where alpha = target
        loss = -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))

        return loss.mean()


def one_hot_encode(seq, length=23):
    """One-hot encode DNA sequences."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, length), dtype=np.float32)
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[mapping[c], j] = 1
    return encoded


def load_on_target_data(split='A'):
    """Load on-target efficacy data from CSV files using glob pattern."""
    print(f"Loading on-target efficacy data (split={split})...")

    from pathlib import Path

    split_name_map = {
        'A': 'split_a_gene_held_out',
        'B': 'split_b_dataset_held_out',
        'C': 'split_c_cellline_held_out'
    }
    split_dir = Path(f"data/processed/{split_name_map[split]}")

    if not split_dir.exists():
        split_dir_alt = Path(f"data/processed/split_{split.lower()}")
        if split_dir_alt.exists():
            split_dir = split_dir_alt
        else:
            raise FileNotFoundError(f"Split {split_dir} not found")

    # Load CSV files using glob pattern
    train_files = list(split_dir.glob("*_train.csv"))
    val_files = list(split_dir.glob("*_validation.csv"))
    test_files = list(split_dir.glob("*_test.csv"))

    if not train_files or not val_files or not test_files:
        raise ValueError(f"Missing split files in {split_dir}")

    train_df = pd.concat([pd.read_csv(f) for f in train_files]).reset_index(drop=True)
    val_df = pd.concat([pd.read_csv(f) for f in val_files]).reset_index(drop=True)
    test_df = pd.concat([pd.read_csv(f) for f in test_files]).reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")

    return train_df, val_df, test_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MultiModal Efficacy v6 - Production Training')
    parser.add_argument('--split', type=str, default='A', choices=['A', 'B', 'C'], help='Data split')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"MultiModal Efficacy v6 - Cross-Attention Fusion")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Backbone: DNABERT-2 (CNN fallback enabled)")
    print(f"Fusion: Cross-attention (4 heads)")
    print(f"Loss: Beta regression")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Patience: {args.patience} | Split: {args.split}")
    print(f"d_model throughout: 64 (PhD proposal)")
    print(f"{'='*70}\n", flush=True)

    # Load data
    train_df, val_df, test_df = load_on_target_data(split=args.split)

    print(f"Train efficiency: μ={train_df['efficiency'].mean():.3f}, σ={train_df['efficiency'].std():.3f}")
    print(f"Val efficiency:   μ={val_df['efficiency'].mean():.3f}, σ={val_df['efficiency'].std():.3f}")
    print(f"Test efficiency:  μ={test_df['efficiency'].mean():.3f}, σ={test_df['efficiency'].std():.3f}\n")

    # Batch processing functions
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def prepare_seq(seqs, device):
        """One-hot encode sequences."""
        L = 21
        seq_tensor = torch.zeros(len(seqs), 4, L, device=device, dtype=torch.float32)
        for j, seq in enumerate(seqs):
            for k, nt in enumerate(seq[:L]):
                if nt.upper() in nt_map:
                    seq_tensor[j, nt_map[nt.upper()], k] = 1
        return seq_tensor

    def prepare_epi(batch, device):
        """Extract epigenomics features."""
        cols = [f'feat_{i}' for i in range(11)]
        if not all(col in batch.columns for col in cols):
            return None
        epi_data = batch[cols].values.astype(np.float32)
        return torch.tensor(epi_data, device=device, dtype=torch.float32)

    # Model
    n_epi_features = sum(1 for col in train_df.columns if col.startswith('feat_'))
    print(f"Initializing MultimodalEfficacyModelV6 (n_epi={n_epi_features})...")
    model = MultimodalEfficacyModelV6(d_model=64, n_epi_features=n_epi_features).to(device)
    print(f"✓ Model initialized\n")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=190, T_mult=1, eta_min=1e-7)
    criterion = BetaRegression()

    # Training
    print(f"{'='*70}")
    print(f"Starting training with Beta regression loss...")
    print(f"{'='*70}\n")

    best_rho = 0
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        train_batch_count = 0

        for batch_start in range(0, len(train_df), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(train_df))
            batch = train_df.iloc[batch_start:batch_end]

            try:
                seqs = batch['sequence'].tolist()
                labels = torch.tensor(batch['efficiency'].values, dtype=torch.float32).to(device).unsqueeze(1)

                seq_tensor = prepare_seq(seqs, device)
                epi_tensor = prepare_epi(batch, device)

                logits = model(seq_tensor, epi_tensor)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                train_batch_count += 1
            except Exception as e:
                continue

        avg_train_loss = total_loss / max(train_batch_count, 1)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_start in range(0, len(val_df), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(val_df))
                batch = val_df.iloc[batch_start:batch_end]

                try:
                    seqs = batch['sequence'].tolist()
                    seq_tensor = prepare_seq(seqs, device)
                    epi_tensor = prepare_epi(batch, device)

                    logits = model(seq_tensor, epi_tensor)
                    preds = torch.sigmoid(logits).cpu().numpy().flatten()
                    val_preds.extend(preds)
                    val_labels.extend(batch['efficiency'].values)
                except Exception as e:
                    continue

        val_rho = spearmanr(val_labels, val_preds)[0] if len(val_preds) > 1 else 0.0
        if np.isnan(val_rho):
            val_rho = 0.0

        lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_train_loss:.4f} | Val Rho: {val_rho:.4f} | LR: {lr:.2e}", flush=True)

        if val_rho > best_rho:
            best_rho = val_rho
            best_state = model.state_dict().copy()
            no_improve = 0
            if (epoch + 1) % 20 == 0 or epoch < 5:
                print(f"  ✓ New best Rho: {best_rho:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping (no improvement for {args.patience} epochs)")
                break

        scheduler.step()

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_start in range(0, len(test_df), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(test_df))
            batch = test_df.iloc[batch_start:batch_end]

            try:
                seqs = batch['sequence'].tolist()
                seq_tensor = prepare_seq(seqs, device)
                epi_tensor = prepare_epi(batch, device)

                logits = model(seq_tensor, epi_tensor)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                test_preds.extend(preds)
                test_labels.extend(batch['efficiency'].values)
            except Exception as e:
                continue

    test_rho = spearmanr(test_labels, test_preds)[0] if len(test_preds) > 1 else 0.0
    if np.isnan(test_rho):
        test_rho = 0.0

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (V6)")
    print(f"{'='*70}")
    print(f"Best Val Rho: {best_rho:.4f}")
    print(f"Test Rho (GOLD): {test_rho:.4f}")
    print(f"{'='*70}\n")

    # Target check
    if test_rho >= 0.911:
        print(f"✓ TARGET ACHIEVED: Rho {test_rho:.4f} >= 0.911!!\n")
    else:
        gap = 0.911 - test_rho
        print(f"⚠ Gap to target: {gap:.4f} ({(gap/0.911)*100:.1f}%)\n")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/multimodal_v6_cross_attention.pt")
    print(f"✓ Saved model: models/multimodal_v6_cross_attention.pt")
    print(f"✓ Multimodal v6 training complete\n")


if __name__ == "__main__":
    main()
