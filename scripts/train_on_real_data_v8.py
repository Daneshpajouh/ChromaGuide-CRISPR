#!/usr/bin/env python3
"""
MultiModal Efficacy v8 - Improved Multi-Head Attention Fusion
Diagnostic version with feature normalization and stronger fusion.
Target: Spearman rho >= 0.911 on test set (gene-held-out Split A)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}\n")

# Import from chromaguide
import sys
sys.path.insert(0, "/Users/studio/Desktop/PhD/Proposal/src")

# ============================================================================
# DATA LOADING
# ============================================================================

def one_hot_encode_sequence(seq, length=23):
    """One-hot encode DNA sequence to (4, length) for Conv1d."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, length), dtype=np.float32)
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[mapping[c], j] = 1
    return encoded

def load_on_target_data(split='A'):
    """Load multimodal efficacy prediction data for gene-held-out split."""
    data_dir = f"data/processed/split_a_gene_held_out"

    print(f"Loading on-target efficacy data (split={split})...")

    train_files = sorted(glob.glob(f"{data_dir}/*_train.csv"))
    val_files = sorted(glob.glob(f"{data_dir}/*_validation.csv"))
    test_files = sorted(glob.glob(f"{data_dir}/*_test.csv"))

    def load_split(files):
        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    df_train = load_split(train_files)
    df_val = load_split(val_files)
    df_test = load_split(test_files)

    # Extract sequences and features (correct column names)
    def prepare_data(df):
        # One-hot encode sequences
        X_seq = np.array([one_hot_encode_sequence(seq) for seq in df['sequence'].values])
        # Get epigenomic features (feat_0 through feat_10)
        epi_cols = [col for col in df.columns if col.startswith('feat_')]
        X_epi = df[epi_cols].values.astype(np.float32)
        y = df['efficiency'].values.astype(np.float32)
        return X_seq, X_epi, y

    X_train_seq, X_train_epi, y_train = prepare_data(df_train)
    X_val_seq, X_val_epi, y_val = prepare_data(df_val)
    X_test_seq, X_test_epi, y_test = prepare_data(df_test)

    print(f"Train: {len(X_train_seq)} | Val: {len(X_val_seq)} | Test: {len(X_test_seq)}")
    print(f"Train efficiency: μ={y_train.mean():.3f}, σ={y_train.std():.3f}")
    print(f"Val efficiency:   μ={y_val.mean():.3f}, σ={y_val.std():.3f}")
    print(f"Test efficiency:  μ={y_test.mean():.3f}, σ={y_test.std():.3f}")

    # Compute feature statistics for normalization
    epi_mean = X_train_epi.mean(axis=0)
    epi_std = X_train_epi.std(axis=0) + 1e-8
    print(f"\nEpigenomic features (raw):")
    print(f"  Mean per feature: {epi_mean}")
    print(f"  Std per feature:  {epi_std}\n")

    return (X_train_seq, X_train_epi, y_train), \
           (X_val_seq, X_val_epi, y_val), \
           (X_test_seq, X_test_epi, y_test), \
           (epi_mean, epi_std)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultiheadAttentionFusion(nn.Module):
    """Multi-head attention fusion (stronger than simple gating)."""
    def __init__(self, d_model=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0

        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections for attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, h_seq, h_epi):
        """Cross-attention: use sequence as query, epigenomics as key/value."""
        batch_size = h_seq.shape[0]

        # Project to q, k, v
        q = self.q_proj(h_seq).view(batch_size, 1, self.n_heads, self.head_dim)
        k = self.k_proj(h_epi).view(batch_size, 1, self.n_heads, self.head_dim)
        v = self.v_proj(h_epi).view(batch_size, 1, self.n_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, n_heads, 1, head_dim)
        k = k.transpose(1, 2)  # (batch, n_heads, 1, head_dim)
        v = v.transpose(1, 2)  # (batch, n_heads, 1, head_dim)

        # Compute attention scores
        scores = torch.einsum('bnqd,bnkd->bnqk', q, k) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.einsum('bnqk,bnkd->bnqd', attn_weights, v)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, self.d_model)
        out = self.out_proj(context)

        # Residual connection
        fused = h_seq + out
        fused = self.layer_norm(fused)

        return fused


class MultimodalEfficacyModelV8(nn.Module):
    """Multimodal efficacy prediction with multi-head attention fusion."""
    def __init__(self, d_model=64, n_epi_features=11):
        super().__init__()
        self.d_model = d_model

        # Stronger 1D-CNN to encode one-hot sequences to d_model
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # Global max pool
        )
        self.seq_proj = nn.Linear(32, d_model)
        self.seq_norm = nn.LayerNorm(d_model)

        # Stronger epigenomic feature encoder with feature-wise normalization
        self.epi_input_proj = nn.Linear(n_epi_features, 128)  # Project up first
        self.epi_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, d_model),
            nn.ReLU()
        )
        self.epi_norm = nn.LayerNorm(d_model)

        # Multi-head attention fusion (stronger than gating)
        self.fusion = MultiheadAttentionFusion(d_model=d_model, n_heads=4, dropout=0.2)

        # Output head for regression
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1] for efficacy
        )

    def forward(self, seq_onehot, epi_features):
        # seq_onehot: (batch, 4, 23) one-hot encoded sequences
        # epi_features: (batch, n_epi) - already normalized

        # Encode sequences through CNN
        h_seq = self.seq_cnn(seq_onehot)  # (batch, 32, 1)
        h_seq = h_seq.squeeze(-1)  # (batch, 32)
        h_seq = self.seq_proj(h_seq)  # (batch, d_model)
        h_seq = self.seq_norm(h_seq)

        # Encode epigenomic features
        h_epi = self.epi_input_proj(epi_features)  # (batch, 128)
        h_epi = self.epi_encoder(h_epi)  # (batch, d_model)
        h_epi = self.epi_norm(h_epi)

        # Fuse with multi-head attention
        h_fused = self.fusion(h_seq, h_epi)  # (batch, d_model)

        # Output prediction
        logits = self.output_head(h_fused)  # (batch, 1)

        return logits


class SequenceOnlyModel(nn.Module):
    """Sequence-only baseline for comparison."""
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model

        # 1D-CNN encoder
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.seq_proj = nn.Linear(32, d_model)
        self.seq_norm = nn.LayerNorm(d_model)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_onehot, epi_features=None):
        h_seq = self.seq_cnn(seq_onehot)
        h_seq = h_seq.squeeze(-1)
        h_seq = self.seq_proj(h_seq)
        h_seq = self.seq_norm(h_seq)
        logits = self.output_head(h_seq)
        return logits

# ============================================================================
# BETA REGRESSION LOSS
# ============================================================================

class BetaRegression(nn.Module):
    """Beta regression loss for bounded [0,1] predictions."""
    def forward(self, logits, targets):
        logits = torch.clamp(logits, 1e-7, 1 - 1e-7)
        loss = -(targets * torch.log(logits) + (1 - targets) * torch.log(1 - logits))
        return loss.mean()

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, criterion):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for X_seq, X_epi, y in dataloader:
        X_seq = X_seq.to(device)
        X_epi = X_epi.to(device)
        y = y.to(device).view(-1, 1)

        # Forward pass
        logits = model(X_seq, X_epi)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * len(y)

    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, device):
    """Validate and compute Spearman rho."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_seq, X_epi, y in dataloader:
            X_seq = X_seq.to(device)
            X_epi = X_epi.to(device)

            logits = model(X_seq, X_epi)

            all_preds.append(logits.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    rho, _ = spearmanr(targets, preds)
    return rho

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("=" * 70)
    print("MultiModal Efficacy v8 - Multi-Head Attention Fusion (Improved)")
    print("=" * 70)
    print("Backbone: CNN on one-hot sequences (64->64->32)")
    print("Epigenomic: Deeper encoder (11->128->256->128->64)")
    print("Fusion: Multi-head attention with residuals")
    print("Epochs: 300 | Batch: 128 | LR: 5e-4")
    print("Patience: 100 | Gradient clip: 1.0")
    print("Scheduler: CosineAnnealingWarmRestarts (T_0=150)")
    print("=" * 70)
    print()

    # Load data
    (X_train_seq, X_train_epi, y_train), \
    (X_val_seq, X_val_epi, y_val), \
    (X_test_seq, X_test_epi, y_test), \
    (epi_mean, epi_std) = load_on_target_data(split='A')

    # Normalize epigenomic features using training set statistics
    X_train_epi_norm = (X_train_epi - epi_mean) / epi_std
    X_val_epi_norm = (X_val_epi - epi_mean) / epi_std
    X_test_epi_norm = (X_test_epi - epi_mean) / epi_std

    print("✓ Epigenomic features normalized using training statistics\n")

    # Prepare loaders with normalized features
    def prepare_loader(X_seq, X_epi, y, batch_size=128, shuffle=True):
        dataset = TensorDataset(
            torch.from_numpy(X_seq).float(),
            torch.from_numpy(X_epi).float(),
            torch.from_numpy(y).float()
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = prepare_loader(X_train_seq, X_train_epi_norm, y_train, batch_size=128, shuffle=True)
    val_loader = prepare_loader(X_val_seq, X_val_epi_norm, y_val, batch_size=128, shuffle=False)
    test_loader = prepare_loader(X_test_seq, X_test_epi_norm, y_test, batch_size=128, shuffle=False)

    # Train sequence-only baseline first
    print("=" * 70)
    print("BASELINE: Sequence-Only Model")
    print("=" * 70)

    seq_only_model = SequenceOnlyModel(d_model=64).to(device)
    optimizer = optim.AdamW(seq_only_model.parameters(), lr=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=1)
    criterion = BetaRegression()

    best_rho = -1
    patience = 0
    patience_max = 50

    for epoch in range(1, 151):  # 150 epochs for baseline
        train_loss = train_epoch(seq_only_model, train_loader, optimizer, device, criterion)
        val_rho = validate(seq_only_model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val Rho: {val_rho:.4f}")

        if val_rho > best_rho:
            best_rho = val_rho
            patience = 0
        else:
            patience += 1

        if patience >= patience_max:
            break

        scheduler.step()

    seq_only_test = validate(seq_only_model, test_loader, device)
    print(f"✓ Sequence-only TEST Rho: {seq_only_test:.4f}\n")

    # Now train multimodal model
    print("=" * 70)
    print("MULTIMODAL: v8 Multi-Head Attention Fusion")
    print("=" * 70)

    model = MultimodalEfficacyModelV8(d_model=64, n_epi_features=X_train_epi.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=1)
    criterion = BetaRegression()

    best_rho = -1
    patience = 0
    patience_max = 100  # More patience for multimodal

    for epoch in range(1, 301):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_rho = validate(model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val Rho: {val_rho:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_rho > best_rho:
            best_rho = val_rho
            patience = 0
        else:
            patience += 1

        if patience >= patience_max:
            print(f"\nEarly stopping at epoch {epoch}")
            break

        scheduler.step()

    # Evaluate on test set
    print("\n" + "=" * 70)
    test_rho = validate(model, test_loader, device)
    print(f"SEQUENCE-ONLY TEST Rho:  {seq_only_test:.4f}")
    print(f"MULTIMODAL v8 TEST Rho:  {test_rho:.4f}")
    print(f"Improvement from epigenomics: {test_rho - seq_only_test:+.4f}")
    print(f"Target: >= 0.911")
    gap = 0.911 - test_rho
    if gap > 0:
        print(f"⚠ Gap to target: {gap:.4f} ({100*gap/0.911:.1f}%)")
    else:
        print(f"✓ EXCEEDED target by {-gap:.4f} ({100*(-gap)/0.911:.1f}%)")
    print("=" * 70)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/multimodal_v8_multihead_fusion.pt")
    print(f"\n✓ Saved model: models/multimodal_v8_multihead_fusion.pt")
    print(f"✓ Multimodal v8 training complete\n")

if __name__ == "__main__":
    main()
