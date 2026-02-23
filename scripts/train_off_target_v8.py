#!/usr/bin/env python3
"""
Off-Target v8 - Deeper CNN Architecture + Optimized Ensemble
Improved CNN depth and ensemble voting strategy.
Target: AUROC >= 0.99 (ensemble of 10 models)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================================
# DATA LOADING
# ============================================================================

def one_hot_encode(seq, length=23):
    """One-hot encode DNA sequence."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((length, 4), dtype=np.float32)
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[j, mapping[c]] = 1
    return encoded.T  # (4, length)

def load_crispoff_data(path="data/raw/crisprofft/crispoff_data.txt"):
    """Load CRISPRoff data."""
    seqs = []
    labels = []

    print(f"Loading {path}...")
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip header
            parts = line.strip().split('\t')
            if len(parts) < 35:
                continue
            try:
                guide_seq = parts[21]  # guide_seq is at column 21
                target_status = parts[33]  # target_status is at column 33
                if target_status not in ["ON", "OFF"] or not guide_seq or len(guide_seq) < 20:
                    continue
                seqs.append(guide_seq)
                labels.append(1 if target_status == "OFF" else 0)  # 1=OFF-target, 0=ON-target
            except:
                continue

    total = len(seqs)
    on_target = sum(1 for l in labels if l == 0)
    off_target = sum(1 for l in labels if l == 1)
    print(f"Loaded {total} | ON: {on_target} | OFF: {off_target}")

    # Train/val/test split (80/10/10)
    tr_idx = int(0.8 * len(seqs))
    va_idx = int(0.9 * len(seqs))

    X_train = np.array([one_hot_encode(s) for s in seqs[:tr_idx]])
    X_val = np.array([one_hot_encode(s) for s in seqs[tr_idx:va_idx]])
    X_test = np.array([one_hot_encode(s) for s in seqs[va_idx:]])

    y_train = np.array(labels[:tr_idx], dtype=np.float32)
    y_val = np.array(labels[tr_idx:va_idx], dtype=np.float32)
    y_test = np.array(labels[va_idx:], dtype=np.float32)

    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Train OFF-target rate: {y_train.mean():.4f}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ============================================================================
# DEEPER CNN ARCHITECTURE
# ============================================================================

class DeepCNNScorer(nn.Module):
    """Deeper 1D-CNN for off-target classification."""
    def __init__(self):
        super().__init__()

        # Deeper convolutional layers with multiple kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(4, 128, kernel_size=3, padding=1),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
        ])

        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for c in [128, 128, 64, 64, 32]])

        # Multi-scale pooling: max + avg on different kernel sizes
        self.multi_kernel_convs = nn.ModuleList([
            nn.Conv1d(4, 32, kernel_size=4, padding=1),
            nn.Conv1d(4, 32, kernel_size=5, padding=2),
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
        ])

        # Adaptive pooling for variable length
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # FC head: concatenate all features
        # Main path: 32 from last conv
        # Avg pooling: 32 from avg pool
        # Multi-kernel: 32*3 = 96 from three kernels
        # Total: 32 + 32 + 96 = 160
        fc_input = 32 + 32 + 96

        self.fc = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Save original input for parallel multi-kernel path
        x_orig = x

        # Main path: deep conv stack (sequential)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)  # Apply conv to output of previous layer
            x = bn(x)
            x = torch.relu(x)

        # Max and average pool on final activation
        max_feat = self.pool(x).squeeze(-1)  # (batch, 32)
        avg_feat = self.avg_pool(x).squeeze(-1)

        # Multi-kernel features (parallel to main path, applied to original input)
        multi_feats = []
        for conv in self.multi_kernel_convs:
            feat = conv(x_orig)  # Apply to original input
            feat = torch.relu(feat)
            feat = self.pool(feat).squeeze(-1)  # (batch, 32)
            multi_feats.append(feat)

        multi_feat = torch.cat(multi_feats, dim=1)  # (batch, 96)

        # Concatenate all features
        combined = torch.cat([max_feat, avg_feat, multi_feat], dim=1)

        # FC head
        logits = self.fc(combined)

        return logits

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).view(-1, 1)

        logits = model(X)

        # Focal loss for class imbalance
        pos_weight = torch.tensor(214.5, device=device)
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    """Validate and compute AUROC."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    auroc = roc_auc_score(targets, preds)
    return auroc

def train_one_model(train_loader, val_loader, device, seed=0, epochs=200):
    """Train a single model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DeepCNNScorer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

    best_auroc = 0
    patience = 0
    patience_max = 30

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_auroc = validate(model, val_loader, device)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience = 0
        else:
            patience += 1

        if patience >= patience_max:
            break

        scheduler.step()

        if epoch % 50 == 0:
            print(f"    Epoch {epoch:3d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f}")

    return model, best_auroc

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Off-Target v8 - Deeper CNN + Multi-Scale Features")
    print("=" * 70)
    print("Architecture: 5-layer CNN + multi-kernel parallel path")
    print("FC: 256->128->64->1 with dropout")
    print("Loss: BCEWithLogitsLoss(pos_weight=214.5)")
    print("Ensemble: 10 models with seeds 0-9")
    print("=" * 70)
    print()

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_crispoff_data()

    # Data loaders
    def get_loader(X, y, batch_size=256, shuffle=True):
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float()
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = get_loader(X_train, y_train, batch_size=256, shuffle=True)
    val_loader = get_loader(X_val, y_val, batch_size=256, shuffle=False)
    test_loader = get_loader(X_test, y_test, batch_size=256, shuffle=False)

    # Train ensemble
    models = []
    aurocs = []

    print("Training 10-model ensemble...\n")

    for model_id in range(10):
        print(f"Model {model_id + 1}/10 (seed={model_id}):")
        model, best_auroc = train_one_model(train_loader, val_loader, device, seed=model_id, epochs=200)
        models.append(model)
        aurocs.append(best_auroc)
        print(f"    → Best AUROC: {best_auroc:.4f}\n")

    # Evaluate ensemble
    print("=" * 70)
    print("ENSEMBLE EVALUATION")
    print("=" * 70)

    ensemble_model = nn.ModuleList(models).to(device)

    all_test_preds = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            model_preds = []
            for model in models:
                logits = model(X)
                probs = torch.sigmoid(logits).cpu().numpy()
                model_preds.append(probs)
            model_preds = np.array(model_preds)  # (10, batch, 1)

            # Ensemble voting: average probabilities
            ensemble_probs = model_preds.mean(axis=0)
            all_test_preds.append(ensemble_probs)

    ensemble_preds = np.concatenate(all_test_preds).flatten()

    ensemble_auroc = roc_auc_score(y_test, ensemble_preds)

    print(f"\nIndividual model AUROCs:")
    for i, auroc in enumerate(aurocs):
        print(f"  Model {i+1}: {auroc:.4f}")

    print(f"\nMean individual AUROC: {np.mean(aurocs):.4f}")
    print(f"Std individual AUROC: {np.std(aurocs):.4f}")
    print(f"Ensemble Test AUROC: {ensemble_auroc:.4f}")
    print(f"Target: >= 0.99")

    gap = 0.99 - ensemble_auroc
    if gap > 0:
        print(f"⚠ Gap to target: {gap:.4f} ({100*gap/0.99:.1f}%)")
    else:
        print(f"✓ EXCEEDED target by {-gap:.4f}")

    print("=" * 70)

    # Save models
    os.makedirs("models", exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f"models/off_target_v8_model_{i}.pt")

    print(f"\n✓ Saved 10 models to models/off_target_v8_model_*.pt")
    print(f"✓ Off-target v8 training complete\n")

if __name__ == "__main__":
    main()
