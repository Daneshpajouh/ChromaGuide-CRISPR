#!/usr/bin/env python3
"""
Off-target prediction v7 - Lean 1D-CNN Scorer
Simplified architecture for fast training on MPS device.
Uses Conv1d layers to learn k-mer patterns.
10-model ensemble with different random seeds.
Target: AUROC >= 0.99
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}\n")


def one_hot_encode(seq, length=23):
    """One-hot encode: (4, 23) for Conv1d."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, length), dtype=np.float32)
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[mapping[c], j] = 1
    return encoded

def load_crispoff_data(data_path):
    """Load CRISPRoff dataset."""
    seqs, labels = [], []
    print(f"Loading {data_path}...")

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split('\t')
            if len(parts) < 35:
                continue
            try:
                guide_seq = parts[21]
                target_status = parts[33]
                if target_status not in ["ON", "OFF"] or not guide_seq or len(guide_seq) < 20:
                    continue
                seqs.append(guide_seq)
                labels.append(1 if target_status == "OFF" else 0)
            except:
                continue

    X = np.array([one_hot_encode(seq) for seq in seqs]).astype(np.float32)
    y = np.array(labels).astype(np.float32)
    print(f"Loaded {len(seqs)} | ON: {(y==0).sum():.0f} | OFF: {(y==1).sum():.0f}\n")
    return X, y



class LeanCNNScorer(nn.Module):
    """Lightweight 1D-CNN for off-target scoring."""
    def __init__(self, num_filters=32):
        super().__init__()
        # Simple CNN blocks learning k-mer patterns
        self.conv = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8),  # Reduce to 8 features
        )
        self.head = nn.Sequential(
            nn.Linear(num_filters * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, 4, 23)
        x = self.conv(x)  # (batch, num_filters, 8)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.head(x)  # (batch, 1) logits
        return x



def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device).view(-1, 1)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    probs = np.concatenate(probs).flatten()
    labels = np.concatenate(labels).flatten()
    return roc_auc_score(labels, probs)

def train_one_model(X_train, y_train, X_val, y_val, seed, epochs=200, batch_size=512, lr=5e-4):
    """Train single CNN model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=batch_size
    )

    model = LeanCNNScorer(num_filters=32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([200.0], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

    best_auroc = 0
    patience, patience_max = 0, 30

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_auroc = validate(model, val_loader)

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience = 0
        else:
            patience += 1

        if patience >= patience_max:
            print(f"    → Best AUROC: {best_auroc:.4f}")
            break

        scheduler.step()

    return best_auroc


def main():
    print("=" * 70)
    print("Off-Target v7 - Lean 1D-CNN (Proof-of-Concept)")
    print("=" * 70)
    print()

    # Load & split
    X, y = load_crispoff_data("data/raw/crisprofft/crispoff_data.txt")

    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n_tr, n_val = int(0.8 * len(X)), int(0.1 * len(X))

    X_train, y_train = X[idx[:n_tr]], y[idx[:n_tr]]
    X_val, y_val = X[idx[n_tr:n_tr+n_val]], y[idx[n_tr:n_tr+n_val]]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Total: {len(X)}\n")

    # Train ensemble
    results = []
    for i in range(10):
        print(f"Model {i+1}/10 (seed={i}):")
        auroc = train_one_model(X_train, y_train, X_val, y_val, seed=i, epochs=200)
        results.append(auroc)
        print()

    print("=" * 70)
    print(f"ENSEMBLE RESULTS:")
    print(f"  Mean AUROC: {np.mean(results):.4f} ± {np.std(results):.4f}")
    print(f"  Range: [{np.min(results):.4f}, {np.max(results):.4f}]")
    print(f"  Target: >= 0.99")
    print("=" * 70)

if __name__ == "__main__":
    main()
