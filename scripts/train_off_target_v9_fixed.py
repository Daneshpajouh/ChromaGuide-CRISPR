#!/usr/bin/env python3
"""
V9 OFF-TARGET: Maximum-power architecture to EXCEED AUROC >= 0.99 target
Simplified for actual CRISPRoffT data structure
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}\n")


def one_hot_encode(seq, length=23):
    """One-hot encode DNA sequence."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    vec = np.zeros((length, 4), dtype=np.float32)
    seq = str(seq).upper()[:length]
    for i, c in enumerate(seq):
        if i < length and c in mapping:
            vec[i, mapping[c]] = 1
    return vec.flatten()


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        class_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (class_weight * focal_weight * ce).mean()


class TransformerOffTarget(nn.Module):
    """Transformer-based off-target classifier."""
    def __init__(self, d_model=128, n_heads=8, dropout=0.3):
        super().__init__()

        # Embedding
        self.embed = nn.Linear(4, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 23, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.ln = nn.LayerNorm(d_model)

        # Classification head with residuals
        self.head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: (batch, 92) = 23 * 4
        x = x.view(x.size(0), 23, 4)  # (batch, 23, 4)
        x = self.embed(x)  # (batch, 23, d_model)
        x += self.pos_enc
        x = self.transformer(x)  # (batch, 23, d_model)
        x = self.ln(x)
        x = x.mean(dim=1)  # Global pooling
        logit = self.head(x).squeeze(1)  # (batch,)
        return logit


def load_crispofft_data(data_path):
    """Load CRISPRoffT data."""
    print(f"Loading CRISPRoffT data from {data_path}...")

    seqs, labels = [], []

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            parts = line.strip().split('\t')
            if len(parts) < 35:
                continue

            try:
                guide = parts[21]
                target_status = parts[33] if len(parts) > 33 else "ON"

                if target_status not in ["ON", "OFF"]:
                    continue
                if not guide or len(guide) < 20:
                    continue

                label = 1.0 if target_status == "OFF" else 0.0
                seqs.append(guide)
                labels.append(label)
            except:
                continue

    print(f"  Loaded {len(seqs)} sequences")

    X = np.array([one_hot_encode(s) for s in seqs]).astype(np.float32)
    y = np.array(labels).astype(np.float32)

    print(f"  ON: {int((y==0).sum())}, OFF: {int((y==1).sum())}")
    print(f"  Ratio: {(y==1).sum()/(y==0).sum():.3f}:1\n")

    return X, y


def train_model(model, train_loader, X_val, y_val, seed=0, epochs=300):
    """Train a single off-target model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)

    best_auc = 0
    patience = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for seq_batch, label_batch in train_loader:
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device)

            logits = model(seq_batch)
            loss = criterion(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Val
        with torch.no_grad():
            model.eval()
            val_logits = model(X_val)
            val_probs = torch.sigmoid(val_logits)
            val_auc = roc_auc_score(y_val.cpu().numpy(), val_probs.cpu().numpy())

        if epoch % 30 == 0:
            print(f"  E{epoch:3d} | Loss {train_loss:.4f} | Val AUROC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= 20:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_auc


def main():
    print("="*70)
    print("V9 OFF-TARGET: EXCEED 0.99 AUROC TARGET")
    print("="*70 + "\n")

    # Load data
    data_path = '/Users/studio/Desktop/PhD/Proposal/data/raw/crisprofft/CRISPRoffT_all_targets.txt'
    X, y = load_crispofft_data(data_path)

    # Split
    np.random.seed(42)
    n = len(X)
    idx = np.random.permutation(n)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:n_train+n_val]], y[idx[n_train:n_train+n_val]]
    X_test, y_test = X[idx[n_train+n_val:]], y[idx[n_train+n_val:]]

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}\n")

    # Weighted loader
    pos_weight = ((y_train==0).sum() / (y_train==1).sum()) * 2.0
    weights = np.zeros_like(y_train)
    weights[y_train == 0] = 1.0
    weights[y_train == 1] = pos_weight

    sampler = WeightedRandomSampler(weights, len(y_train), replacement=True)
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=256, sampler=sampler)

    # Train ensemble
    models = []
    val_aucs = []

    n_models = 20

    for i in range(n_models):
        if i % 5 == 0:
            print(f"\n{'='*70}")
            print(f"MODELS {i+1}-{min(i+5, n_models)}/{n_models}")
            print(f"{'='*70}")

        d_model = np.random.choice([64, 128, 192])
        model = TransformerOffTarget(d_model=d_model)
        trained_model, val_auc = train_model(model, train_loader, X_val, y_val, seed=i, epochs=300)

        models.append(trained_model)
        val_aucs.append(val_auc)

        torch.save(trained_model.state_dict(), f'/Users/studio/Desktop/PhD/Proposal/models/off_target_v9_seed{i}.pt')
        print(f"    Model {i+1}: Val AUROC {val_auc:.4f}")

    # Test ensemble
    print(f"\n{'='*70}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*70}")

    X_test_t = torch.FloatTensor(X_test).to(device)

    ensemble_probs = np.zeros(len(y_test))

    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(X_test_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            ensemble_probs += probs

    ensemble_probs /= len(models)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)

    print(f"\nIndividual Val AUROC: {[f'{a:.4f}' for a in val_aucs]}")
    print(f"Mean: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"\nEnsemble Test AUROC: {ensemble_auc:.4f}")
    print(f"Target: 0.99")
    print(f"Achievement: {ensemble_auc/0.99*100:.1f}%")

    if ensemble_auc >= 0.99:
        print("\n✅ TARGET ACHIEVED!")
    else:
        gap = 0.99 - ensemble_auc
        print(f"\n⚠️ Gap: {gap:.4f} ({gap/0.99*100:.1f}%)")

    # Save ensemble
    torch.save({
        'models': [m.state_dict() for m in models],
        'ensemble_probs': ensemble_probs,
        'val_aucs': val_aucs,
        'test_auc': ensemble_auc
    }, '/Users/studio/Desktop/PhD/Proposal/models/off_target_v9_ensemble.pt')

    print(f"\n✓ Ensemble saved\n")


if __name__ == '__main__':
    main()
