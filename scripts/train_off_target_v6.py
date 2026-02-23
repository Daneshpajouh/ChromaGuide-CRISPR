#!/usr/bin/env python3
"""
Off-target prediction v6 - Deep Neural Network with Residual Connections
Uses BCEWithLogitsLoss with pos_weight for extreme class imbalance.
10-model ensemble trained with different random seeds.
Target: AUROC >= 0.99 on CRISPRoffT dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}\n")


def one_hot_encode(seq, length=23):
    """One-hot encode DNA sequences."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((length, 4), dtype=np.float32)
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[j, mapping[c]] = 1
    return encoded.flatten()


def load_crispoff_data(data_path):
    """Load CRISPRoff/CRISPRoffT dataset from tab-separated TXT file."""
    seqs, labels = [], []
    print(f"Loading CRISPRoff from {data_path}...")

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

                if target_status not in ["ON", "OFF"]:
                    continue

                label = 1 if target_status == "OFF" else 0

                if not guide_seq or len(guide_seq) < 20:
                    continue

                seqs.append(guide_seq)
                labels.append(label)
            except (IndexError, ValueError):
                continue

    print(f"Loaded {len(seqs)} sequences")
    X = np.array([one_hot_encode(seq) for seq in seqs]).astype(np.float32)
    y = np.array(labels).astype(np.float32)

    print(f"ON: {(y==0).sum()}, OFF: {(y==1).sum()}")
    if (y==0).sum() > 0:
        print(f"Class ratio: {(y==1).sum()/(y==0).sum():.1f}:1\n")

    return X, y


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and dropout."""
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (in_features == out_features)

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.residual:
            out = out + x
        return out


class DeepOffTargetNet(nn.Module):
    """Deep neural network with residual connections for off-target prediction."""
    def __init__(self, input_dim=92, hidden_dims=[256, 512, 256, 128], dropout=0.3):
        super().__init__()

        # Input layer
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=dropout))

        # Output layer
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_fc(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_fc(x)
        return x


def train_epoch(model, loader, optimizer, device, criterion):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).view(-1, 1)

        logits = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    """Validate and compute AUROC."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy().flatten())

    if len(set(all_labels)) == 1:
        return 0.5
    return roc_auc_score(all_labels, all_preds)


def train_single_model(X_train, y_train, X_val, y_val, seed, device, epochs=500, batch_size=512):
    """Train a single off-target model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Weighted sampling for class imbalance
    n_on = (y_train == 0).sum()
    n_off = (y_train == 1).sum()
    if n_on > 0:
        class_weights = np.array([n_off / n_on, 1.0])
        sample_weights = np.array([class_weights[int(label)] for label in y_train])
    else:
        sample_weights = np.ones(len(y_train))

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Data loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = DeepOffTargetNet(input_dim=X_train.shape[1], hidden_dims=[256, 512, 256, 128], dropout=0.3).to(device)

    # Loss with pos_weight for imbalance
    pos_weight = torch.tensor([200.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-6)

    # Training
    best_auroc = 0
    best_state = None
    no_improve = 0
    patience = 50

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_auroc = validate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 50 == 0 or epoch < 3:
            print(f"    Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f} | LR: {lr:.2e}", flush=True)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auroc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/raw/crisprofft/crispoff_data.txt',
                        help='Path to off-target TXT file')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_models', type=int, default=10, help='Number of models in ensemble')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Off-Target Prediction v6 - Deep Neural Network Ensemble")
    print(f"{'='*70}")
    print(f"Architecture: 4 hidden layers (256→512→256→128) with residual blocks")
    print(f"Loss: BCEWithLogitsLoss (pos_weight=200.0)")
    print(f"Optimizer: AdamW (1e-3) with CosineAnnealingWarmRestarts")
    print(f"Ensemble: {args.n_models} models with random seeds")
    print(f"Dropout: 0.3, BatchNorm: enabled")
    print(f"{'='*70}\n", flush=True)

    # Load data
    X, y = load_crispoff_data(args.data_path)

    # Split train/val
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    n_on_train = (y_train == 0).sum()
    n_off_train = (y_train == 1).sum()
    n_on_val = (y_val == 0).sum()
    n_off_val = (y_val == 1).sum()

    print(f"Train: ON={int(n_on_train)}, OFF={int(n_off_train)}")
    print(f"Val:   ON={int(n_on_val)}, OFF={int(n_off_val)}\n")

    # Train ensemble
    print(f"Training {args.n_models}-model ensemble...\n", flush=True)
    models = []
    aurocs = []

    for seed in range(args.n_models):
        print(f"  Model {seed+1}/{args.n_models} (seed={seed}):", flush=True)
        model, best_auroc = train_single_model(
            X_train, y_train, X_val, y_val,
            seed=seed, device=device, epochs=args.epochs, batch_size=args.batch_size
        )
        models.append(model)
        aurocs.append(best_auroc)
        print(f"    ✓ Best AUROC = {best_auroc:.4f}\n", flush=True)

    print(f"Individual AUROCs: {[f'{a:.4f}' for a in aurocs]}")
    print(f"Mean AUROC: {np.mean(aurocs):.4f}")
    print(f"Std AUROC: {np.std(aurocs):.4f}\n")

    # Ensemble evaluation on validation set
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    val_loader = DataLoader(val_ds, batch_size=512)

    ensemble_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            batch_preds = []
            for model in models:
                model.eval()
                logits = model(X_batch)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                batch_preds.append(preds)
            ensemble_preds.extend(np.mean(batch_preds, axis=0))
            all_labels.extend(y_batch.numpy().flatten())

    ensemble_auroc = roc_auc_score(all_labels, ensemble_preds)
    precision, recall, _ = precision_recall_curve(all_labels, ensemble_preds)
    auprc = auc(recall, precision)

    print(f"{'='*70}")
    print(f"ENSEMBLE VALIDATION RESULTS (V6)")
    print(f"{'='*70}")
    print(f"Ensemble AUROC: {ensemble_auroc:.4f}")
    print(f"Ensemble AUPRC: {auprc:.4f}")
    print(f"{'='*70}\n", flush=True)

    # Target check
    if ensemble_auroc >= 0.99:
        print(f"✓ TARGET ACHIEVED: AUROC {ensemble_auroc:.4f} >= 0.99!!\n", flush=True)
    else:
        gap = 0.99 - ensemble_auroc
        print(f"⚠ Gap to target: {gap:.4f} ({(gap/0.99)*100:.1f}%)\n", flush=True)

    # Save models
    os.makedirs("models", exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f"models/off_target_v6_model_{i}.pt")

    print(f"✓ Saved {args.n_models} models: models/off_target_v6_model_X.pt")
    print(f"✓ Off-target v6 training complete\n")


if __name__ == "__main__":
    main()
