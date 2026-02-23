#!/usr/bin/env python3
"""
Off-target prediction v5 - Production Ready
Uses Asymmetric Loss (ASL) for extreme class imbalance, attention pooling, deeper CNN.
Target: AUROC >= 0.99 on CRISPRoffT dataset.
Ensemble of 5 models trained with different random seeds.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
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
    encoded = np.zeros((4, length), dtype=np.float32)
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[mapping[c], j] = 1
    return encoded


def load_crisprofft_data(data_path):
    """Load CRISPRoffT dataset."""
    seqs, labels = [], []
    print(f"Loading CRISPRoffT from {data_path}...")

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
    print(f"Class ratio: {(y==1).sum()/(y==0).sum():.1f}:1\n")

    return X, y


class AttentionPool(nn.Module):
    """Attention-based pooling instead of max/avg pool."""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """x: (batch, channels, seq_len)"""
        weights = self.attention(x)  # (batch, 1, seq_len)
        weighted = x * weights  # (batch, channels, seq_len)
        return weighted.sum(dim=2) / (weights.sum(dim=2) + 1e-8)  # (batch, channels)


class DeepOffTargetCNN(nn.Module):
    """Deeper CNN with attention pooling for off-target prediction."""
    def __init__(self):
        super().__init__()

        # Input: (batch, 4, 23)
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Attention pooling
        self.attention_pool = AttentionPool(512)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Attention pooling
        x = self.attention_pool(x)  # (batch, 512)
        x = self.fc(x)  # (batch, 1)
        return x


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for extreme class imbalance.

    ASL(p) = (1-p)^gamma+ * log(p) for positives
            p^gamma- * log(1-p) for negatives

    gamma+ = 0 (don't focus on easy positives)
    gamma- = 4 (focus on hard negatives)
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, labels):
        """logits: (batch, 1), labels: (batch, 1)"""
        p = torch.sigmoid(logits)

        # BCE for numerical stability
        p = torch.clamp(p, self.clip, 1 - self.clip)

        # Positive class (label=1)
        pos_loss = -(labels) * torch.log(p) * torch.pow(1 - p, self.gamma_pos)

        # Negative class (label=0)
        neg_loss = -(1 - labels) * torch.log(1 - p) * torch.pow(p, self.gamma_neg)

        return (pos_loss + neg_loss).mean()


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


def train_single_model(X_train, y_train, X_val, y_val, seed, device, epochs=200, batch_size=512):
    """Train a single off-target model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Weighted sampling
    class_weights = np.array([
        (y_train == 1).sum() / (y_train == 0).sum(),  # weight for ON (minority)
        1.0  # weight for OFF (majority)
    ])
    sample_weights = np.array([class_weights[int(label)] for label in y_train])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Data loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = DeepOffTargetCNN().to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0)

    # Training
    best_auroc = 0
    best_state = None
    no_improve = 0
    patience = 30

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_auroc = validate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f} | LR: {lr:.2e}", flush=True)

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
    parser.add_argument('--data_path', type=str, default='data/raw/crisprofft/CRISPRoffT_all_targets.txt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_models', type=int, default=5, help='Number of models in ensemble')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Off-Target Prediction v5 - Ensemble Training")
    print(f"{'='*70}")
    print(f"Loss: Asymmetric Loss (gamma_neg=4, gamma_pos=0)")
    print(f"Optimizer: AdamW (1e-3) with CosineAnnealingLR")
    print(f"Architecture: Deep CNN with attention pooling")
    print(f"Ensemble: {args.n_models} models with random seeds")
    print(f"{'='*70}\n", flush=True)

    # Load data
    X, y = load_crisprofft_data(args.data_path)

    # Split train/val
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: ON={int((y_train==0).sum())}, OFF={int((y_train==1).sum())}")
    print(f"Val:   ON={int((y_val==0).sum())}, OFF={int((y_val==1).sum())}\n")

    # Train ensemble
    print(f"Training {args.n_models}-model ensemble...\n", flush=True)
    models = []
    aurocs = []

    for seed in range(args.n_models):
        print(f"Model {seed+1}/{args.n_models} (seed={seed}):", flush=True)
        model, best_auroc = train_single_model(
            X_train, y_train, X_val, y_val,
            seed=seed, device=device, epochs=args.epochs, batch_size=args.batch_size
        )
        models.append(model)
        aurocs.append(best_auroc)
        print(f"  Completed: Best AUROC = {best_auroc:.4f}\n", flush=True)

    print(f"Ensemble AUROCs: {[f'{a:.4f}' for a in aurocs]}")
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
    print(f"ENSEMBLE VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Ensemble AUROC: {ensemble_auroc:.4f}")
    print(f"Ensemble AUPRC: {auprc:.4f}")
    print(f"{'='*70}\n", flush=True)

    # Target check
    if ensemble_auroc >= 0.99:
        print(f"✓ TARGET ACHIEVED: AUROC {ensemble_auroc:.4f} >= 0.99!!\n", flush=True)
    else:
        gap = 0.99 - ensemble_auroc
        print(f"⚠ Gap: {gap:.4f} ({(gap/0.99)*100:.1f}%)\n", flush=True)

    # Save models
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f"off_target_ensemble_model_{i}.pt")

    print(f"✓ Saved {args.n_models} models: off_target_ensemble_model_X.pt")
    print(f"✓ Training complete\n")


if __name__ == "__main__":
    main()
