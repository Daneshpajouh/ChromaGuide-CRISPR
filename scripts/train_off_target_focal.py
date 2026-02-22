#!/usr/bin/env python3
"""
Off-target prediction model with FOCAL LOSS for extreme class imbalance.
Target: AUROC >= 0.99

This addresses the 99.54% OFF-target imbalance with focal loss (alpha=0.25, gamma=2.0)
which downweights easy samples and focuses on hard examples.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def one_hot_encode(seq, length=23):
    """One-hot encode DNA sequences"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, length))
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if j < length and c in mapping:
            encoded[mapping[c], j] = 1
    return encoded

def load_crisprofft_data(data_path, max_samples=None):
    """Load CRISPRoffT dataset with validation"""
    seqs, labels = [], []
    
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 35:  # Need at least 35 columns
                continue
            
            try:
                # Column 22 (index 21) = Guide_sequence
                # Column 34 (index 33) = Identity (ON/OFF)
                guide_seq = parts[21]
                target_status = parts[33]  # "ON" or "OFF"
                
                # Convert ON/OFF to binary: OFF=1 (off-target), ON=0 (on-target)
                if target_status not in ["ON", "OFF"]:
                    continue
                
                label = 1 if target_status == "OFF" else 0
                
                # Skip invalid sequences
                if not guide_seq or len(guide_seq) < 20:
                    continue
                
                seqs.append(guide_seq)
                labels.append(label)
                
                if max_samples and len(seqs) >= max_samples:
                    break
            except (IndexError, ValueError):
                continue
    
    print(f"Loaded {len(seqs)} raw sequences")
    
    # One-hot encode
    X = np.array([one_hot_encode(seq) for seq in seqs]).astype(np.float32)
    y = np.array(labels).astype(np.float32)
    
    # VALIDATION: Check encoding quality
    print(f"Encoded shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    print(f"Non-zero elements in batch: {np.sum(X[:10] > 0)}")
    print(f"Sample encoding (should be 4 x 23): {X[0].shape}")
    if X[0].sum() == 0:
        raise ValueError("ERROR: First sample encoding is all zeros!")
    
    # Check label distribution
    print(f"Label distribution - ON: {(y==0).sum()}, OFF: {(y==1).sum()}")
    
    return X, y


class OffTargetCNN(nn.Module):
    """Advanced CNN for off-target prediction - IMPROVED CAPACITY"""
    def __init__(self):
        super().__init__()
        # Increased capacity to handle complex patterns
        self.conv1 = nn.Conv1d(4, 256, kernel_size=8, padding=3)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=8, padding=3)
        self.conv3 = nn.Conv1d(512, 256, kernel_size=8, padding=3)
        self.conv4 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        # Larger FC layers for better capacity
        self.fc = nn.Sequential(
            nn.Linear(128 * 1, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn3(torch.relu(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn4(torch.relu(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance.
    
    Addresses class imbalance by down-weighting easy (correct) samples
    and focusing learning on hard (misclassified) samples.
    
    Formula: FL = -alpha * (1-pt)^gamma * log(pt)
    where:
    - pt: model predicted probability for true class
    - alpha: balancing factor (0.25 in our case)
    - gamma: focusing parameter (2.0 in our case)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output (batch_size, 1)
            targets: Binary labels (batch_size, 1)
        Returns:
            Scalar focal loss
        """
        # Convert logits to probabilities
        p = torch.sigmoid(logits)
        
        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # pt is the probability of target class
        pt = p * targets + (1 - p) * (1 - targets)
        
        # Apply focal weighting
        focal_weight = self.alpha * torch.pow(1.0 - pt, self.gamma)
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


def train_epoch(model, train_loader, optimizer, device, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_feats, batch_labels in train_loader:
        batch_feats = batch_feats.to(device)
        batch_labels = batch_labels.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        logits = model(batch_feats)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch_labels)
    
    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch_feats, batch_labels in val_loader:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_feats)
            # Convert logits to probabilities for AUROC calculation
            preds_prob = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds_prob)
            all_labels.extend(batch_labels.cpu().numpy().flatten())
    
    auroc = roc_auc_score(all_labels, all_preds)
    return auroc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/raw/crisprofft/CRISPRoffT_all_targets.txt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    X, y = load_crisprofft_data(args.data_path)
    
    print(f"Data shape: {X.shape}, Labels: {y.shape}")
    print(f"OFF-target samples: {int(y.sum())} / {len(y)}")
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Model and optimizer
    model = OffTargetCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Use Focal Loss for extreme class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    best_auroc = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_auroc = validate(model, val_loader, device)
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f} | LR: {lr:.6f}")
        
        scheduler.step(val_auroc)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), "best_offtarget_model_focal.pt")
        else:
            patience_counter += 1
            # Early stopping if no improvement for 15 epochs
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}: no improvement for 15 epochs")
                break
    
    print(f"Training finished. Best AUROC: {best_auroc:.4f}")
    print(f"Model saved to best_offtarget_model_focal.pt")


if __name__ == "__main__":
    main()
