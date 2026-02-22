#!/usr/bin/env python3
"""
Off-target prediction model using CRISPRoffT dataset.
Target: AUROC >= 0.99
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

def train_epoch(model, train_loader, optimizer, device, pos_weight=None):
    """Train for one epoch - USE WEIGHTED LOSS FOR CLASS IMBALANCE"""
    model.train()
    total_loss = 0
    # Use BCEWithLogitsLoss with pos_weight to handle class imbalance
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    for batch_feats, batch_labels in train_loader:
        batch_feats = batch_feats.to(device)
        batch_labels = batch_labels.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        preds = model(batch_feats)
        loss = criterion(preds, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_labels)
    
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    all_preds, all_labels = [], []
    criterion = nn.BCEWithLogitsLoss()
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
            loss = criterion(logits, batch_labels.view(-1, 1))
            total_loss += loss.item() * len(batch_labels)
    
    auroc = roc_auc_score(all_labels, all_preds)
    avg_loss = total_loss / len(val_loader.dataset)
    return auroc, avg_loss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/raw/crisprofft/CRISPRoffT_all_targets.txt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    X, y = load_crisprofft_data(args.data_path)
    
    print(f"Data shape: {X.shape}, Labels: {y.shape}")
    print(f"OFF-target samples: {int(y.sum())} / {len(y)}")
    
    # CALCULATE CLASS WEIGHT FOR IMBALANCE HANDLING
    n_off = int(y.sum())
    n_on = len(y) - n_off
    class_ratio = n_on / n_off if n_off > 0 else 1.0
    pos_weight = torch.tensor([class_ratio], device=device, dtype=torch.float32)
    print(f"Class ratio (ON/OFF): {class_ratio:.2f}, pos_weight: {pos_weight.item():.2f}")
    
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_auroc = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight)
        val_auroc, val_loss = validate(model, val_loader, device)
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f} | LR: {lr:.6f}")
        
        scheduler.step(val_auroc)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), "best_offtarget_model_v4.pt")
        else:
            patience_counter += 1
            # Early stopping if no improvement for 10 epochs
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}: no improvement for 10 epochs")
                break
    
    print(f"Training finished. Best AUROC: {best_auroc:.4f}")
    print(f"Model saved to best_offtarget_model_v4.pt")

if __name__ == "__main__":
    main()
