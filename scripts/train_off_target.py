"""Train Off-Target Scorer for ChromaGuide.

Uses classification logic on aligned guide-target pairs.
Dataset: CRISPRoffT SOTA Benchmark.
Target: AUROC > 0.99
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

# Fix: Use correct package import for Narval environment
try:
    from chromaguide.off_target import OffTargetScorer, CandidateFinder
except ImportError:
    try:
        from src.chromaguide.off_target import OffTargetScorer, CandidateFinder
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.join(os.getcwd(), 'src'))
        from chromaguide.off_target import OffTargetScorer, CandidateFinder


class OffTargetDataset(Dataset):
    """Simple loader for Guide-Target pairs."""
    def __init__(self, data_path: str, max_samples: int = None):
        print(f"Loading off-target data from {data_path}...")
        df = pd.read_csv(data_path, sep="\t", on_bad_lines='skip')

        # Methodology: Use high-quality GUIDE-seq and validated sites
        # Binary label: ON/OFF (ON = 1, OFF = 0)
        # We also need to filter for SpCas9 (NGG) if possible
        if 'Cas9_type' in df.columns:
            df = df[df['Cas9_type'].str.contains('SpCas9', na=True, case=False)]

        self.guides = df['Guide_sequence'].values
        self.targets = df['Target_sequence'].values

        # Label: If 'Validation' is TRUE or Score > threshold
        if 'Validation' in df.columns:
            self.labels = (df['Identity'] == 'ON').astype(float).values
        else:
            self.labels = np.zeros(len(df))

        if max_samples:
            self.guides = self.guides[:max_samples]
            self.targets = self.targets[:max_samples]
            self.labels = self.labels[:max_samples]

        print(f"Loaded {len(self.labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        guide_seq = self.guides[idx]
        target_seq = self.targets[idx]
        label = self.labels[idx]

        # Encode alignment
        features = CandidateFinder.encode_alignment(guide_seq, target_seq)
        return features, torch.tensor([label], dtype=torch.float32)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Path on Narval
    data_path = "/home/amird/chromaguide_experiments/data/raw/crisprofft/CRISPRoffT_all_targets.txt"
    if not Path(data_path).exists():
        print(f"Data not found at {data_path}. Checking local relative path...")
        data_path = "data/raw/crisprofft/CRISPRoffT_all_targets.txt"
        if not Path(data_path).exists():
            print("Data not found. Skipping training.")
            return

    # Use a bigger max_samples if running on Narval
    max_samples = 250000
    dataset = OffTargetDataset(data_path, max_samples=max_samples)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=4)

    model = OffTargetScorer(in_channels=10).to(device)

    # Class weight 20:1 as per methodology (ratio of OFF to ON in typical screen)
    pos_weight = torch.tensor([5.0]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_auc = 0
    save_dir = Path("/home/amird/chromaguide_experiments/models/off_target")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Off-target training on {len(train_ds)} samples...")
    for epoch in range(15):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = nn.BCELoss()(out, y) # Scorer has Sigmoid
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                all_preds.extend(out.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        try:
            auc = roc_auc_score(all_labels, all_preds)
            pr_auc = average_precision_score(all_labels, all_preds)
        except:
            auc = 0.5
            pr_auc = 0

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val ROC-AUC: {auc:.4f} | Val PR-AUC: {pr_auc:.4f}")

        scheduler.step(auc)

        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), save_dir / "off_target_cnn_best.pt")
            print(f"New best model saved with AUC: {auc:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train()
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds = model(x).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        val_pr = average_precision_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | AUC: {val_auc:.4f} | PR-AUC: {val_pr:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "models/off_target_best.pt")

if __name__ == "__main__":
    train()
