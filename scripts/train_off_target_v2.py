import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import argparse
import json

# Add src to path - use relative path for portability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from chromaguide.off_target import OffTargetScorer, CandidateFinder

class CRISPRoffTDataset(Dataset):
    def __init__(self, file_path, limit=None):
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        # Label 1 if Identity is 'ON' or validated OFF-target
        if 'Validation' in df.columns:
            df['label'] = df['Validation'].astype(float)
        else:
            # Fallback for different dataset versions
            df['label'] = ((df['Identity'] == 'ON') | (df['Identity'] == 'OFF')).astype(float)

        self.guides = df['Guide_sequence'].values
        self.targets = df['Target_sequence'].values
        self.labels = df['label'].values

        if limit:
            self.guides = self.guides[:limit]
            self.targets = self.targets[:limit]
            self.labels = self.labels[:limit]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        g, t = self.guides[idx], self.targets[idx]
        feat = CandidateFinder.encode_alignment(g, t)
        # Ensure no nans in features
        feat = torch.nan_to_num(feat)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        # Also ensure label is finite
        label = torch.nan_to_num(label, nan=0.0)
        return feat, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/raw/crisprofft/CRISPRoffT_all_targets.txt")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.data_path):
        print(f"Data not found at {args.data_path}. Please check path.")
        return

    dataset = CRISPRoffTDataset(args.data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = OffTargetScorer().to(device)

    # Use BCELoss since Scorer has Sigmoid
    # For highly imbalanced data, we can use a weight in the loss
    # but with BCELoss it's applied per-dataset or manually in the training loop.
    # We will use BCEWithLogitsLoss by removing Sigmoid from model temporarily OR
    # just stick with BCELoss and clamp.
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    results = {
        "epochs": [],
        "best_auroc": 0.0
    }

    print(f"Starting Off-target training on {len(train_ds)} samples...", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            preds = model(feats)
            # Clamp for stability
            preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                preds = model(feats)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)

        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": total_loss/len(train_loader),
            "auroc": auroc,
            "auprc": auprc
        }
        results["epochs"].append(epoch_metrics)

        print(f"Epoch {epoch+1} | Loss: {epoch_metrics['loss']:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}", flush=True)

        if auroc > results["best_auroc"]:
            results["best_auroc"] = auroc
            torch.save(model.state_dict(), "best_offtarget_model.pt")
            print(f"NEW BEST AUROC: {auroc:.4f}", flush=True)

    # Save final results
    with open("offtarget_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Training complete. Best AUROC: {results['best_auroc']:.4f}")

if __name__ == '__main__':
    main()
