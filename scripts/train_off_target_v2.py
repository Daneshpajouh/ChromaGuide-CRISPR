import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import argparse

# Add src to path
sys.path.insert(0, '/home/amird/chromaguide_experiments/src')
from chromaguide.off_target import OffTargetScorer, encode_alignment

class CRISPRoffTDataset(Dataset):
    def __init__(self, file_path, limit=None):
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        # Filter for validation classes or scores
        # Label 1 if Identity is 'ON' or validated OFF-target
        df['label'] = (df['Identity'] == 'ON') | (df['Identity'] == 'OFF')
        # Actually, in this dataset, 'OFF' means it IS an off-target cleavage site.
        # Let's check 'Validation' column
        df['label'] = df['Validation'].astype(int)
        
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
        feat = encode_alignment(g, t)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return feat, label

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/home/amird/chromaguide_experiments/data/raw/crisprofft/CRISPRoffT_all_targets.txt"

    dataset = CRISPRoffTDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = OffTargetScorer().to(device)
    
    # Class weights for imbalance (~50:1 as per methodology)
    pos_weight = torch.tensor([50.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Using Logits version for stability
    # Note: Scorer has Sigmoid at end. I'll remove it or adjust criterion.
    # Actually, Scorer has Sigmoid. Let's adjust to BCELoss.
    criterion = nn.BCELoss() 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting Off-target training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            preds = model(feats)
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
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
        
        if auroc >= 0.99:
            print("Target AUROC reached!")
            torch.save(model.state_dict(), "best_offtarget_model.pt")

if __name__ == '__main__':
    main()
