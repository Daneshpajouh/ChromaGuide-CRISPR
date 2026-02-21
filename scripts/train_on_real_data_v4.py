import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from pathlib import Path
import json
import logging
import sys
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from chromaguide.chromaguide_model import ChromaGuideModel

# Disable Torch compilation if it causes issues on cluster
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import h5py

def main():
    parser = argparse.ArgumentParser(description="ChromaGuide Production Training")
    parser.add_argument("--backbone", type=str, default="dnabert2", choices=["cnn_gru", "mamba", "dnabert2"])
    parser.add_argument("--fusion", type=str, default="gate", choices=["gate", "concat"])
    parser.add_argument("--use_epi", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--output_name", type=str, default="results/chromaguide_gold/v5_metrics.json")
    parser.add_argument("--data_h5", type=str, default="data/real/processed/multimodal_data.h5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # UPDATED: Use HDF5 data instead of CSV
    h5_path = args.data_h5
    gold_csv_path = "test_set_GOLD.csv"

    if not os.path.exists(h5_path):
        # Fallback to absolute path on Narval for my user
        h5_path = "/home/amird/chromaguide_experiments/data/real/processed/multimodal_data.h5"
    if not os.path.exists(gold_csv_path):
        gold_csv_path = "/home/amird/chromaguide_experiments/test_set_GOLD.csv"

    print(f"Loading multimodal data from: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        all_sequences = f['sequences'][:].astype(str)
        all_efficiencies = f['efficiencies'][:]
        all_epigenomics = f['epigenomics'][:]

    print(f"Loading GOLD test set from: {gold_csv_path}")
    test_df = pd.read_csv(gold_csv_path)
    gold_seq_set = set(test_df["sequence"].tolist())

    # Create masks for Train/Val vs GOLD
    is_gold = np.array([seq in gold_seq_set for seq in all_sequences])

    # Split non-gold into train and val
    non_gold_indices = np.where(~is_gold)[0]
    np.random.seed(42)
    np.random.shuffle(non_gold_indices)

    train_split = int(0.9 * len(non_gold_indices))
    train_idx = non_gold_indices[:train_split]
    val_idx = non_gold_indices[train_split:]
    gold_idx = np.where(is_gold)[0]

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | GOLD: {len(gold_idx)}")

    tokenizer = None
    if args.backbone == "dnabert2":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    model = ChromaGuideModel(
        encoder_type=args.backbone,
        d_model=768 if args.backbone == "dnabert2" else 256,
        seq_len=23,
        num_epi_tracks=3,
        num_epi_bins=100,
        use_epigenomics=args.use_epi,
        use_gate_fusion=(args.fusion == "gate"),
        dropout=0.1,
    )

    model.to(device)

    # CRITICAL: DNABERT-2 Alibi tensors often hide in sub-loops and don't move with .to(device)
    if args.backbone == "dnabert2":
        for name, buffer in model.seq_encoder.backbone.named_buffers():
            if "alibi" in name:
                model.seq_encoder.backbone.register_buffer(name, buffer.to(device), persistent=False)

    # High-performance optimizer setup
    if args.backbone == "dnabert2":
        optimizer = torch.optim.AdamW([
            {'params': model.seq_encoder.parameters(), 'lr': args.lr},
            {'params': [p for n, p in model.named_parameters() if 'seq_encoder' not in n], 'lr': args.lr * 10}
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    class CRISPRDataset(Dataset):
        def __init__(self, sequences, efficiencies, epigenomics, indices, tokenizer=None, seq_len=23):
            self.seqs = sequences[indices]
            self.labels = efficiencies[indices]
            self.epi = epigenomics[indices]
            self.tokenizer = tokenizer
            self.seq_len = seq_len
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            s = str(self.seqs[idx])
            t = torch.tensor([self.labels[idx]], dtype=torch.float32)
            e = torch.tensor(self.epi[idx], dtype=torch.float32)

            if self.tokenizer:
                encoded = self.tokenizer(s, padding="max_length", max_length=self.seq_len, truncation=True, return_tensors="pt")
                return {"seq": encoded["input_ids"].squeeze(0), "epi": e, "label": t}
            else:
                mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1], "N": [0.25,0.25,0.25,0.25]}
                enc = [mapping.get(b.upper(), mapping["N"]) for b in s[:23]]
                while len(enc) < 23: enc.append(mapping["N"])
                return {"seq": torch.tensor(enc, dtype=torch.float32).transpose(0, 1), "epi": e, "label": t}

    train_loader = DataLoader(CRISPRDataset(all_sequences, all_efficiencies, all_epigenomics, train_idx, tokenizer),
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(CRISPRDataset(all_sequences, all_efficiencies, all_epigenomics, val_idx, tokenizer),
                            batch_size=args.batch_size, shuffle=False)
    gold_loader = DataLoader(CRISPRDataset(all_sequences, all_efficiencies, all_epigenomics, gold_idx, tokenizer),
                             batch_size=args.batch_size, shuffle=False)

    results = {"epochs": [], "best_val_rho": 0, "gold_rho": 0}
    print("Starting production training...", flush=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            seq, epi, label = batch["seq"].to(device), batch["epi"].to(device), batch["label"].to(device)

            output = model(seq, epi)
            loss_dict = model.compute_loss(output, label)
            loss = loss_dict['total_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        p_val, y_val = [], []
        with torch.no_grad():
            for batch in val_loader:
                seq, epi, label = batch["seq"].to(device), batch["epi"].to(device), batch["label"].to(device)
                p = model(seq, epi)["mu"]
                p_val.extend(p.cpu().numpy().flatten())
                y_val.extend(label.cpu().numpy().flatten())

        val_rho, _ = spearmanr(y_val, p_val)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Rho: {val_rho:.4f}", flush=True)

        if val_rho > results["best_val_rho"]:
            results["best_val_rho"] = float(val_rho)
            torch.save(model.state_dict(), "best_model.pt")

            # Eval on Gold set
            p_gold, y_gold = [], []
            for batch in gold_loader:
                seq, epi, label = batch["seq"].to(device), batch["epi"].to(device), batch["label"].to(device)
                p = model(seq, epi)["mu"]
                p_gold.extend(p.cpu().numpy().flatten())
                y_gold.extend(label.cpu().numpy().flatten())
            gold_rho, _ = spearmanr(y_gold, p_gold)
            results["gold_rho"] = float(gold_rho)
            print(f"  --> NEW BEST! GOLD Rho: {gold_rho:.4f}", flush=True)

        results["epochs"].append({"epoch": epoch+1, "val_rho": val_rho})

    os.makedirs(os.path.dirname(args.output_name), exist_ok=True)
    with open(args.output_name, "w") as f:
        json.dump(results, f, indent=4)
    print(f"DONE. Final GOLD Rho: {results['gold_rho']:.4f}")

if __name__ == "__main__":
    main()
