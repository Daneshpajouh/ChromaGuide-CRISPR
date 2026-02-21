import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from pathlib import Path
import json
import logging
import sys
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from chromaguide.chromaguide_model import ChromaGuideModel

# Disable Torch compilation if it causes issues on cluster
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def main():
    parser = argparse.ArgumentParser(description="ChromaGuide Production Training V5 (Multimodal)")
    parser.add_argument("--backbone", type=str, default="dnabert2", choices=["cnn_gru", "mamba", "dnabert2"])
    parser.add_argument("--fusion", type=str, default="gate", choices=["gate", "concat"])
    parser.add_argument("--use_epi", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--input", type=str, default="/home/amird/chromaguide_experiments/data/real/processed/enriched_multimodal.h5")
    parser.add_argument("--gold", type=str, default="/home/amird/chromaguide_experiments/test_set_GOLD.csv")
    parser.add_argument("--output_name", type=str, default="results/chromaguide_gold/v5_metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load data from HDF5
    print(f"Loading multimodal data from: {args.input}")
    with h5py.File(args.input, 'r') as f:
        all_sequences = np.array(f['sequences']).astype(str)
        all_efficiencies = np.array(f['efficiencies'])
        all_epigenomics = np.array(f['epigenomics'])

    # Load GOLD test set
    test_df = pd.read_csv(args.gold)
    gold_sequences = set(test_df["sequence"].tolist())

    # Split into train/val/test
    # We remove GOLD sequences from training
    train_indices = []
    val_indices = []
    gold_indices = []

    for i, seq in enumerate(all_sequences):
        if seq in gold_sequences:
            gold_indices.append(i)
        else:
            # Simple 90/10 split for non-gold
            if hash(seq) % 10 < 9:
                train_indices.append(i)
            else:
                val_indices.append(i)

    print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | GOLD Matches: {len(gold_indices)}")

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
        use_epigenomics=False if not args.use_epi else True,
        use_gate_fusion=(args.fusion == "gate"),
        dropout=0.1,
    )

    model.to(device)

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

    class MultimodalDataset(Dataset):
        def __init__(self, sequences, efficiencies, epigenomics, indices, tokenizer=None):
            self.seqs = sequences[indices]
            self.labels = efficiencies[indices]
            self.epi = epigenomics[indices]
            self.tokenizer = tokenizer

        def __len__(self): return len(self.labels)

        def __getitem__(self, idx):
            s = str(self.seqs[idx])
            t = torch.tensor([self.labels[idx]], dtype=torch.float32)
            e = torch.tensor(self.epi[idx], dtype=torch.float32)

            if self.tokenizer:
                encoded = self.tokenizer(s, padding="max_length", max_length=23, truncation=True, return_tensors="pt")
                return {"seq": encoded["input_ids"].squeeze(0), "epi": e, "label": t}
            else:
                mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1], "N": [0.25,0.25,0.25,0.25]}
                enc = [mapping.get(b.upper(), mapping["N"]) for b in s[:23]]
                while len(enc) < 23: enc.append(mapping["N"])
                return {"seq": torch.tensor(enc, dtype=torch.float32).transpose(0, 1), "epi": e, "label": t}

    train_loader = DataLoader(MultimodalDataset(all_sequences, all_efficiencies, all_epigenomics, train_indices, tokenizer), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MultimodalDataset(all_sequences, all_efficiencies, all_epigenomics, val_indices, tokenizer), batch_size=args.batch_size, shuffle=False)
    gold_loader = DataLoader(MultimodalDataset(all_sequences, all_efficiencies, all_epigenomics, gold_indices, tokenizer), batch_size=args.batch_size, shuffle=False)

    results = {"epochs": [], "best_val_rho": 0, "gold_rho": 0}
    print("Starting production training V5...", flush=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
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
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                seq, epi, label = batch["seq"].to(device), batch["epi"].to(device), batch["label"].to(device)
                output = model(seq, epi)
                preds.extend(output.cpu().numpy().flatten())
                targets.extend(label.cpu().numpy().flatten())

        rho, _ = spearmanr(preds, targets)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Rho={rho:.4f}")

        results["epochs"].append({"epoch": epoch+1, "val_rho": float(rho)})

        if rho > results["best_val_rho"]:
            results["best_val_rho"] = float(rho)
            torch.save(model.state_dict(), "best_model_v5.pt")

    # Final evaluation on GOLD
    model.load_state_dict(torch.load("best_model_v5.pt"))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in gold_loader:
            seq, epi, label = batch["seq"].to(device), batch["epi"].to(device), batch["label"].to(device)
            output = model(seq, epi)
            preds.extend(output.cpu().numpy().flatten())
            targets.extend(label.cpu().numpy().flatten())

    gold_rho, _ = spearmanr(preds, targets)
    results["gold_rho"] = float(gold_rho)
    print(f"Final GOLD Rho: {gold_rho:.4f}")

    os.makedirs(os.path.dirname(args.output_name), exist_ok=True)
    with open(args.output_name, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
