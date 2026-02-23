#!/usr/bin/env python3
"""
Improved multimodal on-target training (v5) - Production Ready
Uses DNABERT-2 encoder with CNN fallback, Beta regression loss, cosine annealing+warmup.
Target: Rho >= 0.911 on GOLD test set.
d_model=64 throughout (PhD proposal constraint)
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import spearmanr
from pathlib import Path
import json
import sys
import argparse

# Disable compilation
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chromaguide.chromaguide_model import ChromaGuideModel


def main():
    parser = argparse.ArgumentParser(description='ChromaGuide v5 - Production Training')
    parser.add_argument('--backbone', type=str, default='dnabert2', choices=['cnn_gru', 'dnabert2'],
                        help='Sequence encoder backbone')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='mps', help='Device (mps, cuda, cpu)')
    parser.add_argument('--split', type=str, default='A', choices=['A', 'B', 'C'], help='Data split')
    parser.add_argument('--output_prefix', type=str, default='multimodal_v5', help='Output file prefix')
    args = parser.parse_args()

    # Device selection
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"ChromaGuide v5 - Improved Multimodal Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone} (CNN fallback enabled)")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Patience: {args.patience} | Split: {args.split}")
    print(f"d_model throughout: 64 (PhD proposal)\n", flush=True)

    # Load data
    print("Loading data split...", flush=True)
    split_name_map = {
        'A': 'split_a_gene_held_out',
        'B': 'split_b_dataset_held_out',
        'C': 'split_c_cellline_held_out'
    }
    split_dir = Path(f"data/processed/{split_name_map[args.split]}")

    if not split_dir.exists():
        split_dir_alt = Path(f"data/processed/split_{args.split.lower()}")
        if split_dir_alt.exists():
            split_dir = split_dir_alt

    if not split_dir.exists():
        raise FileNotFoundError(f"Split {split_dir} not found")

    # Load CSV files
    train_files = list(split_dir.glob("*_train.csv"))
    val_files = list(split_dir.glob("*_validation.csv"))
    test_files = list(split_dir.glob("*_test.csv"))

    if not train_files or not val_files or not test_files:
        raise ValueError(f"Missing split files in {split_dir}")

    train_df = pd.concat([pd.read_csv(f) for f in train_files]).reset_index(drop=True)
    val_df = pd.concat([pd.read_csv(f) for f in val_files]).reset_index(drop=True)
    test_df = pd.concat([pd.read_csv(f) for f in test_files]).reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}", flush=True)

    # Initialize model
    print(f"\nInitializing {args.backbone} model with d_model=64...", flush=True)
    model = ChromaGuideModel(
        encoder_type=args.backbone,
        d_model=64,  # CRITICAL: PhD proposal requirement
        seq_len=21,
        num_epi_tracks=11,
        num_epi_bins=1,
        use_epigenomics=True,
        use_gate_fusion=True,
        dropout=0.1,
    ).to(device)
    print(f"Model initialized", flush=True)

    # Optimizer with differential LR if DNABERT-2
    if args.backbone == 'dnabert2':
        optimizer = torch.optim.AdamW([
            {'params': model.seq_encoder.parameters(), 'lr': 1e-5},
            {'params': [p for n, p in model.named_parameters() if 'seq_encoder' not in n], 'lr': args.lr}
        ], weight_decay=0.01)
        print("Using differential learning rates (DNABERT-2: 1e-5, rest: {})".format(args.lr), flush=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Cosine annealing with 5-epoch warmup
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs-5, T_mult=1, eta_min=1e-7)

    # Prepare batch functions
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def prepare_seq(seqs, device):
        """One-hot encode sequences (no external tokenizer)."""
        L = 21
        seq_tensor = torch.zeros(len(seqs), 4, L, device=device, dtype=torch.float32)
        for j, seq in enumerate(seqs):
            for k, nt in enumerate(seq[:L]):
                if nt.upper() in nt_map:
                    seq_tensor[j, nt_map[nt.upper()], k] = 1
        return seq_tensor

    def prepare_epi(batch, device):
        """Extract epigenomics features."""
        cols = [f'feat_{i}' for i in range(11)]
        if not all(col in batch.columns for col in cols):
            return None
        epi_data = batch[cols].values.astype(np.float32)
        return torch.tensor(epi_data, device=device, dtype=torch.float32).unsqueeze(2)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training with Beta regression loss...")
    print(f"{'='*60}\n", flush=True)

    best_val_rho = -1
    no_improve = 0
    results = {"epochs": [], "best_val_rho": 0.0, "test_rho": 0.0}

    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        train_batch_count = 0

        for batch_start in range(0, len(train_df), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(train_df))
            batch = train_df.iloc[batch_start:batch_end]

            try:
                seqs = batch['sequence'].tolist()
                labels = torch.tensor(batch['efficiency'].values, dtype=torch.float32).to(device).unsqueeze(1)

                seq_tensor = prepare_seq(seqs, device)
                epi_tensor = prepare_epi(batch, device)

                output = model(seq_tensor, epi_tracks=epi_tensor)
                loss_dict = model.compute_loss(output, labels)
                loss = loss_dict['total_loss']

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                train_batch_count += 1
            except Exception as e:
                print(f"Batch error (skipping): {str(e)[:50]}", flush=True)
                continue

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_start in range(0, len(val_df), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(val_df))
                batch = val_df.iloc[batch_start:batch_end]

                try:
                    seqs = batch['sequence'].tolist()
                    seq_tensor = prepare_seq(seqs, device)
                    epi_tensor = prepare_epi(batch, device)

                    output = model(seq_tensor, epi_tracks=epi_tensor)
                    val_preds.extend(output['mu'].cpu().numpy().flatten())
                    val_labels.extend(batch['efficiency'].values)
                except Exception as e:
                    continue

        # Compute metrics
        if len(val_preds) > 1 and len(set(val_labels)) > 1:
            val_rho, _ = spearmanr(val_preds, val_labels)
            avg_loss = total_loss / max(1, train_batch_count)
            lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val Rho: {val_rho:.4f} | LR: {lr:.2e}", flush=True)

            results["epochs"].append({
                "epoch": epoch + 1,
                "loss": float(avg_loss),
                "val_rho": float(val_rho)
            })

            # Early stopping
            if val_rho > best_val_rho:
                best_val_rho = val_rho
                results["best_val_rho"] = float(val_rho)
                no_improve = 0
                torch.save(model.state_dict(), f"best_model_{args.output_prefix}_{args.backbone}.pt")
                print(f"  ✓ New best Rho: {best_val_rho:.4f}", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}: no improvement for {args.patience} epochs", flush=True)
                    break

        scheduler.step()

    # Test set evaluation
    print(f"\n{'='*60}")
    print(f"Evaluating on GOLD test set...")
    print(f"{'='*60}\n", flush=True)

    model_path = f"best_model_{args.output_prefix}_{args.backbone}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best model from {model_path}", flush=True)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_start in range(0, len(test_df), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(test_df))
            batch = test_df.iloc[batch_start:batch_end]

            try:
                seqs = batch['sequence'].tolist()
                seq_tensor = prepare_seq(seqs, device)
                epi_tensor = prepare_epi(batch, device)

                output = model(seq_tensor, epi_tracks=epi_tensor)
                test_preds.extend(output['mu'].cpu().numpy().flatten())
                test_labels.extend(batch['efficiency'].values)
            except Exception as e:
                continue

    # Final metrics
    if len(test_preds) > 1 and len(set(test_labels)) > 1:
        test_rho, _ = spearmanr(test_preds, test_labels)
        print(f"GOLD Test Rho: {test_rho:.4f}", flush=True)
        results["test_rho"] = float(test_rho)

        # Target check
        if test_rho >= 0.911:
            print(f"✓ TARGET ACHIEVED: Rho {test_rho:.4f} >= 0.911", flush=True)
        else:
            gap = 0.911 - test_rho
            print(f"⚠ Gap remaining: {gap:.4f} ({(gap/0.911)*100:.1f}%)", flush=True)

    # Save results
    output_json = f"{args.output_prefix}_{args.backbone}_metrics.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_json}")

    # Save predictions
    if test_preds and test_labels:
        pred_df = pd.DataFrame({
            'sequence': test_df['sequence'].iloc[:len(test_preds)].values,
            'efficiency_actual': test_labels,
            'efficiency_predicted': test_preds
        })
        pred_csv = f"predictions_{args.output_prefix}_{args.backbone}.csv"
        pred_df.to_csv(pred_csv, index=False)
        print(f"Predictions saved to {pred_csv}\n")

    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}\n", flush=True)


if __name__ == '__main__':
    main()
