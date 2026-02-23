#!/usr/bin/env python3
"""
Improved multimodal on-target prediction with fallback backbone support.
Target: Rho >= 0.911

This script supports multiple backbones:
1. DNABERT2 (preferred if transformers works)
2. CNN-GRU (fallback if transformers has import issues)
3. CNN_GRU (base ChromaGuide model)

The use of d_model=64 per PhD Proposal constraint.
"""

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

# Add src to path for ChromaGuide imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chromaguide.chromaguide_model import ChromaGuideModel

# Disable Torch compilation if it causes issues on cluster
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Try to import transformers, but don't fail if PIL has issues
tokenizer = None
tokenizer_available = False
try:
    from transformers import AutoTokenizer
    # Test tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    tokenizer_available = True
    print("✓ DNABERT-2 tokenizer loaded successfully")
except Exception as e:
    print(f"⚠ Could not load DNABERT-2 tokenizer (this is okay, using fallback): {str(e)[:100]}")
    tokenizer_available = False


def main():
    parser = argparse.ArgumentParser(description='ChromaGuide Production Training - Improved')
    parser.add_argument('--backbone', type=str, default='cnn_gru', choices=['cnn_gru', 'mamba', 'dnabert2'],
                        help='Backbone types (dnabert2 requires transformers, falls back to cnn_gru if unavailable)')
    parser.add_argument('--fusion', type=str, default='gate', choices=['gate', 'concat'], help='Fusion type')
    parser.add_argument('--use_epi', action='store_true', default=True, help='Use epigenomics')
    parser.add_argument('--no_epi', action='store_false', dest='use_epi', help='Disable epigenomics')
    parser.add_argument('--learning_rate', '--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='mps', help='Device for training (mps, cuda, cpu)')
    parser.add_argument('--split', type=str, default='A', choices=['A', 'B', 'C'], help='Split type')
    parser.add_argument('--output_name', type=str, default='on_target_metrics.json', help='Output JSON file')
    args = parser.parse_args()

    # Determine device
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            print(f"Warning: MPS not available. Using CPU instead.")
            device = torch.device('cpu')
        else:
            device = torch.device('mps')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print(f"Warning: CUDA not available. Using CPU instead.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}", flush=True)

    # Handle backbone selection
    backbone = args.backbone
    if args.backbone == 'dnabert2' and not tokenizer_available:
        print(f"⚠ DNABERT-2 requested but tokenizer not available. Using CNN-GRU fallback.", flush=True)
        backbone = 'cnn_gru'

    print(f"Config: Backbone={backbone}, Fusion={args.fusion}, UseEpi={args.use_epi}, Split={args.split}", flush=True)

    # Load data from splits
    split_name_map = {
        'A': 'split_a_gene_held_out',
        'B': 'split_b_dataset_held_out',
        'C': 'split_c_cellline_held_out'
    }
    split_dir = Path(f"data/processed/{split_name_map[args.split.upper()]}")
    if not split_dir.exists():
        # Try symlink variant
        split_dir_alt = Path(f"data/processed/split_{args.split.lower()}")
        if split_dir_alt.exists():
            split_dir = split_dir_alt
            print(f"Using alternative split directory: {split_dir}", flush=True)
        else:
            raise FileNotFoundError(f"Split directory {split_dir} not found")

    # Combine all cell lines for training/val/test
    train_csv_files = list(split_dir.glob("*_train.csv"))
    val_csv_files = list(split_dir.glob("*_validation.csv"))
    test_csv_files = list(split_dir.glob("*_test.csv"))

    if not train_csv_files or not val_csv_files or not test_csv_files:
        print(f"Warning: Some CSV files missing. Found:")
        print(f"  Train: {len(train_csv_files)} files")
        print(f"  Val: {len(val_csv_files)} files")
        print(f"  Test: {len(test_csv_files)} files")
        if not train_csv_files or not val_csv_files or not test_csv_files:
            raise ValueError("Missing split files")

    train_dfs = [pd.read_csv(f) for f in train_csv_files]
    val_dfs = [pd.read_csv(f) for f in val_csv_files]
    test_dfs = [pd.read_csv(f) for f in test_csv_files]

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    print(f"Data loaded: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}", flush=True)

    # Initialize Model
    print(f"Initializing model with {backbone} backbone...", flush=True)
    model = ChromaGuideModel(
        encoder_type=backbone,
        d_model=64,  # PhD Proposal constraint
        seq_len=21,
        num_epi_tracks=11 if args.use_epi else 0,
        num_epi_bins=1,
        use_epigenomics=args.use_epi,
        use_gate_fusion=(args.fusion == 'gate'),
        dropout=0.1,
    ).to(device)

    print(f"Model initialized successfully", flush=True)

    # Optimization
    if backbone == 'dnabert2' and tokenizer is not None:
        # Low LR for transformer backbone
        optimizer = torch.optim.AdamW([
            {'params': model.seq_encoder.parameters(), 'lr': 1e-5},
            {'params': [p for n, p in model.named_parameters() if 'seq_encoder' not in n], 'lr': 1e-4}
        ], weight_decay=0.01)
        print("Using differential learning rates for DNABERT-2", flush=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        print(f"Using Adam optimizer with LR {args.learning_rate}", flush=True)

    # Prepare batch functions
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def prepare_batch_seq(seqs, backbone, tokenizer, device):
        if backbone == 'dnabert2' and tokenizer is not None:
            try:
                return tokenizer(seqs, return_tensors='pt', padding=True, truncation=True).to(device)['input_ids']
            except Exception as e:
                print(f"Tokenizer error: {e}, falling back to one-hot encoding")
                backbone = 'cnn_gru'

        # One-hot encoding for CNN-GRU or fallback
        L = len(seqs[0]) if seqs else 21
        seq_tensor = torch.zeros(len(seqs), 4, L, device=device, dtype=torch.float32)
        for j, seq in enumerate(seqs):
            for k, nt in enumerate(seq[:L]):
                if nt.upper() in nt_map:
                    seq_tensor[j, nt_map[nt.upper()], k] = 1
        return seq_tensor

    def prepare_batch_epi(batch, use_epi, device):
        if not use_epi:
            return None
        # Extract feat_0 to feat_10
        cols = [f'feat_{i}' for i in range(11)]
        if not all(col in batch.columns for col in cols):
            print(f"Warning: Epi features not fully available. Available columns: {batch.columns.tolist()}")
            return None
        epi_data = batch[cols].values.astype(np.float32)
        # Shape (B, 11) -> (B, 11, 1)
        return torch.tensor(epi_data, device=device, dtype=torch.float32).unsqueeze(2)

    # Training
    print(f"\n=== Starting Training ===", flush=True)
    best_val_rho = -1
    no_improve = 0
    results = {"epochs": [], "best_val_rho": 0.0}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        curr_train = train_df.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(curr_train), args.batch_size):
            batch = curr_train.iloc[i : i+args.batch_size]
            try:
                seqs = batch['sequence'].tolist()
                labels = torch.tensor(batch['efficiency'].values, dtype=torch.float32).to(device).unsqueeze(1)

                seq_tensor = prepare_batch_seq(seqs, backbone, tokenizer, device)
                epi_tensor = prepare_batch_epi(batch, args.use_epi, device)

                output = model(seq_tensor, epi_tracks=epi_tensor)
                loss_dict = model.compute_loss(output, labels)
                loss = loss_dict['total_loss']

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error in training batch: {e}", flush=True)
                continue

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for i in range(0, len(val_df), args.batch_size):
                batch = val_df.iloc[i : i+args.batch_size]
                try:
                    seqs = batch['sequence'].tolist()
                    seq_tensor = prepare_batch_seq(seqs, backbone, tokenizer, device)
                    epi_tensor = prepare_batch_epi(batch, args.use_epi, device)

                    output = model(seq_tensor, epi_tracks=epi_tensor)
                    val_preds.extend(output['mu'].cpu().numpy().flatten())
                    val_labels.extend(batch['efficiency'].values)
                except Exception as e:
                    print(f"Error in validation: {e}", flush=True)
                    continue

        if len(val_preds) > 0 and len(val_labels) > 0:
            val_rho, _ = spearmanr(val_preds, val_labels)
            epoch_loss = total_loss / max(1, (len(train_df) // args.batch_size))
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Val Rho: {val_rho:.4f}", flush=True)

            results["epochs"].append({"epoch": epoch+1, "loss": epoch_loss, "val_rho": val_rho})

            if val_rho > best_val_rho:
                best_val_rho = val_rho
                results["best_val_rho"] = float(val_rho)
                save_name = f"best_model_on_target_{backbone}.pt"
                torch.save(model.state_dict(), save_name)
                print(f"  -> New best Rho: {best_val_rho:.4f}", flush=True)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}: no improvement for {args.patience} epochs", flush=True)
                    break

    # Final Test
    print(f"\n=== Evaluating on Test Set ===", flush=True)
    model_path = f"best_model_on_target_{backbone}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded best model from {model_path}", flush=True)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for i in range(0, len(test_df), args.batch_size):
            batch = test_df.iloc[i : i+args.batch_size]
            try:
                seqs = batch['sequence'].tolist()
                seq_tensor = prepare_batch_seq(seqs, backbone, tokenizer, device)
                epi_tensor = prepare_batch_epi(batch, args.use_epi, device)
                output = model(seq_tensor, epi_tracks=epi_tensor)
                test_preds.extend(output['mu'].cpu().numpy().flatten())
                test_labels.extend(batch['efficiency'].values)
            except Exception as e:
                print(f"Error in test evaluation: {e}", flush=True)
                continue

    if len(test_preds) > 0 and len(test_labels) > 0:
        test_rho, _ = spearmanr(test_preds, test_labels)
        print(f"Final Test Rho: {test_rho:.4f}", flush=True)
        results["test_rho"] = float(test_rho)
    else:
        print("Warning: Could not compute test Rho (no predictions)", flush=True)

    output_path = args.output_name
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}", flush=True)

    if len(test_preds) > 0 and len(test_labels) > 0:
        pd.DataFrame({'sequence': test_df['sequence'][:len(test_preds)],
                      'efficiency': test_labels,
                      'prediction': test_preds}).to_csv(f"test_results_on_target_{backbone}.csv", index=False)
        print(f"Predictions saved to test_results_on_target_{backbone}.csv", flush=True)


if __name__ == '__main__':
    main()
