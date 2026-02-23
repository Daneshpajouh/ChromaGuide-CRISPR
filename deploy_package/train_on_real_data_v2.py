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

def main():
    parser = argparse.ArgumentParser(description='ChromaGuide Production Training')
    parser.add_argument('--backbone', type=str, default='cnn_gru', choices=['cnn_gru', 'mamba', 'dnabert2'], help='Backbone types')
    parser.add_argument('--fusion', type=str, default='gate', choices=['gate', 'concat'], help='Fusion type')
    parser.add_argument('--use_epi', action='store_true', default=True, help='Use epigenomics')
    parser.add_argument('--no_epi', action='store_false', dest='use_epi', help='Disable epigenomics')
    parser.add_argument('--learning_rate', '--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cpu', help='Device for training')
    parser.add_argument('--split', type=str, default='A', choices=['A', 'B', 'C'], help='Split type')
    parser.add_argument('--output_name', type=str, default='on_target_metrics.json', help='Output JSON file')
    args = parser.parse_args()

    # Determine device
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}", flush=True)
    print(f"Config: Backbone={args.backbone}, Fusion={args.fusion}, UseEpi={args.use_epi}, Split={args.split}", flush=True)

    # Load data from splits
    split_name_map = {
        'A': 'split_a_gene_held_out',
        'B': 'split_b_dataset_held_out',
        'C': 'split_c_cellline_held_out'
    }
    split_dir = Path(f"data/processed/{split_name_map[args.split.upper()]}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} not found. Run preprocessing first.")

    # Combine all cell lines for training/val/test
    train_dfs = [pd.read_csv(f) for f in split_dir.glob("*_train.csv")]
    val_dfs = [pd.read_csv(f) for f in split_dir.glob("*_validation.csv")]
    test_dfs = [pd.read_csv(f) for f in split_dir.glob("*_test.csv")]

    if not train_dfs or not val_dfs or not test_dfs:
        raise ValueError("Missing split files. Ensure splits are generated correctly.")

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    # Initialize Tokenizer (if needed)
    tokenizer = None
    if args.backbone == 'dnabert2':
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load DNABERT-2 tokenizer online. Using local fallback if available. Error: {e}")

    # Initialize Model
    model = ChromaGuideModel(
        encoder_type=args.backbone,
        d_model=64, # PhD Proposal constraint
        seq_len=21,
        num_epi_tracks=11 if args.use_epi else 0,
        num_epi_bins=1,
        use_epigenomics=args.use_epi,
        use_gate_fusion=(args.fusion == 'gate'),
        dropout=0.1,
    ).to(device)

    # Optimization
    if args.backbone == 'dnabert2':
        # Low LR for transformer backbone
        optimizer = torch.optim.AdamW([
            {'params': model.seq_encoder.parameters(), 'lr': 1e-5},
            {'params': [p for n, p in model.named_parameters() if 'seq_encoder' not in n], 'lr': 1e-4}
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Model initialized: {args.backbone} with {args.fusion} fusion")

    # Training Loop
    best_val_rho = -1
    no_improve = 0
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    results = {"epochs": [], "best_val_rho": 0.0}

    def prepare_batch_seq(seqs, backbone, tokenizer, device):
        if backbone == 'dnabert2' and tokenizer is not None:
            return tokenizer(seqs, return_tensors='pt', padding=True).to(device)['input_ids']
        else:
            # Map sequence to (B, 4, 21)
            # Find the actual length of the sequences in the dataset
            # If they are 21, use 21. If 23, use 23. The model was fixed to 21 or adapts if d_model matches.
            # Actually, DeepHF is 21-nt usually, but our model expects 23? Let's check.
            # Chapter 2 says 21-nt.
            L = len(seqs[0]) if seqs else 23
            seq_tensor = torch.zeros(len(seqs), 4, L, device=device)
            for j, seq in enumerate(seqs):
                for k, nt in enumerate(seq[:L]):
                    if nt.upper() in nt_map: seq_tensor[j, nt_map[nt.upper()], k] = 1
            return seq_tensor

    def prepare_batch_epi(batch, use_epi, device):
        if not use_epi: return None
        # Extract feat_0 to feat_10
        cols = [f'feat_{i}' for i in range(11)]
        epi_data = batch[cols].values.astype(np.float32)
        # Shape (B, 11) -> (B, 11, 1) to match (B, tracks, bins)
        return torch.tensor(epi_data, device=device).unsqueeze(2)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        curr_train = train_df.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(curr_train), args.batch_size):
            batch = curr_train.iloc[i : i+args.batch_size]
            seqs = batch['sequence'].tolist()
            labels = torch.tensor(batch['efficiency'].values, dtype=torch.float32).to(device).unsqueeze(1)

            seq_tensor = prepare_batch_seq(seqs, args.backbone, tokenizer, device)
            epi_tensor = prepare_batch_epi(batch, args.use_epi, device)

            output = model(seq_tensor, epi_tracks=epi_tensor)
            loss_dict = model.compute_loss(output, labels)
            loss = loss_dict['total_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for i in range(0, len(val_df), args.batch_size):
                batch = val_df.iloc[i : i+args.batch_size]
                seqs = batch['sequence'].tolist()
                seq_tensor = prepare_batch_seq(seqs, args.backbone, tokenizer, device)
                epi_tensor = prepare_batch_epi(batch, args.use_epi, device)

                output = model(seq_tensor, epi_tracks=epi_tensor)
                val_preds.extend(output['mu'].cpu().numpy().flatten())
                val_labels.extend(batch['efficiency'].values)

        val_rho, _ = spearmanr(val_preds, val_labels)
        epoch_loss = total_loss/(len(train_df)/args.batch_size)
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Rho: {val_rho:.4f}", flush=True)

        results["epochs"].append({"epoch": epoch+1, "loss": epoch_loss, "val_rho": val_rho})

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            results["best_val_rho"] = float(val_rho)
            save_name = f"best_model_on_target.pt"
            torch.save(model.state_dict(), save_name)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience: break

    # Final Test
    print("Evaluating on GOLD set...")
    if os.path.exists("best_model_on_target.pt"):
        model.load_state_dict(torch.load("best_model_on_target.pt"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for i in range(0, len(test_df), args.batch_size):
            batch = test_df.iloc[i : i+args.batch_size]
            seqs = batch['sequence'].tolist()
            seq_tensor = prepare_batch_seq(seqs, args.backbone, tokenizer, device)
            epi_tensor = prepare_batch_epi(batch, args.use_epi, device)
            output = model(seq_tensor, epi_tracks=epi_tensor)
            test_preds.extend(output['mu'].cpu().numpy().flatten())
            test_labels.extend(batch['efficiency'].values)

    test_rho, _ = spearmanr(test_preds, test_labels)
    print(f"FINAL GOLD Rho: {test_rho:.4f}")
    results["gold_rho"] = float(test_rho)

    output_path = args.output_name
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

    pd.DataFrame({'sequence': test_df['sequence'], 'efficiency': test_labels, 'prediction': test_preds}).to_csv(f"gold_results_on_target.csv", index=False)

if __name__ == '__main__':
    main()
