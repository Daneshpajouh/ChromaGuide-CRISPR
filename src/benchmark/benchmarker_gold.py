
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import argparse
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, ndcg_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel # Fallback
# Try import RAG model
try:
    from src.model.crispr_rag_tta import CRISPR_RAG_TTA
except ImportError:
    pass

class GoldBenchmarkDataset(Dataset):
    """Simple wrapper to load pre-split Gold CSV directly."""
    def __init__(self, csv_path="test_set_GOLD.csv", context_window=4096):
        self.data = pd.read_csv(csv_path)
        self.context_window = context_window
        # Use existing CRISPRoffTDataset logic for tokenization if possible
        # Or simple re-implementation for stability
        # For consistency, we wrap the logic or create a minimal one.
        # Let's instantiate a "dummy" CRISPRoffTDataset just to reuse its processing methods if needed
        # But data loading is safer if independent.
        # For this sprint, let's reuse the robust loader but force it to use OUR dataframe

        self.loader_ref = CRISPRoffTDataset(use_mini=True) # Just for methods
        self.loader_ref.data = self.data # Inject data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Delegate to the robust processing logic
        return self.loader_ref.__getitem__(idx)

def run_benchmark(model_path, data_path="test_set_GOLD.csv", output_file="benchmark_results_gold.csv", device="cuda"):
    print(f"=== GOLD STANDARD BENCHMARK ===")
    print(f"Model: {model_path}")
    print(f"Data:  {data_path}")

    # 1. Load Data
    full_ds = CRISPRoffTDataset(use_mini=False, data_path_override=data_path)
    # The dataset loader might reload from disk, we need to ensure it uses the override logic
    # We added `data_path_override` in train.py logic discussion but did we verify `crisprofft.py` supports it?
    # If not, we might fall back to standard load.
    # Let's assume standard load for now or patch.
    # Actually, simpler: Initialize Dataset with NO file, then inject dataframe.

    ds = CRISPRoffTDataset(use_mini=True) # Init lightweight
    ds.data = pd.read_csv(data_path) # Inject GOLD data
    print(f"Loaded {len(ds)} samples from Gold Set.")

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 2. Load Model
    # Determine type (RAG vs Standard)
    checkpoint = torch.load(model_path, map_location=device)

    # Heuristic: Try RAG first
    try:
        model = CRISPR_RAG_TTA()
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded as CRISPR_RAG_TTA")
    except:
        print("Fallback to CRISPROModel")
        model = CRISPROModel(vocab_size=5)
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    # 3. Inference
    all_preds = []
    all_targets = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue

            seq = batch['sequence'].to(device)
            epi = batch['epigenetics'].to(device)
            target = batch['efficiency'].numpy()

            # Forward
            if hasattr(model, 'forward_rag'):
                # Handle RAG
                out = model(seq, epi) # wrapper should handle it
            else:
                out = model(seq, epi)

            # unpack
            if isinstance(out, dict):
                score = out['regression']
            elif isinstance(out, tuple):
                score = out[1] # usually (cls, reg)
            else:
                score = out

            all_preds.extend(score.cpu().numpy().flatten())
            all_targets.extend(target.flatten())

    # 4. Metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Handle NaNs
    mask = ~np.isnan(all_targets) & ~np.isnan(all_preds)
    p = all_preds[mask]
    t = all_targets[mask]

    spearman, p_val = spearmanr(t, p)
    pearson, _ = pearsonr(t, p)

    from sklearn.metrics import roc_auc_score, ndcg_score, average_precision_score

    # AUC (Binarize at > 0)
    t_bin = (t > 0).astype(int)
    if len(np.unique(t_bin)) > 1:
        auc = roc_auc_score(t_bin, p)
        auc_pr = average_precision_score(t_bin, p)
    else:
        auc = 0.5
        auc_pr = 0.5

    print("\n------------------------------------------------")
    print(f"RESULTS for {model_path}")
    print(f"Spearman Rho: {spearman:.4f} (p={p_val:.2e})")
    print(f"Pearson R:    {pearson:.4f}")
    print(f"AUC-ROC:      {auc:.4f}")
    print(f"AUC-PR:       {auc_pr:.4f} (CCLMoff Target: >0.81)")
    print("------------------------------------------------\n")

    # Save
    results = pd.DataFrame([{
        'model': model_path,
        'spearman': spearman,
        'pearson': pearson,
        'auc_roc': auc,
        'auc_pr': auc_pr,
        'n_samples': len(p)
    }])

    if not os.path.exists(output_file):
        results.to_csv(output_file, index=False)
    else:
        results.to_csv(output_file, mode='a', header=False, index=False)

def collate_fn(batch):
    # Same as train.py
    batch = [x for x in batch if x['efficiency'] is not None and not np.isnan(x['efficiency'])]
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data", type=str, default="test_set_GOLD.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    run_benchmark(args.checkpoint, args.data, device=args.device)
