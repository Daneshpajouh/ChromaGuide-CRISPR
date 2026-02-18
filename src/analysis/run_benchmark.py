import torch
import torch.nn as nn
import argparse
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.model.crispro import CRISPROModel
from src.data.crisprofft import CRISPRoffTDataset

def evaluate_split(model, dataset, name="Test", device="cpu"):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds_cls = []
    all_targets_cls = []
    all_preds_reg = []
    all_targets_reg = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue

            # Unpack (Handle varying batch structure)
            if isinstance(batch, dict):
                 seqs = batch['sequence'].to(device)
                 epi = batch['epigenetics'].to(device)
                 targets = batch['efficiency'].to(device)
                 bio = batch.get('biophysics', None)
                 if bio is not None: bio = bio.to(device)
            else:
                 # Tuple fallback
                 continue

            targets_cls = (targets > 0).float()
            targets_reg = targets

            outputs = model(seqs, epigenetics=epi, biophysics=bio, causal=True, return_latents=True)

            # Handle Dict
            if isinstance(outputs, dict):
                pred_cls = outputs['classification']
                pred_reg = outputs['regression']
            else:
                pred_cls, pred_reg = outputs

            all_preds_cls.extend(pred_cls.squeeze().cpu().numpy())
            all_targets_cls.extend(targets_cls.cpu().numpy())

            # Masked Regression
            mask = targets_cls.cpu().numpy() > 0
            if mask.any():
                pred_numpy = pred_reg.squeeze().cpu().numpy()
                target_numpy = targets_reg.cpu().numpy()
                if pred_numpy.ndim == 0:
                     pred_numpy = np.array([pred_numpy])
                     target_numpy = np.array([target_numpy])

                all_preds_reg.extend(pred_numpy[mask])
                all_targets_reg.extend(target_numpy[mask])

    # Metrics
    auc = 0.5
    if len(np.unique(all_targets_cls)) > 1:
        auc = roc_auc_score(all_targets_cls, all_preds_cls)

    mse = mean_squared_error(all_targets_reg, all_preds_reg) if len(all_targets_reg) > 0 else 0.0

    print(f"[{name}] AUC: {auc:.4f} | MSE: {mse:.4f}")
    return auc, mse

def run_benchmark(checkpoint, data_path, device="cpu"):
    print("=== ZERO-SHOT BENCHMARK ===")

    # 1. Load Data
    full_dataset = CRISPRoffTDataset(data_path_override=data_path, context_window=4096)
    df = full_dataset.data # Access raw DF

    if 'Species' not in df.columns:
        print("Error: 'Species' column required for Zero-Shot Benchmarking.")
        return

    # 2. Split
    human_indices = df[df['Species'] == 'Homo sapiens'].index.tolist()
    mouse_indices = df[df['Species'] == 'Mus musculus'].index.tolist()

    human_data = torch.utils.data.Subset(full_dataset, human_indices)
    mouse_data = torch.utils.data.Subset(full_dataset, mouse_indices)

    print(f"Human Samples: {len(human_indices)}")
    print(f"Mouse Samples: {len(mouse_indices)}")

    # 3. Load Model
    model = CRISPROModel(
        d_model=256,
        n_layers=4,
        vocab_size=23,
        n_modalities=6,
        use_causal=True,
        use_quantum=True,
        use_topo=True
    )

    # Load weights
    try:
        sd = torch.load(checkpoint, map_location=device)
        model.load_state_dict(sd, strict=False)
        print("Model loaded.")
    except Exception as e:
        print(f"Checkpoint check failed: {e}")
        print("Running with random weights for testing pipeline.")

    model.to(device)

    # 4. Evaluate
    print("\n>>> EVALUATING SOURCE DOMAIN (Human)...")
    auc_h, mse_h = evaluate_split(model, human_data, name="Human (Source)", device=device)

    print("\n>>> EVALUATING TARGET DOMAIN (Mouse)...")
    auc_m, mse_m = evaluate_split(model, mouse_data, name="Mouse (Zero-Shot)", device=device)

    print("\n=== RESULTS ===")
    print(f"Transfer Delta (AUC): {auc_m - auc_h:.4f}")
    if auc_m > 0.85:
        print("SUCCESS: Zero-Shot Target Met (>0.85).")
    else:
        print("STATUS: Below Target (Needs more training).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/raw/crisprofft/CRISPRoffT_all_targets.txt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_benchmark(args.checkpoint, args.data, device=device)
