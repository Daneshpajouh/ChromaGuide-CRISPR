import torch
import torch.nn as nn
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.model.crispro import CRISPROModel
from src.data.crisprofft import CRISPRoffTDataset

def intervention_forward(model, seqs, epi, bio, do_chromatin=None):
    """
    Performs a forward pass with a Causal Intervention.
    do_chromatin: float or tensor. If set, overrides the 'z_chrom' latent.
    """
    intervention = {}
    if do_chromatin is not None:
        # Let's assume 'A' (Accessibility) is the downstream node we want to control via 'C'.
        # Or better, we define 'C' (Chromatin) intervention.
        # Let's check causal.py again to be sure what keys it supports.
        # Assuming it is generic or we map 'chrom' -> 'C' or 'A'.
        # Let's try 'A' (Accessibility) as the proxy for Chromatin state.

        # Create a broadcasted tensor for intervention value
        # We need the batch size from seqs
        val_tensor = torch.full((seqs.size(0), 32), do_chromatin, device=seqs.device)
        intervention['A'] = val_tensor

    # 1. Forward Pass with Intervention
    # We leverage the updated CRISPROModel.forward which handles pooling and intervention
    outputs = model(seqs, epigenetics=epi, biophysics=bio, causal=True, causal_intervention=intervention)

    # 2. Extract Causal Output
    # outputs['causal'] = (logits, latents)
    if 'causal' in outputs:
         logits, _ = outputs['causal']
         return logits
    else:
         raise ValueError("Model did not return causal output!")

def run_counterfactual(checkpoint, data_path, device="cpu"):
    print("=== COUNTERFACTUAL LAB: Virtual CRISPR Experiments ===")

    # 1. Load Model
    model = CRISPROModel(
        d_model=256,
        n_layers=4,
        vocab_size=23,
        n_modalities=6, # Standard
        use_causal=True,
        use_quantum=True,
        use_topo=True
    )

    try:
        sd = torch.load(checkpoint, map_location=device)
        model.load_state_dict(sd, strict=False)
        print("Model loaded.")
    except:
        print("Warning: Running with random weights (Infrastructure Test).")

    model.to(device)
    model.eval()

    # 2. Load Sample Data (Human)
    # We want a batch of sequences to test "Same Sequence, Different Chromatin"
    dataset = CRISPRoffTDataset(data_path, context_window=4096, max_samples=100)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)

    batch = next(iter(loader))
    seqs = batch['sequence'].to(device)
    epi = batch['epigenetics'].to(device)
    bio = batch.get('biophysics', None)
    if bio is not None: bio = bio.to(device)

    # 3. Run Experiment
    # Scenario A: Natural (Observational)
    print(">> Running Observational Pass...")
    with torch.no_grad():
        y_natural = intervention_forward(model, seqs, epi, bio, do_chromatin=None).squeeze()

    # Scenario B: Force Open Chromatin (High Accessibility)
    # Value 5.0 (activations roughly -5 to 5?)
    print(">> Intervention: do(Accessibility = 2.0) [OPEN]")
    with torch.no_grad():
        y_open = intervention_forward(model, seqs, epi, bio, do_chromatin=2.0).squeeze()

    # Scenario C: Force Closed Chromatin (Low Accessibility)
    print(">> Intervention: do(Accessibility = -2.0) [CLOSED]")
    with torch.no_grad():
        y_closed = intervention_forward(model, seqs, epi, bio, do_chromatin=-2.0).squeeze()

    # 4. Analysis (ATE)
    # Average Treatment Effect
    ate_open = (y_open - y_natural).mean().item()
    ate_closed = (y_closed - y_natural).mean().item()

    print("\n=== CAUSAL REPORT ===")
    print(f"Baseline Efficiency (Mean): {y_natural.mean().item():.4f}")
    print(f"ATE (Open Chromatin): {ate_open:+.4f}")
    print(f"ATE (Closed Chromatin): {ate_closed:+.4f}")

    if ate_open > 0 and ate_closed < 0:
        print("CONCLUSION: Positive Causal Link verified (Open -> Higher Eff).")
    else:
        print("CONCLUSION: Causal Link Unclear (or Random Weights).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/raw/crisprofft/CRISPRoffT_all_targets.txt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_counterfactual(args.checkpoint, args.data, device=device)
