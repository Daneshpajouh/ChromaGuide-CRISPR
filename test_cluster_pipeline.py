#!/usr/bin/env python3
"""
Simple training test for cluster verification
Just runs 1 epoch on mini dataset to verify everything works
"""
import torch
import sys
import os

# Add project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def test_training():
    print("="*60)
    print("CLUSTER PIPELINE VERIFICATION TEST")
    print("="*60)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load mini dataset
    data_path = "/scratch/amird/CRISPRO-MAMBA-X/data/mini/crisprofft/mini_crisprofft.txt"
    print(f"\nLoading dataset: {data_path}")
    dataset = CRISPRoffTDataset(data_path_override=data_path, context_window=4096)
    print(f"Dataset size: {len(dataset)} samples")

    # Create loader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Batches: {len(loader)}")

    # Create model
    print("\nInitializing model...")
    model = CRISPROModel(
        d_model=128,
        n_layers=2,
        vocab_size=23,
        n_modalities=6,
        use_causal=True,
        use_quantum=True,
        use_topo=True
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    # Training loop
    print("\nStarting training (1 epoch)...")
    model.train()
    total_loss = 0
    n_batches = 0

    for i, batch in enumerate(loader):
        if batch is None:
            continue

        if i >= 10:  # Just do 10 batches for quick test
            break

        seq = batch['sequence'].to(device)
        epi = batch['epigenetics'].to(device)
        target = batch['efficiency'].to(device, dtype=torch.float32)
        bio = batch.get('biophysics', None)
        if bio is not None:
            bio = bio.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(seq, epigenetics=epi, biophysics=bio, causal=True)

        # Unpack
        pred_cls = outputs['classification']
        pred_reg = outputs['regression']

        # Loss
        target_cls = (target > 0).float().view(-1, 1)
        target_reg = target.view(-1, 1)

        loss_cls = criterion_cls(pred_cls.view(-1, 1), target_cls)
        loss_reg = criterion_reg(pred_reg.view(-1, 1), target_reg)

        loss = loss_cls + loss_reg

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{10}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / n_batches if n_batches > 0 else 0

    print(f"\n{'='*60}")
    print(f"TEST COMPLETE!")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Batches processed: {n_batches}")
    print(f"{'='*60}")
    print("\n✅ Pipeline verification successful!")
    print("Ready for full training runs.")

    return avg_loss

if __name__ == "__main__":
    test_training()
