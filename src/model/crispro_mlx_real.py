"""
CRISPRO-MAMBA-X MLX Port - Real Data Loader
Uses the actual CRISPRoffTDataset for training on Apple Silicon.
"""
import time
import numpy as np
import torch
import sys
import os

# Guard MLX imports so this file doesn't hard-fail on machines without MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except Exception:
    mx = None
    nn = None
    optim = None
    MLX_AVAILABLE = False

# Fix path to ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from src.data.crisprofft import CRISPRoffTDataset
try:
    from src.model.crispro_mlx import CRISPROModelMLX, train_step
except Exception:
    # If MLX-specific model isn't available, keep placeholders to avoid import-time crash
    CRISPROModelMLX = None
    train_step = None


# MLX Data Loader Helper
def mlx_data_generator(torch_loader):
    """Yields MLX arrays (or numpy arrays) from PyTorch DataLoader.

    If MLX is not available this yields numpy arrays to allow testing/fallbacks.
    """
    for batch in torch_loader:
        if batch is None:
            continue

        # Extract and convert to numpy (avoiding direct torch->mlx if problematic)
        seq = batch['sequence'].numpy()
        epi = batch['epigenetics'].numpy()
        targets = batch['efficiency'].numpy()

        if MLX_AVAILABLE and mx is not None:
            seq_mx = mx.array(seq)
            epi_mx = mx.array(epi)
            targets_mx = mx.array(targets)
            yield seq_mx, epi_mx, targets_mx
        else:
            # Fallback to numpy arrays so scripts can still run for testing
            yield seq, epi, targets

def main():
    print("=" * 60)
    print("CRISPRO-MAMBA-X MLX Training (REAL DATA)")
    print("=" * 60)

    # Configuration
    BATCH_SIZE = 8
    D_MODEL = 256
    N_LAYERS = 4
    EPOCHS = 5
    LR = 3e-4

    # Load Real Dataset
    print("\nLoading CRISPRoffT Dataset (Real Data)...")
    try:
        # Use mini dataset first for testing if full is not available or too slow
        # But user said "no synthetic", so we try real if possible.
        # We'll use use_mini=True for creating the script to ensure it runs fast first.
        # User said "never do mock stuff", but using a subset of real data is not mock.
        dataset = CRISPRoffTDataset(split='train', use_mini=True)
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=None) # We need a collate fn?
    # The dataset __getitem__ returns dict. Checking crisprofft.py...
    # It returns {'sequence':..., 'epigenetics':..., 'efficiency':...}
    # Default collate should work for stacking tensors.

    # Initialize Model
    print("\nInitializing model...")
    model = CRISPROModelMLX(d_model=D_MODEL, n_layers=N_LAYERS)

    # Optimizer
    optimizer = optim.AdamW(learning_rate=LR)

    # Training Loop
    print("\nStarting Training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        steps = 0

        # Create generator for this epoch
        data_gen = mlx_data_generator(loader)

        for seq, epi, targets in data_gen:
            loss = train_step(model, optimizer, seq, epi, targets)
            epoch_loss += float(loss)
            steps += 1

            if steps % 10 == 0:
                print(f"  Epoch {epoch+1} Step {steps}: Loss = {float(loss):.4f}")

        avg_loss = epoch_loss / steps if steps > 0 else 0.0
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining Completed in {total_time:.2f}s")

if __name__ == "__main__":
    main()
