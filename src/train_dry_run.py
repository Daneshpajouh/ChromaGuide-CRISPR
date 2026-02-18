import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.open_crispr import OpenCRISPRDataset
from src.model.crispro import CRISPROModel
import time

def train_dry_run():
    print("=== CRISPRO-MAMBA-X: Training Dry Run ===")

    # 1. Device Config
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Training on: {device.upper()}")

    # 2. Data
    print("Loading Data (Mini SOTA)...")
    from src.data.crisprofft import CRISPRoffTDataset
    dataset = CRISPRoffTDataset(use_mini=True)

    # Robust Collate: Filter NaNs and ensure Float32
    def collate_fn(batch):
        # Filter None or NaN entries
        clean_batch = []
        for x in batch:
            if x['efficiency'] is not None and not torch.isnan(torch.tensor(x['efficiency'])):
                clean_batch.append(x)

        if not clean_batch:
            return None # Handle empty batch in loop

        return torch.utils.data.dataloader.default_collate(clean_batch)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 3. Model
    print("Initializing Unified CRISPRO Model (DNA + Real Epigenetics)...")
    from src.model.crispro import CRISPROModel

    # Vocabulary helper
    # Simple AA/DNA tokenizer map
    vocab = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}
    def tokenize(seq_list):
        max_len = max([len(s) for s in seq_list])
        batch = torch.zeros(len(seq_list), max_len, dtype=torch.long)
        for i, s in enumerate(seq_list):
            for j, char in enumerate(s):
                batch[i, j] = vocab.get(char.upper(), 0)
        return batch

    model = CRISPROModel(
        d_model=64, # Matches Mamba Block dim
        n_layers=2,
        n_modalities=6 # matches epi_vocab length in loader
    ).to(device)

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-4) # Slightly lower LR for stability
    criterion = nn.MSELoss()

    # 5. Loop (1 Epoch)
    print("Starting Training Loop (Real Data Only)...")
    model.train()
    start_time = time.time()

    losses = []

    for i, batch in enumerate(dataloader):
        if batch is None: continue

        # Real Sequences
        raw_seqs = batch["sequence"]
        sequences = tokenize(raw_seqs).to(device)

        # Real Epigenetics (Batch, N_Tracks)
        epigenetics = batch["epigenetics"].to(device)

        # Real Targets
        targets = batch["efficiency"].to(device, dtype=torch.float32)

        # Forward (Unified)
        optimizer.zero_grad()
        predictions = model(sequences, epi_tracks=epigenetics)

        loss = criterion(predictions, targets)

        if torch.isnan(loss):
            print(f"WARNING: NaN Loss at step {i}. Skipping update.")
            continue

        loss.backward()

        # Gradient Clipping (Essential for Mamba/RNNs on MPS)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())

        print(f"Step {i}: Loss = {loss.item():.4f}")

        if i >= 20: break # Stop after 20 steps for dry run

    duration = time.time() - start_time
    print(f"\nDry Run Complete.")
    print(f"Time: {duration:.2f}s")
    print(f"Final Loss: {losses[-1]:.4f}")

    if losses[-1] < losses[0]:
        print("SUCCESS: Loss is decreasing. Model is learning.")
    else:
        print("WARNING: Loss did not decrease. Check hyperparameters.")

if __name__ == "__main__":
    train_dry_run()
