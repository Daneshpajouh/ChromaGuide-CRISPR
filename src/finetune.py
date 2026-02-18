import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
import os
import tqdm
import pandas as pd

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from src.model.crispro import CRISPROModel
from src.data.crisprofft import CRISPRoffTDataset

def finetune(args):
    print("=== FEW-SHOT FINE-TUNING (Cas12a) ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data (Few-Shot Adapter)
    print(f"Loading Few-Shot Data: {args.data}")
    dataset = CRISPRoffTDataset(data_path_override=args.data, context_window=4096)
    print(f"Samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Load Model
    print("Initializing Foundation Model...")
    model = CRISPROModel(
        d_model=256,
        n_layers=4, # Must match pretraining
        vocab_size=23,
        n_modalities=6,
        use_causal=True,
        use_quantum=True,
        use_topo=True
    )

    # Load Weights
    try:
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd, strict=False)
        print("Pretrained Weights Loaded.")
    except Exception as e:
        print(f"Weight Load Warning: {e}")
        print("Proceeding with Random Weights (for testing pipeline).")

    model.to(device)

    # 3. FREEZE BACKBONE (Transfer Learning)
    # We freeze everything except the heads and maybe the last layer?
    # Strategy: Freeze Encoder, Train Heads.
    print("Freezing Backbone (Mamba Encoder + Embeddings)...")
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze Heads
    print("Unfreezing Prediction Heads...")
    for param in model.class_head.parameters():
        param.requires_grad = True
    for param in model.reg_head.parameters():
        param.requires_grad = True

    # Novelty: Unfreeze Biophysical Gate? Allows adapting to new physics (Cas12a vs Cas9)?
    if hasattr(model, 'physics_proj'):
        for param in model.physics_proj.parameters():
            param.requires_grad = True

    # 4. Optimizer
    # Low LR for fine-tuning
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    # 5. Training Loop
    model.train()
    history = []

    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            if batch is None: continue

            # Unpack
            seqs = batch['sequence'].to(device)
            epi = batch['epigenetics'].to(device)
            targets_raw = batch['efficiency'].to(device, dtype=torch.float32)
            bio = batch.get('biophysics', None)
            if bio is not None: bio = bio.to(device)

            targets_cls = (targets_raw > 0).float().view(-1, 1)
            targets_reg = targets_raw.view(-1, 1)

            # Forward
            optimizer.zero_grad()
            outputs = model(seqs, epigenetics=epi, biophysics=bio, causal=True) # Check args

            # Unpack
            preds_cls = outputs['classification']
            preds_reg = outputs['regression']

            # Loss
            loss_cls = criterion_cls(preds_cls, targets_cls)
            loss_reg = criterion_reg(preds_reg, targets_reg)

            loss = loss_cls + loss_reg

            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        history.append(avg_loss)

    # 6. Save Adapter
    print("Saving Fine-Tuned Adapter...")
    os.makedirs("checkpoints/adapters", exist_ok=True)
    save_path = "checkpoints/adapters/cas12a_adapter.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Foundation Model")
    parser.add_argument("--data", type=str, required=True, help="Path to Few-Shot Data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    finetune(args)
