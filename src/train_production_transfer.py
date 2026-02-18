import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import sys
import argparse
import numpy as np

sys.path.append(os.getcwd())

from src.model.deepmens import DeepMEnsExact # Using DeepMEns for this demo, Mamba compatible
# from src.model.mamba_deepmens import MambaDeepMEns # Or Mamba
from src.data.crisprofft import CRISPRoffTDataset
from src.train_deepmens import DeepMEnsDatasetWrapper

def train_production_transfer():
    print("🚀 LAUNCHING PRODUCTION TRANSFER LEARNING (General -> Hek293t) 🚀")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model_path = "models/deepmens/deepmens_seed_0.pt"

    if not os.path.exists(base_model_path):
        print("Waiting for Base Model...")
        return

    # 1. Load Pre-trained General Model
    model = DeepMEnsExact(seq_len=23).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    print("Loaded General DeepMEns Model.")

    # 2. Prepare Target Data (SaCas9 Only - Real Domain Shift)
    print("Loading Target Domain Data (SaCas9)...")
    full_dataset = CRISPRoffTDataset(split='train', use_mini=False)

    # Filter for SaCas9 entries
    # CRITICAL FIX: Use Integer Positions (iloc) for Subset, not Index Labels (loc)
    # df.index are labels. We need 0..N positions.
    df = full_dataset.df.reset_index(drop=True) # Reset index to ensure loc==iloc consistency for safety
    # Or just find integer indices
    indices = df.index[df['Cas9_type'] == 'SaCas9'].tolist()

    # Update wrapper's internal DF ref if needed, but wrapper holds ref to original.
    # Actually, simpler: Reset index on the dataset's DF so .iloc matches .loc
    full_dataset.df = df

    print(f"Found {len(indices)} SaCas9 samples for Transfer Learning.")

    # Calculate Normalization Stats
    # DEPRECATED: Using Log1p normalization consistent with Base Dataset
    # sa_scores = df.loc[indices, 'Score']
    # min_score = sa_scores.min()
    # max_score = sa_scores.max()
    # print(f"Deactivating MinMax: Using log1p normalization instead.")

    if len(indices) == 0:
        print("Error: No SaCas9 samples found.")
        return

    # Create Train/Val Split (80/20) for Rigorous Evaluation
    # We have ~150 samples. 120 Train, 30 Test.
    split_point = int(0.8 * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    print(f"Split: {len(train_indices)} Train, {len(val_indices)} Validation.")

    train_dataset = Subset(DeepMEnsDatasetWrapper(full_dataset), train_indices)
    val_dataset = Subset(DeepMEnsDatasetWrapper(full_dataset), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize criterion for all evaluations
    criterion = nn.MSELoss()

    # 2b. BASELINE EVALUATION (Before Domain Adaptation)
    print("Evaluating Baseline (General Model) on Target Domain...")
    model.eval()
    baseline_loss = 0
    with torch.no_grad():
        for seq, shape, pos, label in val_loader:
             seq, shape, pos, label = seq.to(device), shape.to(device), pos.to(device), label.to(device)
             label_norm = torch.log1p(label) / torch.log1p(torch.tensor(1000.0, device=device))
             label_norm = torch.clamp(label_norm, 0.0, 1.0)
             pred = model(seq, shape, pos).squeeze()
             loss = criterion(pred, label_norm.float())
             baseline_loss += loss.item()
    print(f"📉 Baseline Validation MSE: {baseline_loss/len(val_loader):.6f}")

    # 3. Fine-Tuning Loop (Higher LR for Domain Shift)
    print("Fine-tuning on Target Domain...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) # Increased from 1e-5 to break freeze

    model.train()
    for epoch in range(5): # Increase to 5 epochs for convergence
        loss_epoch = 0
        for seq, shape, pos, label in train_loader:
             seq, shape, pos, label = seq.to(device), shape.to(device), pos.to(device), label.to(device)

             # NORMALIZE LABEL TO [0,1]
             # Score is usually 0-100 or 0-1000. DeepMEns outputs [0,1].
             # FIX: Use log1p logic from crisprofft.py
             # target = math.log1p(raw_val) / math.log1p(1000)
             label_norm = torch.log1p(label) / torch.log1p(torch.tensor(1000.0, device=device))
             label_norm = torch.clamp(label_norm, 0.0, 1.0)

             if epoch == 0 and loss_epoch == 0:
                 print(f"DEBUG: Label Min={label.min()}, Max={label.max()}")
                 print(f"DEBUG: Label Norm Min={label_norm.min()}, Max={label_norm.max()}")

             optimizer.zero_grad()
             pred = model(seq, shape, pos).squeeze()

             if epoch == 0 and loss_epoch == 0:
                 print(f"DEBUG: Pred Min={pred.min()}, Max={pred.max()}")

             loss = criterion(pred, label_norm.float()) # Ensure float
             loss.backward()
             optimizer.step()
             loss_epoch += loss.item()
        print(f"Transfer Epoch {epoch+1}: Train Loss {loss_epoch/len(train_loader):.4f}")

    # Final Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for seq, shape, pos, label in val_loader:
             seq, shape, pos, label = seq.to(device), shape.to(device), pos.to(device), label.to(device)
             label_norm = torch.log1p(label) / torch.log1p(torch.tensor(1000.0, device=device))
             label_norm = torch.clamp(label_norm, 0.0, 1.0)

             pred = model(seq, shape, pos).squeeze()
             loss = criterion(pred, label_norm.float())
             val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"✅ Transfer Learning Complete.")
    print(f"📊 Final Validation MSE (Held-out SaCas9): {avg_val_loss:.6f}")

    os.makedirs("models/transfer", exist_ok=True)
    torch.save(model.state_dict(), "models/transfer/deepmens_sacas9.pt")
    print("✅ Transfer Learning Complete: SaCas9 Specialist Model Saved.")

if __name__ == "__main__":
    train_production_transfer()
