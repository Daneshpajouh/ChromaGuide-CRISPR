import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse

sys.path.append(os.getcwd())

from src.model.mamba_deepmens import MambaDeepMEns
from src.data.crisprofft import CRISPRoffTDataset
from src.train_deepmens import DeepMEnsDatasetWrapper

def transfer_learning_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Transfer Learning: {args.pretrained_path} -> Cas9")

    # 1. Load Pre-trained Model
    model = MambaDeepMEns(seq_len=23).to(device)
    if os.path.exists(args.pretrained_path):
        print(f"Loading weights from {args.pretrained_path}")
        # Note: strict=False allows loading if head dimensions differ slightly
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
    else:
        print("⚠️ Pretrained weights not found. Using random init (Validation only).")

    # Load Data (Target Task = Cas9)
    train_base = CRISPRoffTDataset(split='train', use_mini=args.use_mini)
    train_dataset = DeepMEnsDatasetWrapper(train_base)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.MSELoss()

    # ==========================
    # STAGE 1: HEAD TUNING
    # ==========================
    print("\n🧊 STAGE 1: Freezing Backbone, Training Head")

    # Freeze Backbone
    # We identify backbone by layers that are NOT 'fc' or 'output'
    # Or explicitly: conv_seq, mamba, conv_shape, etc.
    for name, param in model.named_parameters():
        if "fc" in name or "output" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # Train Stage 1
    for epoch in range(args.epochs_stage1):
        model.train()
        loss_epoch = 0
        for seq, shape, pos, label in train_loader:
            seq, shape, pos, label = seq.to(device), shape.to(device), pos.to(device), label.to(device)
            optimizer_stage1.zero_grad()
            pred = model(seq, shape, pos).squeeze()
            loss = criterion(pred, label)
            loss.backward()
            optimizer_stage1.step()
            loss_epoch += loss.item()
        print(f"Stage 1 Epoch {epoch+1} Loss: {loss_epoch/len(train_loader):.4f}")

    # ==========================
    # STAGE 2: FINE-TUNING
    # ==========================
    print("\n🔥 STAGE 2: Unfreezing All, Fine-Tuning (Low LR)")

    # Unfreeze All
    for param in model.named_parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.AdamW(model.parameters(), lr=1e-5) # Low LR

    # Train Stage 2
    for epoch in range(args.epochs_stage2):
        model.train()
        loss_epoch = 0
        for seq, shape, pos, label in train_loader:
            seq, shape, pos, label = seq.to(device), shape.to(device), pos.to(device), label.to(device)
            optimizer_stage2.zero_grad()
            pred = model(seq, shape, pos).squeeze()
            loss = criterion(pred, label)
            loss.backward()
            optimizer_stage2.step()
            loss_epoch += loss.item()
        print(f"Stage 2 Epoch {epoch+1} Loss: {loss_epoch/len(train_loader):.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "mamba_cas9_finetuned.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved fine-tuned model to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to PrimeNet weights")
    parser.add_argument("--epochs_stage1", type=int, default=5, help="Stage 1 Epochs")
    parser.add_argument("--epochs_stage2", type=int, default=10, help="Stage 2 Epochs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_mini", action="store_true")
    parser.add_argument("--output_dir", type=str, default="models/transfer")

    args = parser.parse_args()
    transfer_learning_pipeline(args)
