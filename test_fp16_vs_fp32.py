import torch
import os

# Quick test to see if disabling FP16 fixes NaN
print("Testing FP16 vs FP32 on GPU...")

# Load modules
import sys
sys.path.insert(0, "/home/amird/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X")

from src.model.crispro import CRISPROModel
from src.utils.loss import CombinedLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Initialize model
model = CRISPROModel(d_model=256, n_layers=4, context_window=4096).to(device)
criterion = CombinedLoss()

# Create dummy input
batch_size = 4
seq_len = 4096
dummy_seqs = torch.randint(0, 4, (batch_size, seq_len), device=device)
dummy_epi = torch.randn(batch_size, 5, device=device)
targets = torch.rand(batch_size, device=device)

# Test with FP16
print("\n=== Testing with FP16 (mixed precision) ===")
with torch.amp.autocast(device_type=device, dtype=torch.float16):
    pred_cls, pred_reg = model(dummy_seqs, dummy_epi)
    pred_cls = pred_cls.float()
    pred_reg = pred_reg.float()
    loss = criterion(pred_cls, pred_reg, targets, targets)
print(f"Loss with FP16: {loss.item()}")

# Test with FP32
print("\n=== Testing with FP32 (no mixed precision) ===")
pred_cls, pred_reg = model(dummy_seqs, dummy_epi)
loss = criterion(pred_cls, pred_reg, targets, targets)
print(f"Loss with FP32: {loss.item()}")
