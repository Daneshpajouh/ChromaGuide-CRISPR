
import torch
import torch.nn as nn
import sys
import os

# Mock the optimizer to test logic
sys.path.append("/Users/studio/Desktop/PhD/Proposal")
from src.model.geometric_optimizer import GeometricCRISPROptimizer

def test_geometric_local():
    print("Testing Geometric Optimizer Locally (Mac MPS/CPU)...")

    # 1. Setup Simple Model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = SimpleModel().to(device)
    optimizer = GeometricCRISPROptimizer(model, lr=0.1, damping=1e-3)

    # 2. Dummy Data
    inputs = torch.randn(32, 10).to(device)
    targets = torch.randn(32, 1).to(device)

    # 3. Step
    print("Step 1: Compute Fisher Matrix & Update...")
    loss = optimizer.step(inputs, targets)
    print(f"Loss: {loss:.4f}")

    # Check weights changed
    print("Verifying weight update...")
    original_weight = model.linear.weight.clone()
    optimizer.step(inputs, targets)
    new_weight = model.linear.weight

    if not torch.equal(original_weight, new_weight):
        print("✅ Weights updated successfully via Natural Gradient!")
    else:
        print("❌ Weights did not change!")

    print("\nLocal geometric verification complete.")

if __name__ == "__main__":
    test_geometric_local()
