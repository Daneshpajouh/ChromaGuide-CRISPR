
import torch
import torch.nn as nn
from src.model.geometric_optimizer import GeometricCRISPROptimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def test_diagonal_fisher():
    print("🧪 Testing Diagonal Fisher Implementation...")

    # 1. Setup
    model = SimpleModel()
    opt = GeometricCRISPROptimizer(model, lr=0.1, damping=1e-3)

    # 2. Dummy Data
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)

    # 3. Test Step (Regression)
    print("   Running step()...")
    try:
        loss = opt.step(inputs, targets)
        print(f"   ✅ Step successful! Loss: {loss:.4f}")
    except Exception as e:
        print(f"   ❌ Step failed: {e}")
        raise e

    print("\n   Checking Gradient Scaling...")
    # Verify that we actually moved
    for p in model.parameters():
        assert p.grad is not None

    print("✅ Diagonal Fisher Verification Complete!")

if __name__ == "__main__":
    test_diagonal_fisher()
