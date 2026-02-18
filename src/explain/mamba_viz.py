import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class MambaSaliency:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def compute_saliency(self, x_seq, x_shape, x_pos):
        """
        Compute Input Gradient Saliency.
        x_seq: (1, 4, 23) - Requires Grad
        """
        # Enable grad for input
        x_seq.requires_grad = True

        # Forward
        score = self.model(x_seq, x_shape, x_pos)

        # Backward
        self.model.zero_grad()
        score.backward()

        # Saliency = Magnitude of Gradient
        grad = x_seq.grad.data.abs() # (1, 4, 23)
        saliency, _ = torch.max(grad, dim=1) # (1, 23) collapse channels

        return saliency.squeeze().cpu().numpy()

    def effective_receptive_field(self, seq_len=23):
        """
        Analyze ERF by computing gradient of center output w.r.t input.
        """
        # Random input
        x_seq = torch.randn(1, 4, seq_len, requires_grad=True)
        x_shape = torch.randn(1, 4, seq_len)
        x_pos = torch.arange(seq_len).unsqueeze(0)

        # Forward
        score = self.model(x_seq, x_shape, x_pos)

        # Gradient
        score.backward()

        erf = x_seq.grad.data.abs().sum(dim=1).squeeze() # (23,)
        return erf.cpu().numpy()

if __name__ == "__main__":
    # Test with Mock Model
    class MockModel(nn.Module):
        def forward(self, x, s, p):
            return (x * 0.5).sum()

    viz = MambaSaliency(MockModel())
    x = torch.randn(1, 4, 23)
    s = torch.randn(1, 4, 23)
    p = torch.arange(23).unsqueeze(0)

    sal = viz.compute_saliency(x, s, p)
    print("Saliency Map (Mock):", sal)
