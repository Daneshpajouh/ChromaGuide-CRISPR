"""Quantum-inspired feature transforms (prototypical).

These features are lightweight, differentiable transforms intended as
experiment-friendly proxies for quantum-inspired corrections described in
the research plan. They are not quantum simulations.
"""
import torch
import torch.nn as nn


class SinusoidalQuantumFeatures(nn.Module):
    """Generate sinusoidal positional-like features to mimic interference.

    Uses multiple frequencies and optional learnable amplitudes.
    """
    def __init__(self, in_dim: int, n_freqs: int = 8, learnable: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.freqs = torch.linspace(1.0, float(n_freqs), n_freqs)
        if learnable:
            self.amplitudes = nn.Parameter(torch.randn(n_freqs) * 0.01)
        else:
            self.register_buffer('amplitudes', torch.ones(n_freqs))

    def forward(self, x: torch.Tensor):
        # x: (B, D)
        # Produce (B, n_freqs) features by global projection
        proj = x.mean(dim=-1, keepdim=False)
        out = []
        for i, f in enumerate(self.freqs.tolist()):
            out.append(torch.sin(proj * f) * self.amplitudes[i])
        return torch.stack(out, dim=-1)
