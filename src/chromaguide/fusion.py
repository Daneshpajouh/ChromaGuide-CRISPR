"""Fusion module for ChromaGuide.

Combines sequence embedding z_s and epigenomic embedding z_e through
concatenation + MLP. Optionally includes an information-theoretic
non-redundancy regularizer (MINE/CLUB) to encourage complementary
representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChromaGuideFusion(nn.Module):
    """Multi-modal fusion: concatenation + MLP.

    Baseline: f = MLP([z_s; z_e]) -> z_fused in R^{d_model}
    
    Can be replaced with gating/cross-attention/mixture-of-experts
    if needed.
    """
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_gate: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_gate = use_gate

        input_dim = d_model * 2  # concat z_s and z_e

        if use_gate:
            # Gated fusion: learn attention weights for each modality
            self.gate = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.Sigmoid(),
            )
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            # Simple concatenation + MLP
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.LayerNorm(d_model),
            )

    def forward(
        self,
        z_s: torch.Tensor,
        z_e: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse sequence and epigenomic embeddings.

        Args:
            z_s: Sequence embedding (batch, d_model)
            z_e: Epigenomic embedding (batch, d_model)

        Returns:
            Fused embedding (batch, d_model)
        """
        h = torch.cat([z_s, z_e], dim=-1)  # (batch, 2*d_model)

        if self.use_gate:
            gate_weights = self.gate(h)  # (batch, d_model)
            fused = gate_weights * z_s + (1 - gate_weights) * z_e
            return self.proj(h) + fused
        else:
            return self.proj(h)


class NonRedundancyRegularizer(nn.Module):
    """Optional MI-based non-redundancy regularizer.

    Implements MINE (Mutual Information Neural Estimation) or CLUB
    (Contrastive Log-ratio Upper Bound) to encourage z_s and z_e
    to capture complementary information.

    Loss = -lambda * I(z_s; z_e)  (minimize mutual information)
    """
    def __init__(self, d_model: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_s: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        """Estimate mutual information between z_s and z_e using MINE."""
        B = z_s.shape[0]

        # Joint samples: (z_s_i, z_e_i)
        joint = torch.cat([z_s, z_e], dim=-1)
        t_joint = self.network(joint)  # (B, 1)

        # Marginal samples: (z_s_i, z_e_j) with shuffled z_e
        perm = torch.randperm(B, device=z_s.device)
        z_e_shuffled = z_e[perm]
        marginal = torch.cat([z_s, z_e_shuffled], dim=-1)
        t_marginal = self.network(marginal)  # (B, 1)

        # MINE lower bound on MI
        mi_estimate = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)

        return mi_estimate
