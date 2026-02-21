"""Fusion module for ChromaGuide.

Combines sequence embedding z_s and epigenomic embedding z_e through
gated attention fusion as specified in the proposal methodology.
Optionally includes an information-theoretic non-redundancy regularizer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedAttentionFusion(nn.Module):
    """Gated attention fusion as specified in proposal methodology.

    Implementation of:
    g = sigmoid(W_g * [h_seq; h_epi] + b_g)
    fused = g * h_seq + (1-g) * h_epi

    Where g is the gating mechanism that learns how much to weight
    sequence vs epigenomic features for each position.
    """
    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model

        # Gate network: projects concatenated features to gate weights
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

        # Optional projection layer for final output
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        h_seq: torch.Tensor,
        h_epi: torch.Tensor,
    ) -> torch.Tensor:
        """Gated attention fusion.

        Args:
            h_seq: Sequence embedding (batch, d_model)
            h_epi: Epigenomic embedding (batch, d_model)

        Returns:
            Fused embedding (batch, d_model)
        """
        # Concatenate sequence and epigenomic features
        h_concat = torch.cat([h_seq, h_epi], dim=-1)  # (batch, 2*d_model)

        # Compute gate weights
        g = self.gate_network(h_concat)  # (batch, d_model)

        # Apply gated fusion
        fused = g * h_seq + (1 - g) * h_epi

        # Optional output projection
        return self.output_proj(fused)


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
            return self.proj(h)
        else:
            return self.proj(h)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for modality interaction.

    Learns interactions between sequence and epigenomic features
    using a multi-head attention mechanism where sequence acts as
    query and epigenomics acts as key/value.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        # Multi-head attention (B, 1, D)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        z_s: torch.Tensor,
        z_e: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention fusion.

        Args:
            z_s: Sequence embedding (batch, d_model)
            z_e: Epigenomic embedding (batch, d_model)

        Returns:
            Fused embedding (batch, d_model)
        """
        # (batch, 1, d_model)
        q = z_s.unsqueeze(1)
        k = v = z_e.unsqueeze(1)

        # Cross-attention: seq attends to epi
        attn_out, _ = self.cross_attn(q, k, v)
        attn_out = attn_out.squeeze(1)

        # Residual + Projection
        out = torch.cat([z_s + attn_out, z_e], dim=-1)
        return self.proj(out)


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
