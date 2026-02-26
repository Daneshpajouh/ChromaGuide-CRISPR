"""Multi-modal fusion modules.

Combines sequence representation z_s ∈ ℝ^d and epigenomic representation
z_e ∈ ℝ^d into a fused representation z ∈ ℝ^(2d).

Implements:
  - GatedAttentionFusion (primary; from proposal Eq. 1)
  - ConcatMLPFusion (ablation baseline)
  - CrossAttentionFusion (ablation)
  - MoEFusion (ablation; mixture-of-experts)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedAttentionFusion(nn.Module):
    """Gated attention fusion (primary method).
    
    From the proposal:
        g = σ(W_g [z_s; z_e] + b_g)     (learned gate)
        z = [g ⊙ z_s; (1-g) ⊙ z_e]      (gated concatenation)
    
    The gate learns which modality is more informative per-sample.
    Modality dropout can mask z_e during training for robustness.
    """
    
    def __init__(
        self,
        seq_dim: int = 64,
        epi_dim: int = 64,
        output_dim: int = 128,
        gate_activation: str = "sigmoid",
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        input_dim = seq_dim + epi_dim
        
        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, seq_dim),
        )
        # Initialize gate bias to control initial modality preference
        if gate_bias_init != 0.0:
            nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)
        
        self.gate_act = nn.Sigmoid() if gate_activation == "sigmoid" else nn.Tanh()
        
        # Optional projection if dims don't match
        self.needs_projection = (seq_dim + epi_dim) != output_dim
        if self.needs_projection:
            self.projection = nn.Sequential(
                nn.Linear(seq_dim + epi_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        
        self.output_dim = output_dim
    
    def forward(
        self, z_s: torch.Tensor, z_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_s: Sequence encoding (batch, seq_dim).
            z_e: Epigenomic encoding (batch, epi_dim).
        
        Returns:
            Fused representation z (batch, output_dim).
        """
        combined = torch.cat([z_s, z_e], dim=-1)  # (batch, seq_dim + epi_dim)
        
        # Compute gate
        g = self.gate_act(self.gate_net(combined))  # (batch, seq_dim)
        
        # Gated fusion
        z = torch.cat([g * z_s, (1 - g) * z_e], dim=-1)  # (batch, seq_dim + epi_dim)
        
        if self.needs_projection:
            z = self.projection(z)
        
        return z
    
    def get_gate_values(self, z_s: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        """Return gate values for interpretability analysis."""
        combined = torch.cat([z_s, z_e], dim=-1)
        return self.gate_act(self.gate_net(combined))


class ConcatMLPFusion(nn.Module):
    """Simple concatenation + MLP fusion (ablation baseline)."""
    
    def __init__(
        self,
        seq_dim: int = 64,
        epi_dim: int = 64,
        output_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(seq_dim + epi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, z_s: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z_s, z_e], dim=-1))


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion (ablation).
    
    Sequence attends to epigenomic features and vice versa,
    then concatenates the cross-attended representations.
    """
    
    def __init__(
        self,
        seq_dim: int = 64,
        epi_dim: int = 64,
        output_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # Seq → Epi attention
        self.seq_to_epi = nn.MultiheadAttention(
            embed_dim=seq_dim, num_heads=n_heads,
            kdim=epi_dim, vdim=epi_dim,
            dropout=dropout, batch_first=True,
        )
        # Epi → Seq attention
        self.epi_to_seq = nn.MultiheadAttention(
            embed_dim=epi_dim, num_heads=n_heads,
            kdim=seq_dim, vdim=seq_dim,
            dropout=dropout, batch_first=True,
        )
        
        self.norm_s = nn.LayerNorm(seq_dim)
        self.norm_e = nn.LayerNorm(epi_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(seq_dim + epi_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, z_s: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension for attention
        z_s_3d = z_s.unsqueeze(1)  # (batch, 1, seq_dim)
        z_e_3d = z_e.unsqueeze(1)  # (batch, 1, epi_dim)
        
        # Cross-attention
        s2e, _ = self.seq_to_epi(z_s_3d, z_e_3d, z_e_3d)  # seq queries epi
        e2s, _ = self.epi_to_seq(z_e_3d, z_s_3d, z_s_3d)  # epi queries seq
        
        s_out = self.norm_s(z_s + s2e.squeeze(1))
        e_out = self.norm_e(z_e + e2s.squeeze(1))
        
        return self.projection(torch.cat([s_out, e_out], dim=-1))


class MoEFusion(nn.Module):
    """Mixture-of-Experts fusion (ablation).
    
    Routes each sample through K expert networks, weighted by a
    learned router. Encourages specialization per modality combination.
    """
    
    def __init__(
        self,
        seq_dim: int = 64,
        epi_dim: int = 64,
        output_dim: int = 128,
        n_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        input_dim = seq_dim + epi_dim
        
        # Router
        self.router = nn.Linear(input_dim, n_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            for _ in range(n_experts)
        ])
    
    def forward(self, z_s: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z_s, z_e], dim=-1)
        
        # Router weights
        weights = F.softmax(self.router(combined), dim=-1)  # (batch, n_experts)
        
        # Expert outputs
        expert_outs = torch.stack([expert(combined) for expert in self.experts], dim=1)  # (batch, n_experts, output_dim)
        
        # Weighted combination
        z = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (batch, output_dim)
        
        return z


def build_fusion(cfg) -> nn.Module:
    """Factory function to build fusion module from config."""
    fusion_type = cfg.type.lower()
    seq_dim = cfg.input_dim // 2
    epi_dim = cfg.input_dim // 2
    
    if fusion_type == "gated_attention":
        return GatedAttentionFusion(
            seq_dim=seq_dim, epi_dim=epi_dim,
            output_dim=cfg.output_dim,
        )
    elif fusion_type == "concat_mlp":
        return ConcatMLPFusion(
            seq_dim=seq_dim, epi_dim=epi_dim,
            output_dim=cfg.output_dim,
        )
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion(
            seq_dim=seq_dim, epi_dim=epi_dim,
            output_dim=cfg.output_dim,
        )
    elif fusion_type == "moe":
        return MoEFusion(
            seq_dim=seq_dim, epi_dim=epi_dim,
            output_dim=cfg.output_dim,
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
