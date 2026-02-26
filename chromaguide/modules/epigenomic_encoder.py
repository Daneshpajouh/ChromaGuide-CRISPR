"""Epigenomic signal encoder.

Encodes binned chromatin accessibility / histone modification signals
around the sgRNA cut site into a fixed-length representation z_e ∈ ℝ^64.

Inputs:
    DNase-seq / ATAC-seq accessibility (1 track)
    H3K4me3 (1 track)
    H3K27ac (1 track)
    → Binned into n_bins=100 per track, window ±1kb around cut site
    → Input shape: (batch, 3, 100) or flattened (batch, 300)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class EpigenomicEncoder(nn.Module):
    """Encoder for epigenomic context around sgRNA cut site.
    
    Supports two architectures:
        - 'mlp': Flatten + MLP (default, fast)
        - 'cnn_1d': 1D CNN over the binned signal, then pool + MLP
    """
    
    def __init__(
        self,
        n_tracks: int = 3,
        n_bins: int = 100,
        encoder_type: str = "mlp",
        hidden_dims: list[int] = [256, 128],
        output_dim: int = 64,
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.n_tracks = n_tracks
        self.n_bins = n_bins
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        
        if encoder_type == "cnn_1d":
            self.cnn = nn.Sequential(
                nn.Conv1d(n_tracks, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                act_fn,
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                act_fn,
                nn.AdaptiveAvgPool1d(1),  # (batch, 64, 1)
            )
            mlp_input_dim = 64
        else:
            self.cnn = None
            mlp_input_dim = n_tracks * n_bins
        
        # MLP layers
        layers = []
        in_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Epigenomic signals.
               Shape (batch, n_tracks, n_bins) for CNN mode, or
               (batch, n_tracks * n_bins) for MLP mode.
        
        Returns:
            z_e of shape (batch, output_dim).
        """
        if self.cnn is not None:
            # Ensure 3D input for CNN
            if x.dim() == 2:
                x = x.view(x.shape[0], self.n_tracks, self.n_bins)
            x = self.cnn(x).squeeze(-1)  # (batch, 64)
        else:
            # Flatten for MLP
            if x.dim() == 3:
                x = x.view(x.shape[0], -1)  # (batch, n_tracks * n_bins)
        
        z_e = self.mlp(x)  # (batch, output_dim)
        return z_e


def build_epigenomic_encoder(cfg) -> EpigenomicEncoder:
    """Factory function to build epigenomic encoder from config."""
    return EpigenomicEncoder(
        n_tracks=len(cfg.get("tracks", ["DNase", "H3K4me3", "H3K27ac"])) if hasattr(cfg, "tracks") else 3,
        n_bins=cfg.get("n_bins", 100) if hasattr(cfg, "n_bins") else 100,
        encoder_type=cfg.type,
        hidden_dims=list(cfg.hidden_dims),
        output_dim=cfg.output_dim,
        dropout=cfg.dropout,
        activation=cfg.activation,
    )
