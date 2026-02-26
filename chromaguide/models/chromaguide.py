"""ChromaGuide: Full on-target efficacy prediction model.

Architecture:
    Sequence Encoder (CNN-GRU / DNABERT-2 / Caduceus / Evo)
        → z_s ∈ ℝ^64
    Epigenomic Encoder (MLP or CNN-1D)
        → z_e ∈ ℝ^64
    Gated Attention Fusion
        → z ∈ ℝ^128
    Beta Regression Head
        → (μ, φ) parameterizing Beta distribution
    Split Conformal Prediction
        → calibrated prediction intervals
"""
from __future__ import annotations
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional

from chromaguide.modules.sequence_encoders import build_sequence_encoder
from chromaguide.modules.epigenomic_encoder import EpigenomicEncoder, build_epigenomic_encoder
from chromaguide.modules.fusion import build_fusion
from chromaguide.modules.prediction_head import BetaRegressionHead


class ChromaGuideModel(nn.Module):
    """Full ChromaGuide on-target efficacy model.
    
    Forward pass:
        (one_hot_seq, epigenomic_signals) → (μ, φ)
    
    With modality dropout (training only):
        - With probability p, mask epigenomic input with zeros
        - Forces sequence encoder to carry primary predictive signal
        - Epigenomic features provide additive boost
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
        # Build sub-modules from config
        self.seq_encoder = build_sequence_encoder(cfg.model.sequence_encoder)
        
        self.epi_encoder = EpigenomicEncoder(
            n_tracks=len(cfg.data.epigenomic.tracks),
            n_bins=cfg.data.epigenomic.n_bins,
            encoder_type=cfg.model.epigenomic_encoder.type,
            hidden_dims=list(cfg.model.epigenomic_encoder.hidden_dims),
            output_dim=cfg.model.epigenomic_encoder.output_dim,
            dropout=cfg.model.epigenomic_encoder.dropout,
            activation=cfg.model.epigenomic_encoder.activation,
        )
        
        self.fusion = build_fusion(cfg.model.fusion)
        
        self.prediction_head = BetaRegressionHead(
            input_dim=cfg.model.prediction_head.input_dim,
            epsilon=cfg.model.prediction_head.epsilon,
            phi_min=cfg.model.prediction_head.phi_min,
            phi_max=cfg.model.prediction_head.phi_max,
        )
        
        # Modality dropout
        self.modality_dropout_enabled = cfg.model.modality_dropout.enabled
        self.modality_dropout_prob = cfg.model.modality_dropout.prob
        
        # Store raw sequences flag (for transformer-based encoders)
        self._needs_raw_sequences = cfg.model.sequence_encoder.type in ["dnabert2", "evo", "nucleotide_transformer"]
    
    def forward(
        self,
        seq: torch.Tensor,
        epi: torch.Tensor,
        raw_sequences: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            seq: One-hot encoded sgRNA sequences (batch, 4, seq_len).
            epi: Epigenomic signals (batch, n_tracks, n_bins).
            raw_sequences: Raw DNA strings (required for DNABERT-2 / Evo).
        
        Returns:
            mu: Predicted mean efficacy (batch, 1), in (0, 1).
            phi: Predicted precision (batch, 1), > phi_min.
        """
        # Sequence encoding
        if self._needs_raw_sequences and raw_sequences is not None:
            z_s = self.seq_encoder(seq, sequences=raw_sequences)
        else:
            z_s = self.seq_encoder(seq)
        
        # Epigenomic encoding
        z_e = self.epi_encoder(epi)
        
        # Modality dropout (training only)
        if self.training and self.modality_dropout_enabled:
            mask = torch.rand(z_e.shape[0], 1, device=z_e.device) > self.modality_dropout_prob
            z_e = z_e * mask.float()
        
        # Fusion
        z = self.fusion(z_s, z_e)
        
        # Prediction
        mu, phi = self.prediction_head(z)
        
        return mu, phi
    
    def predict_with_uncertainty(
        self,
        seq: torch.Tensor,
        epi: torch.Tensor,
        raw_sequences: list[str] | None = None,
    ) -> dict:
        """Predict with full uncertainty information.
        
        Returns dict with:
            mu: Point prediction
            phi: Beta precision
            variance: Predictive variance from Beta
            std: Predictive standard deviation
        """
        self.eval()
        with torch.no_grad():
            mu, phi = self.forward(seq, epi, raw_sequences)
            var = self.prediction_head.variance(mu, phi)
        
        return {
            "mu": mu,
            "phi": phi,
            "variance": var,
            "std": var.sqrt(),
        }
    
    def get_gate_values(self, seq: torch.Tensor, epi: torch.Tensor) -> torch.Tensor:
        """Extract fusion gate values for interpretability."""
        with torch.no_grad():
            if self._needs_raw_sequences:
                z_s = self.seq_encoder(seq)  # will fail for transformer; needs sequences
            else:
                z_s = self.seq_encoder(seq)
            z_e = self.epi_encoder(epi)
            
            if hasattr(self.fusion, 'get_gate_values'):
                return self.fusion.get_gate_values(z_s, z_e)
            else:
                return None
    
    @property
    def encoder_type(self) -> str:
        return self.cfg.model.sequence_encoder.type


class ChromaGuideEnsemble(nn.Module):
    """Ensemble of ChromaGuide models for improved prediction.
    
    Trains K models with different seeds and averages predictions.
    Used for final thesis results.
    """
    
    def __init__(self, models: list[ChromaGuideModel]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(
        self, seq: torch.Tensor, epi: torch.Tensor,
        raw_sequences: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mus, phis = [], []
        for model in self.models:
            mu, phi = model(seq, epi, raw_sequences)
            mus.append(mu)
            phis.append(phi)
        
        # Average predictions
        mu_avg = torch.stack(mus).mean(dim=0)
        phi_avg = torch.stack(phis).mean(dim=0)
        
        return mu_avg, phi_avg
