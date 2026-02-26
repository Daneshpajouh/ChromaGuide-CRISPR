"""Beta regression prediction head.

Maps fused representation z → (μ, φ) for a Beta distribution,
where μ ∈ (0, 1) is the predicted mean efficacy and φ > 2 is the
precision (concentration) parameter.

Loss: L_beta = -log Beta(y; α, β)
where α = μφ, β = (1-μ)φ

This naturally handles bounded [0, 1] regression with
heteroscedastic uncertainty estimates via Var[Y] = μ(1-μ)/(1+φ).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class BetaRegressionHead(nn.Module):
    """Beta regression head for bounded efficacy prediction.
    
    Architecture:
        z ∈ ℝ^d → shared MLP → split into μ-branch and φ-branch
        μ = sigmoid(linear(h))           ∈ (ε, 1-ε)
        φ = softplus(linear(h)) + φ_min  ∈ (φ_min, φ_max)
    
    Outputs (μ, φ) parameterizing Beta(αφ, (1-μ)φ).
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        epsilon: float = 1e-6,
        phi_min: float = 2.0,
        phi_max: float = 1000.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.phi_min = phi_min
        self.phi_max = phi_max
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Mean branch (μ)
        self.mu_head = nn.Linear(hidden_dim, 1)
        
        # Precision branch (φ)
        self.phi_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Fused representation (batch, input_dim).
        
        Returns:
            mu: Predicted mean efficacy (batch, 1), in (ε, 1-ε).
            phi: Predicted precision (batch, 1), in (φ_min, φ_max).
        """
        h = self.shared(z)
        
        # Mean: sigmoid → clamp to (ε, 1-ε) for numerical stability
        mu = torch.sigmoid(self.mu_head(h))
        mu = mu.clamp(self.epsilon, 1.0 - self.epsilon)
        
        # Precision: softplus → clamp to (φ_min, φ_max)
        phi = F.softplus(self.phi_head(h)) + self.phi_min
        phi = phi.clamp(max=self.phi_max)
        
        return mu, phi
    
    def sample(self, mu: torch.Tensor, phi: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample from the predicted Beta distribution.
        
        Args:
            mu: Mean parameter (batch, 1).
            phi: Precision parameter (batch, 1).
            n_samples: Number of samples per prediction.
        
        Returns:
            Samples of shape (n_samples, batch, 1).
        """
        alpha = mu * phi
        beta = (1 - mu) * phi
        dist = Beta(alpha, beta)
        return dist.rsample((n_samples,))
    
    def variance(self, mu: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Compute predictive variance: Var[Y] = μ(1-μ) / (1+φ)."""
        return (mu * (1 - mu)) / (1 + phi)
    
    def log_prob(self, y: torch.Tensor, mu: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Compute log probability under the Beta distribution.
        
        Args:
            y: Target values (batch, 1), in (0, 1).
            mu: Mean parameter (batch, 1).
            phi: Precision parameter (batch, 1).
        
        Returns:
            Log probability (batch, 1).
        """
        y = y.clamp(self.epsilon, 1.0 - self.epsilon)
        alpha = mu * phi
        beta = (1 - mu) * phi
        dist = Beta(alpha, beta)
        return dist.log_prob(y)
