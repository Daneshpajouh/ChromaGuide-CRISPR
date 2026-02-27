"""Enhanced loss functions for ChromaGuide v3.

Key improvements over v2:
    1. Log-Cosh loss (from CRISPR-FMC SOTA — more robust than MSE)
    2. Pairwise ranking loss (directly optimizes Spearman correlation)
    3. Combined loss with adaptive weighting
    4. Temperature-scaled calibration loss
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import math


class LogCoshLoss(nn.Module):
    """Log-Cosh loss function.
    
    L = (1/N) Σ log(cosh(ŷ_i - y_i))
    
    Used by CRISPR-FMC (current SOTA). Properties:
    - Robust to outliers (like MAE)
    - Differentiable everywhere (like MSE)
    - Approximates MSE for small errors, MAE for large errors
    
    Reference: Li et al. (2025) "CRISPR-FMC" Frontiers in Genome Editing
    """
    
    def forward(self, mu: torch.Tensor, phi: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.view_as(mu)
        diff = mu - target
        # Numerically stable log-cosh
        loss = diff + F.softplus(-2.0 * diff) - math.log(2.0)
        return loss.mean()


class PairwiseRankingLoss(nn.Module):
    """Differentiable pairwise ranking loss.
    
    Directly optimizes rank correlation (aligned with Spearman evaluation).
    For each pair (i, j), penalizes when the model ranks them incorrectly.
    
    L = (1/P) Σ_{i<j} max(0, -sign(y_i - y_j) * (ŷ_i - ŷ_j) + margin)
    
    With smooth approximation using sigmoid.
    """
    
    def __init__(self, margin: float = 0.05, n_pairs: int = 512, temperature: float = 1.0):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs  # Sample pairs for efficiency
        self.temperature = temperature
    
    def forward(self, mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = mu.view(-1)
        target = target.view(-1)
        n = mu.shape[0]
        
        if n < 2:
            return torch.tensor(0.0, device=mu.device)
        
        # Sample random pairs for efficiency
        n_pairs = min(self.n_pairs, n * (n - 1) // 2)
        idx_i = torch.randint(0, n, (n_pairs,), device=mu.device)
        idx_j = torch.randint(0, n, (n_pairs,), device=mu.device)
        # Ensure i != j
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        
        if len(idx_i) == 0:
            return torch.tensor(0.0, device=mu.device)
        
        # Compute pairwise differences
        target_diff = target[idx_i] - target[idx_j]
        pred_diff = mu[idx_i] - mu[idx_j]
        
        # Smooth sign using sigmoid
        target_sign = torch.sign(target_diff)
        
        # Hinge-like ranking loss with sigmoid smoothing
        rank_loss = torch.clamp(self.margin - target_sign * pred_diff, min=0.0)
        
        return rank_loss.mean()


class ListwiseRankingLoss(nn.Module):
    """ListNet-style listwise ranking loss.
    
    Uses softmax over predictions and targets to compute
    cross-entropy between rank distributions.
    More directly aligned with Spearman/NDCG than pairwise.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = mu.view(-1)
        target = target.view(-1)
        
        # Convert to rank distributions via softmax
        pred_dist = F.softmax(mu / self.temperature, dim=0)
        target_dist = F.softmax(target / self.temperature, dim=0)
        
        # Cross-entropy between distributions
        loss = -(target_dist * torch.log(pred_dist + 1e-8)).sum()
        
        return loss


class BetaNLL(nn.Module):
    """Beta negative log-likelihood loss.
    
    L_beta = -E[log Beta(y; α, β)]
    where α = μφ, β = (1-μ)φ
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, mu: torch.Tensor, phi: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.view_as(mu).clamp(self.epsilon, 1.0 - self.epsilon)
        mu = mu.clamp(self.epsilon, 1.0 - self.epsilon)
        
        alpha = mu * phi
        beta = (1 - mu) * phi
        
        alpha = alpha.clamp(min=self.epsilon)
        beta = beta.clamp(min=self.epsilon)
        
        dist = Beta(alpha, beta)
        nll = -dist.log_prob(target)
        
        return nll.mean()


class CalibrationLoss(nn.Module):
    """Differentiable calibration penalty with temperature scaling."""
    
    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins
    
    def forward(self, mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = mu.view(-1)
        target = target.view(-1)
        
        bin_centers = torch.linspace(0, 1, self.n_bins, device=mu.device)
        bandwidth = 1.0 / self.n_bins
        
        distances = (mu.unsqueeze(1) - bin_centers.unsqueeze(0)) / bandwidth
        weights = torch.exp(-0.5 * distances ** 2)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        bin_pred = (weights * mu.unsqueeze(1)).sum(dim=0) / (weights.sum(dim=0) + 1e-8)
        bin_true = (weights * target.unsqueeze(1)).sum(dim=0) / (weights.sum(dim=0) + 1e-8)
        
        cal_error = ((bin_pred - bin_true) ** 2).mean()
        return cal_error


class NonRedundancyLoss(nn.Module):
    """Non-redundancy regularizer between modalities."""
    
    def forward(self, z_s: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)
        z_e_centered = z_e - z_e.mean(dim=0, keepdim=True)
        
        numerator = (z_s_centered * z_e_centered).sum(dim=0)
        denominator = z_s_centered.norm(dim=0) * z_e_centered.norm(dim=0) + 1e-8
        corr = numerator / denominator
        
        return corr.abs().mean()


class CalibratedLossV3(nn.Module):
    """Enhanced combined loss for ChromaGuide v3.
    
    L = L_logcosh + λ_rank * L_rank + λ_cal * L_cal + λ_nr * L_NR
    
    Key innovations:
    - Log-Cosh as primary loss (robust, from CRISPR-FMC SOTA)
    - Pairwise ranking loss to optimize rank correlation
    - Adaptive calibration penalty
    """
    
    def __init__(
        self,
        primary_type: str = "logcosh",  # logcosh, beta_nll, mse
        lambda_rank: float = 0.1,  # Ranking loss weight
        lambda_cal: float = 0.05,   # Calibration loss weight
        lambda_nr: float = 0.01,    # Non-redundancy weight
        use_nr: bool = False,
        rank_pairs: int = 512,
    ):
        super().__init__()
        
        if primary_type == "logcosh":
            self.primary = LogCoshLoss()
        elif primary_type == "beta_nll":
            self.primary = BetaNLL()
        else:
            self.primary = None  # Use simple MSE
        
        self.ranking = PairwiseRankingLoss(n_pairs=rank_pairs)
        self.calibration = CalibrationLoss()
        self.non_redundancy = NonRedundancyLoss() if use_nr else None
        
        self.lambda_rank = lambda_rank
        self.lambda_cal = lambda_cal
        self.lambda_nr = lambda_nr
    
    def forward(
        self,
        mu: torch.Tensor,
        phi: torch.Tensor,
        target: torch.Tensor,
        z_s: torch.Tensor | None = None,
        z_e: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        
        # Primary loss
        if self.primary is not None:
            losses["primary"] = self.primary(mu, phi, target)
        else:
            losses["primary"] = F.mse_loss(mu.view(-1), target.view(-1))
        
        # Ranking loss
        losses["ranking"] = self.ranking(mu, target) * self.lambda_rank
        
        # Calibration loss
        losses["calibration"] = self.calibration(mu, target) * self.lambda_cal
        
        # Non-redundancy loss
        if self.non_redundancy is not None and z_s is not None and z_e is not None:
            losses["non_redundancy"] = self.non_redundancy(z_s, z_e) * self.lambda_nr
        else:
            losses["non_redundancy"] = torch.tensor(0.0, device=mu.device)
        
        # Total
        losses["total"] = losses["primary"] + losses["ranking"] + losses["calibration"] + losses["non_redundancy"]
        
        return losses
