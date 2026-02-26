"""Loss functions for ChromaGuide.

Primary loss: L = L_MSE + λ_cal * L_cal + λ_nr * L_NR

Components:
    L_MSE: Mean squared error between μ̂ and y
    L_Beta: Negative log-likelihood of Beta(y; μφ, (1-μ)φ) [alternative to MSE]
    L_cal: Calibration penalty (ECE-inspired differentiable proxy)
    L_NR: Non-redundancy regularizer (optional, via MINE/CLUB)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class BetaNLL(nn.Module):
    """Beta negative log-likelihood loss.
    
    L_beta = -E[log Beta(y; α, β)]
    where α = μφ, β = (1-μ)φ
    
    This is the proper loss for Beta regression. It naturally
    handles heteroscedastic uncertainty through the precision φ.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        mu: torch.Tensor,
        phi: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean (batch, 1), in (0, 1).
            phi: Predicted precision (batch, 1), > 0.
            target: True efficacy (batch,) or (batch, 1), in (0, 1).
        
        Returns:
            Scalar loss (mean over batch).
        """
        target = target.view_as(mu).clamp(self.epsilon, 1.0 - self.epsilon)
        mu = mu.clamp(self.epsilon, 1.0 - self.epsilon)
        
        alpha = mu * phi
        beta = (1 - mu) * phi
        
        # Ensure valid distribution parameters
        alpha = alpha.clamp(min=self.epsilon)
        beta = beta.clamp(min=self.epsilon)
        
        dist = Beta(alpha, beta)
        nll = -dist.log_prob(target)
        
        return nll.mean()


class MSEWithUncertainty(nn.Module):
    """MSE loss weighted by predicted precision.
    
    L = (1/2φ)(y - μ)² + (1/2)log(φ)
    
    Equivalent to Gaussian heteroscedastic loss but applied to
    the Beta regression setting. Encourages the model to:
    - Be more precise where it's confident (large φ)
    - Be less penalized where it's uncertain (small φ)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        mu: torch.Tensor,
        phi: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.view_as(mu)
        
        # Heteroscedastic MSE
        precision_weight = phi.clamp(min=1.0)
        mse = (target - mu) ** 2
        loss = 0.5 * mse * precision_weight - 0.5 * torch.log(precision_weight)
        
        return loss.mean()


class CalibrationLoss(nn.Module):
    """Differentiable calibration penalty.
    
    Approximates Expected Calibration Error (ECE) as a differentiable
    proxy loss. Groups predictions into bins and penalizes deviation
    between average predicted probability and observed frequency.
    """
    
    def __init__(self, n_bins: int = 15, penalty: str = "l2"):
        super().__init__()
        self.n_bins = n_bins
        self.penalty = penalty
    
    def forward(
        self,
        mu: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean efficacy (batch, 1).
            target: True efficacy (batch,) or (batch, 1).
        """
        mu = mu.view(-1)
        target = target.view(-1)
        
        # Soft binning using Gaussian kernel
        bin_centers = torch.linspace(0, 1, self.n_bins, device=mu.device)
        bandwidth = 1.0 / self.n_bins
        
        # Soft assignment to bins
        distances = (mu.unsqueeze(1) - bin_centers.unsqueeze(0)) / bandwidth
        weights = torch.exp(-0.5 * distances ** 2)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Per-bin average prediction and average target
        bin_pred = (weights * mu.unsqueeze(1)).sum(dim=0) / (weights.sum(dim=0) + 1e-8)
        bin_true = (weights * target.unsqueeze(1)).sum(dim=0) / (weights.sum(dim=0) + 1e-8)
        
        # Calibration error
        if self.penalty == "l2":
            cal_error = ((bin_pred - bin_true) ** 2).mean()
        else:
            cal_error = (bin_pred - bin_true).abs().mean()
        
        return cal_error


class NonRedundancyLoss(nn.Module):
    """Non-redundancy regularizer between modalities.
    
    Encourages z_s and z_e to carry complementary (non-redundant)
    information. Uses a simplified mutual information upper bound.
    
    L_NR = |corr(z_s, z_e)|  (simplified; MINE/CLUB for advanced)
    """
    
    def __init__(self, method: str = "correlation"):
        super().__init__()
        self.method = method
    
    def forward(self, z_s: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_s: Sequence representation (batch, d).
            z_e: Epigenomic representation (batch, d).
        """
        if self.method == "correlation":
            # Average absolute correlation across dimensions
            z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)
            z_e_centered = z_e - z_e.mean(dim=0, keepdim=True)
            
            numerator = (z_s_centered * z_e_centered).sum(dim=0)
            denominator = (
                z_s_centered.norm(dim=0) * z_e_centered.norm(dim=0) + 1e-8
            )
            corr = numerator / denominator
            
            return corr.abs().mean()
        
        elif self.method == "hsic":
            # Hilbert-Schmidt Independence Criterion
            n = z_s.shape[0]
            K = torch.mm(z_s, z_s.t())
            L = torch.mm(z_e, z_e.t())
            H = torch.eye(n, device=z_s.device) - 1.0 / n
            
            hsic = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
            return hsic
        
        return torch.tensor(0.0, device=z_s.device)


class CalibratedLoss(nn.Module):
    """Combined loss for ChromaGuide training.
    
    L = L_primary + λ_cal * L_cal + λ_nr * L_NR
    
    where L_primary is MSE or Beta NLL.
    """
    
    def __init__(
        self,
        primary_type: str = "mse",
        lambda_cal: float = 0.1,
        lambda_nr: float = 0.01,
        use_nr: bool = False,
    ):
        super().__init__()
        
        if primary_type == "beta_nll":
            self.primary = BetaNLL()
        elif primary_type == "mse_uncertainty":
            self.primary = MSEWithUncertainty()
        else:
            self.primary = None  # Use simple MSE
        
        self.calibration = CalibrationLoss()
        self.non_redundancy = NonRedundancyLoss() if use_nr else None
        
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
        """
        Returns dict with individual loss components and total.
        """
        losses = {}
        
        # Primary loss
        if self.primary is not None:
            losses["primary"] = self.primary(mu, phi, target)
        else:
            losses["primary"] = F.mse_loss(mu.view(-1), target.view(-1))
        
        # Calibration loss
        losses["calibration"] = self.calibration(mu, target) * self.lambda_cal
        
        # Non-redundancy loss
        if self.non_redundancy is not None and z_s is not None and z_e is not None:
            losses["non_redundancy"] = self.non_redundancy(z_s, z_e) * self.lambda_nr
        else:
            losses["non_redundancy"] = torch.tensor(0.0, device=mu.device)
        
        # Total
        losses["total"] = losses["primary"] + losses["calibration"] + losses["non_redundancy"]
        
        return losses
