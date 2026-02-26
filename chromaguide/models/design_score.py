"""Integrated sgRNA Design Score.

Module 3 of ChromaGuide: combines on-target efficacy, off-target risk,
and prediction uncertainty into a single actionable design score.

D(g) = w_e · E(g) − w_r · R(g) − w_u · U(g)

where:
    E(g) = normalized predicted efficacy (higher = better)
    R(g) = normalized off-target risk (lower = better)
    U(g) = normalized prediction uncertainty (lower = better)
    w_e, w_r, w_u = user-tunable weights (default: 1.0, 0.5, 0.2)
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Optional
from omegaconf import DictConfig


class DesignScorer:
    """Integrated sgRNA design scoring module.
    
    Combines three ChromaGuide components into a single ranking score.
    """
    
    def __init__(
        self,
        w_efficacy: float = 1.0,
        w_risk: float = 0.5,
        w_uncertainty: float = 0.2,
        normalize: str = "minmax",
    ):
        self.w_e = w_efficacy
        self.w_r = w_risk
        self.w_u = w_uncertainty
        self.normalize = normalize
    
    def score(
        self,
        efficacy: np.ndarray,
        risk: np.ndarray,
        uncertainty: np.ndarray,
    ) -> np.ndarray:
        """Compute design score for a batch of sgRNAs.
        
        Args:
            efficacy: Predicted on-target efficacy (n,), in (0, 1).
            risk: Off-target risk score (n,), in (0, 1).
            uncertainty: Prediction interval width (n,), or Beta std.
        
        Returns:
            Design score (n,), higher = better sgRNA.
        """
        # Normalize components to [0, 1]
        e = self._normalize(efficacy)
        r = self._normalize(risk)
        u = self._normalize(uncertainty)
        
        # Composite score (higher = better)
        score = self.w_e * e - self.w_r * r - self.w_u * u
        
        # Rescale to [0, 1]
        score = self._normalize(score)
        
        return score
    
    def rank(
        self,
        efficacy: np.ndarray,
        risk: np.ndarray,
        uncertainty: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Rank sgRNAs by design score.
        
        Args:
            top_k: If set, return only the top-K indices.
        
        Returns:
            Indices sorted by descending design score.
        """
        scores = self.score(efficacy, risk, uncertainty)
        ranking = np.argsort(scores)[::-1]
        
        if top_k is not None:
            ranking = ranking[:top_k]
        
        return ranking
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "minmax":
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min)
            return np.full_like(x, 0.5)
        elif self.normalize == "zscore":
            mu, sigma = x.mean(), x.std()
            if sigma > 0:
                return (x - mu) / sigma
            return np.zeros_like(x)
        return x
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "DesignScorer":
        return cls(
            w_efficacy=cfg.design_score.weights.efficacy,
            w_risk=cfg.design_score.weights.risk,
            w_uncertainty=cfg.design_score.weights.uncertainty,
            normalize=cfg.design_score.normalize,
        )
