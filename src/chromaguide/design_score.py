"""Integrated sgRNA design score for ChromaGuide.

Component 3: Combines calibrated on-target prediction with aggregated
off-target risk and uncertainty to produce a single, actionable sgRNA design score.

Design score S = w_e * mu - w_r * R - w_u * sigma_CI

where:
  - mu: predicted on-target efficacy (mean of Beta regression head)
  - R: aggregated off-target risk
  - sigma_CI: width of the conformal prediction interval (uncertainty)
  - w_e, w_r, w_u: adjustable weights (default 1.0, 0.5, 0.2)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class GuideCandidate:
    """Represents a candidate sgRNA with all scores."""
    guide_seq: str
    target_gene: str
    chrom: str
    position: int
    strand: str
    on_target_score: float
    on_target_ci_lower: float
    on_target_ci_upper: float
    on_target_ci_width: float
    off_target_risk: float
    n_off_target_sites: int
    design_score: float
    rank: int = 0


class IntegratedDesignScore(nn.Module):
    """Computes the integrated sgRNA design score.

    S(g) = w_e * hat_mu(g) - w_r * R(g) - w_u * sigma_CI(g)

    The weights can be user-specified or fixed.
    """
    def __init__(
        self,
        w_e: float = 1.0,
        w_r: float = 0.5,
        w_u: float = 0.2,
        learnable_weights: bool = False,
    ):
        super().__init__()

        if learnable_weights:
            self.w_e = nn.Parameter(torch.tensor(w_e))
            self.w_r = nn.Parameter(torch.tensor(w_r))
            self.w_u = nn.Parameter(torch.tensor(w_u))
        else:
            self.register_buffer('w_e', torch.tensor(w_e))
            self.register_buffer('w_r', torch.tensor(w_r))
            self.register_buffer('w_u', torch.tensor(w_u))

    def forward(
        self,
        on_target_score: torch.Tensor,
        off_target_risk: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """Compute integrated design score.

        Args:
            on_target_score: Predicted on-target efficacy (batch, 1)
            off_target_risk: Aggregated off-target risk (batch, 1)
            uncertainty: Conformal prediction interval width (batch, 1)

        Returns:
            Design score (batch, 1)
        """
        # S = w_e * mu - w_r * R - w_u * sigma
        score = self.w_e * on_target_score - self.w_r * off_target_risk - self.w_u * uncertainty
        return score

    def rank_guides(
        self,
        candidates: List[Dict],
    ) -> List[GuideCandidate]:
        """Rank a list of guide candidates by design score.

        Args:
            candidates: List of dicts with keys:
              - guide_seq, target_gene, chrom, position, strand
              - on_target_score, on_target_ci_lower, on_target_ci_upper
              - off_target_risk, n_off_target_sites

        Returns:
            Sorted list of GuideCandidate objects
        """
        scored = []
        for c in candidates:
            mu = c['on_target_score']
            r = c['off_target_risk']
            lower = c.get('on_target_ci_lower', 0.0)
            upper = c.get('on_target_ci_upper', 1.0)
            sigma = upper - lower

            on_t = torch.tensor([mu])
            off_t = torch.tensor([r])
            unc_t = torch.tensor([sigma])

            with torch.no_grad():
                ds = self.forward(on_t, off_t, unc_t).item()

            scored.append(GuideCandidate(
                guide_seq=c['guide_seq'],
                target_gene=c['target_gene'],
                chrom=c['chrom'],
                position=c['position'],
                strand=c['strand'],
                on_target_score=mu,
                on_target_ci_lower=lower,
                on_target_ci_upper=upper,
                on_target_ci_width=sigma,
                off_target_risk=r,
                n_off_target_sites=c.get('n_off_target_sites', 0),
                design_score=ds,
            ))

        # Sort by design score (descending)
        scored.sort(key=lambda x: x.design_score, reverse=True)

        # Assign ranks
        for i, g in enumerate(scored):
            g.rank = i + 1

        return scored
