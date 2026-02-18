"""Integrated sgRNA design score for ChromaGuide.

Component 3: Combines calibrated on-target prediction with aggregated
off-target risk to produce a single, actionable sgRNA design score.

Design score = w_on * on_target_efficacy - w_off * off_target_risk

The weights can be user-specified or learned.
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
    off_target_risk: float
    n_off_target_sites: int
    design_score: float
    rank: int = 0


class IntegratedDesignScore(nn.Module):
    """Computes the integrated sgRNA design score.

    S(g) = sigma(w_on * hat_y(g) - w_off * R_OT(g))

    where:
      - hat_y(g) is the calibrated on-target efficacy prediction
      - R_OT(g) is the aggregated off-target risk
      - w_on, w_off are learnable or fixed trade-off weights
      - sigma is sigmoid to bound the score in [0, 1]
    """
    def __init__(
        self,
        w_on: float = 1.0,
        w_off: float = 1.0,
        learnable_weights: bool = True,
    ):
        super().__init__()

        if learnable_weights:
            self.w_on = nn.Parameter(torch.tensor(w_on))
            self.w_off = nn.Parameter(torch.tensor(w_off))
        else:
            self.register_buffer('w_on', torch.tensor(w_on))
            self.register_buffer('w_off', torch.tensor(w_off))

    def forward(
        self,
        on_target_score: torch.Tensor,
        off_target_risk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute integrated design score.

        Args:
            on_target_score: Predicted on-target efficacy (batch, 1)
            off_target_risk: Aggregated off-target risk (batch, 1)

        Returns:
            Design score (batch, 1) in [0, 1]
        """
        raw_score = self.w_on * on_target_score - self.w_off * off_target_risk
        return torch.sigmoid(raw_score)

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
            on_t = torch.tensor([c['on_target_score']])
            off_t = torch.tensor([c['off_target_risk']])

            with torch.no_grad():
                ds = self.forward(on_t, off_t).item()

            scored.append(GuideCandidate(
                guide_seq=c['guide_seq'],
                target_gene=c['target_gene'],
                chrom=c['chrom'],
                position=c['position'],
                strand=c['strand'],
                on_target_score=c['on_target_score'],
                on_target_ci_lower=c.get('on_target_ci_lower', 0.0),
                on_target_ci_upper=c.get('on_target_ci_upper', 1.0),
                off_target_risk=c['off_target_risk'],
                n_off_target_sites=c.get('n_off_target_sites', 0),
                design_score=ds,
            ))

        # Sort by design score (descending)
        scored.sort(key=lambda x: x.design_score, reverse=True)

        # Assign ranks
        for i, g in enumerate(scored):
            g.rank = i + 1

        return scored
