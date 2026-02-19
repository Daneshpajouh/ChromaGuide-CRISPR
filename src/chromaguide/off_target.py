"""Off-target prediction module for ChromaGuide.

Component 2 of the ChromaGuide pipeline:
  (a) CandidateFinder: Enumerates plausible off-target sites using
      genome-wide approximate matching (PAM-constrained)
  (b) OffTargetScorer: Scores each candidate site using a CNN over
      aligned guide-target mismatch features

The per-site scores are aggregated to a guide-level off-target risk.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OffTargetSite:
    """Represents a candidate off-target site."""
    chrom: str
    position: int
    strand: str
    sequence: str  # 23nt including PAM
    num_mismatches: int
    num_bulges: int
    mismatch_positions: List[int]
    pam_type: str  # NGG, NAG, etc.


class CandidateFinder:
    """Enumerates candidate off-target sites from a reference genome.

    Search strategy:
      1. Require a valid PAM (NGG for SpCas9; optionally NAG)
      2. Allow up to N mismatches (default 6)
      3. Apply position-dependent weighting (penalizing mismatches
         in distal regions less than in the seed)
      4. Optionally allow a small number of DNA/RNA bulges

    This is treated as an external genome search component;
    we use Cas-OFFinder or similar indexed approximate matching.
    """
    def __init__(
        self,
        max_mismatches: int = 6,
        max_bulges: int = 1,
        pam_types: List[str] = None,
    ):
        self.max_mismatches = max_mismatches
        self.max_bulges = max_bulges
        self.pam_types = pam_types or ['NGG', 'NAG']

    def find_candidates(
        self,
        guide_seq: str,
        genome_index_path: str,
    ) -> List[OffTargetSite]:
        """Find candidate off-target sites using genome-wide search.

        In production, this wraps Cas-OFFinder or a BLAST-like tool.
        Here we provide the interface; the actual genome search uses
        external tools.

        Args:
            guide_seq: 20nt guide sequence (without PAM)
            genome_index_path: Path to indexed reference genome

        Returns:
            List of candidate OffTargetSite objects
        """
        # Placeholder: in production, call Cas-OFFinder here
        # e.g., subprocess.run(['cas-offinder', input_file, 'G', output_file])
        raise NotImplementedError(
            'CandidateFinder.find_candidates() requires external genome '
            'search tool (e.g., Cas-OFFinder). Override this method or '
            'provide pre-computed candidates.'
        )

    @staticmethod
    def encode_alignment(
        guide_seq: str,
        target_seq: str,
    ) -> torch.Tensor:
        """Encode guide-target alignment as a feature tensor.

        Features per position:
          - 4 channels for guide nucleotide (one-hot)
          - 4 channels for target nucleotide (one-hot)
          - 1 channel for mismatch indicator
          - 1 channel for position weight (seed vs distal)

        Returns:
            Feature tensor (10, seq_len)
        """
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        seq_len = min(len(guide_seq), len(target_seq))

        features = torch.zeros(10, seq_len)

        for i in range(seq_len):
            g = guide_seq[i].upper()
            t = target_seq[i].upper()

            # Guide one-hot
            features[nuc_to_idx.get(g, 0), i] = 1.0
            # Target one-hot
            features[4 + nuc_to_idx.get(t, 0), i] = 1.0
            # Mismatch indicator
            features[8, i] = 0.0 if g == t else 1.0
            # Position weight (seed region: positions 1-12 from PAM)
            seed_weight = 1.0 if i >= (seq_len - 12) else 0.5
            features[9, i] = seed_weight

        return features


class OffTargetScorer(nn.Module):
    """Per-site off-target scoring function f_OT.

    Architecture: 1D-CNN over aligned guide-target pairs (Component 5.2).
    Three convolutional layers with 64, 128, and 64 filters.

    Inputs: alignment_features (batch, 10, seq_len)
    Outputs: Per-site probability p_j in [0, 1] of cleavage.

    Features:
      - Guide/Target one-hot (8 channels)
      - Mismatch indicator (1 channel)
      - Position weight (1 channel)
    """
    def __init__(
        self,
        in_channels: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_chromatin: bool = False,
        chromatin_dim: int = 0, # Default to 0 for strict seq-based OT
    ):
        super().__init__()
        self.use_chromatin = use_chromatin

        # Methodology-compliant 1D-CNN
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Global average pool
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        fc_input_dim = 64
        if use_chromatin:
            fc_input_dim += chromatin_dim

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        alignment_features: torch.Tensor,
        chromatin_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score a batch of off-target candidate sites.

        Args:
            alignment_features: Guide-target alignment (batch, 10, seq_len)
            chromatin_features: Optional local chromatin context (batch, chromatin_dim)

        Returns:
            Per-site cleavage probability (batch, 1)
        """
        h = self.conv_layers(alignment_features)
        h = self.pool(h).squeeze(-1) # (batch, 64)

        # Optionally concatenate chromatin features
        if self.use_chromatin and chromatin_features is not None:
            h = torch.cat([h, chromatin_features], dim=-1)

        return self.fc(h)  # (batch, 1)

    def aggregate_risk(
        self,
        per_site_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate per-site off-target scores to guide-level risk.

        Uses 1 - product(1 - p_j) to compute the probability that
        at least one off-target site is cleaved.

        Also computes a weighted sum considering position-dependent risk.

        Args:
            per_site_scores: Per-site probabilities (n_sites, 1)

        Returns:
            Dict with aggregated risk metrics
        """
        p = per_site_scores.squeeze(-1)  # (n_sites,)

        # Probability of at least one off-target cleavage
        prob_any = 1.0 - torch.prod(1.0 - p)

        # Sum of probabilities (expected number of off-target cuts)
        expected_cuts = p.sum()

        # Maximum per-site risk
        max_risk = p.max() if len(p) > 0 else torch.tensor(0.0)

        return {
            'prob_any_offtarget': prob_any,
            'expected_offtarget_cuts': expected_cuts,
            'max_site_risk': max_risk,
            'n_candidate_sites': len(p),
        }
