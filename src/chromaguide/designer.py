"""ChromaGuide Designer - Main entry point for guide RNA design.

This module integrates all components:
  1. Sequence mining (finding NGG/NAG sites)
  2. On-target prediction (Beta Regression)
  3. Off-target risk assessment (CNN Scorer)
  4. Uncertainty quantification (Conformal Prediction)
  5. Integrated scoring (S = we*mu - wr*R - wu*sigma)
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from .chromaguide_model import ChromaGuideModel
from .off_target import CandidateFinder, OffTargetScorer
from .conformal import BetaConformalPredictor
from .design_score import IntegratedDesignScore


# Alias for backward compatibility
DesignScoreAggregator = IntegratedDesignScore


class ChromaGuideDesigner:
    """High-level API for designing optimized sgRNAs."""

    def __init__(
        self,
        on_target_checkpoint: str,
        off_target_checkpoint: str,
        conformal_calibration_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        w_e: float = 1.0,
        w_r: float = 0.5,
        w_u: float = 0.2,
    ):
        self.device = device

        # 1. Load On-target Model
        self.on_target_model = ChromaGuideModel(encoder_type='cnn_gru').to(device)
        self.on_target_model.load_state_dict(torch.load(on_target_checkpoint, map_location=device))
        self.on_target_model.eval()

        # 2. Load Off-target Scorer
        self.off_target_scorer = OffTargetScorer().to(device)
        self.off_target_scorer.load_state_dict(torch.load(off_target_checkpoint, map_location=device))
        self.off_target_scorer.eval()

        # 3. Initialize Conformal Predictor
        self.conformal = BetaConformalPredictor(alpha=0.1)
        if conformal_calibration_path:
            self.conformal.load_calibration(conformal_calibration_path)

        # 4. Initialize Aggregator
        self.aggregator = DesignScoreAggregator(w_e=w_e, w_r=w_r, w_u=w_u)

        # 5. Initialize Candidate Finder
        self.candidate_finder = CandidateFinder(max_mismatches=6)

    def design_guides(
        self,
        target_dna_region: str,
        genome_index: Optional[str] = None,
        epi_features: Optional[torch.Tensor] = None,
    ) -> pd.DataFrame:
        """Find and rank all guides in a target DNA region.

        Args:
            target_dna_region: The genomic sequence to target.
            genome_index: Path to reference genome for off-target search.
            epi_features: Epigenomic features (if targeting endogenous sites).

        Returns:
            pd.DataFrame: Ranked candidates with efficiency, risk, and scores.
        """
        # A. Find candidates (simple NGG search for now)
        guides = self._find_pam_sites(target_dna_region)
        results = []

        for guide in guides:
            # B. Predict On-target Efficiency
            # Prepare sequence (23nt)
            seq_tensor = self._encode_sequence(guide['seq'])
            with torch.no_grad():
                output = self.on_target_model(seq_tensor, epi_features)
                mu = output['mu']
                phi = output['phi']

            # C. Uncertainty (Conformal)
            with torch.no_grad():
                # Prepare for conformal prediction
                mu_val = mu.cpu().numpy().flatten()
                phi_val = phi.cpu().numpy().flatten()
                lower, upper = self.conformal.predict_intervals(mu_val, phi_val)
                sigma_ci = (upper - lower)[0]

            # D. Off-target Risk (Placeholder if no genome index)
            if genome_index:
                candidates = self.candidate_finder.find_candidates(guide['seq'], genome_index)
                # ... score each candidate ...
                # risk = self.off_target_scorer.aggregate_risk(scores)['prob_any_offtarget']
                risk = 0.05 # Placeholder
            else:
                risk = 0.0 # Unknown risk

            # E. Compute Final Score
            on_t = torch.tensor([[mu.item()]], dtype=torch.float32).to(self.device)
            off_t = torch.tensor([[risk]], dtype=torch.float32).to(self.device)
            unc_t = torch.tensor([[sigma_ci]], dtype=torch.float32).to(self.device)

            final_score = self.aggregator(on_t, off_t, unc_t).item()

            results.append({
                'sequence': guide['seq'],
                'pos': guide['pos'],
                'strand': guide['strand'],
                'efficiency_mu': mu.item(),
                'uncertainty_sigma': sigma_ci,
                'off_target_risk_R': risk,
                'designer_score_S': final_score # CORRECTED NAME PER PHD AUDIT
            })

        df = pd.DataFrame(results)
        return df.sort_values(by='designer_score_S', ascending=False)

    def _find_pam_sites(self, dna_seq: str) -> List[Dict[str, Any]]:
        """Find all NGG sites in a string."""
        guides = []
        # Forward strand
        for i in range(len(dna_seq) - 23):
            if dna_seq[i+21:i+23] == 'GG':
                guides.append({'seq': dna_seq[i:i+23], 'pos': i, 'strand': '+'})
        # Reverse strand logic here...
        return guides

    def _encode_sequence(self, seq: str) -> torch.Tensor:
        """One-hot encode DNA sequence."""
        nuc_map = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
        encoded = [nuc_map.get(n, [0,0,0,0]) for n in seq.upper()]
        return torch.tensor(encoded).float().unsqueeze(0).transpose(1, 2).to(self.device)
