"""Main ChromaGuide model - unified multi-modal framework.

ChromaGuide is a three-component framework for sgRNA design:
  1. On-target efficacy prediction (sequence + epigenomics + Beta regression)
  2. Off-target risk prediction (CNN over aligned guide-target pairs)
  3. Integrated design score (balancing efficacy vs. safety)

This module defines the end-to-end model for on-target prediction.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .sequence_encoder import CNNGRUEncoder, MambaSequenceEncoder
from .epigenomic_encoder import EpigenomicEncoder
from .fusion import GatedAttentionFusion, ChromaGuideFusion, CrossAttentionFusion, NonRedundancyRegularizer
from .prediction_head import BetaRegressionHead, BetaNLL


class ChromaGuideModel(nn.Module):
    """ChromaGuide on-target efficacy prediction model.

    Architecture:
      Sequence -> SequenceEncoder -> z_s
      Epigenomics -> EpigenomicEncoder -> z_e
      [z_s; z_e] -> Fusion -> z_f
      z_f -> BetaRegressionHead -> (mu, phi)

    Training loss:
      L = BetaNLL(alpha, beta, y) + lambda_MI * MINE(z_s, z_e)
    """
    def __init__(
        self,
        encoder_type: str = 'cnn_gru',  # 'cnn_gru' or 'mamba'
        d_model: int = 256,
        seq_len: int = 23,
        num_epi_tracks: int = 4,
        num_epi_bins: int = 100,
        use_epigenomics: bool = True,
        use_gate_fusion: bool = False,
        fusion_type: str = 'gate',  # 'gate', 'concat', 'cross_attention'
        use_mi_regularizer: bool = False,
        mi_lambda: float = 0.01,
        dropout: float = 0.2,
        # CNN-GRU specific
        gru_hidden: int = 128,
        gru_layers: int = 2,
        # Mamba specific
        mamba_layers: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.use_epigenomics = use_epigenomics
        self.use_mi_regularizer = use_mi_regularizer
        self.mi_lambda = mi_lambda

        # Sequence encoder
        if encoder_type == 'cnn_gru':
            self.seq_encoder = CNNGRUEncoder(
                in_channels=4,
                seq_len=seq_len,
                d_model=d_model,
                gru_hidden=gru_hidden,
                gru_layers=gru_layers,
                dropout=dropout,
            )
        elif encoder_type == 'mamba':
            self.seq_encoder = MambaSequenceEncoder(
                in_channels=4,
                d_model=d_model,
                n_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=dropout,
            )
        elif encoder_type == 'dnabert2':
            from chromaguide.sequence_encoder import DNABERT2Encoder
            self.seq_encoder = DNABERT2Encoder(
                d_model=768, # Fixed at 768 for DNABERT-2
                dropout=dropout,
            )
        else:
            raise ValueError(f'Unknown encoder type: {encoder_type}')

        # Epigenomic encoder
        if use_epigenomics:
            self.epi_encoder = EpigenomicEncoder(
                num_tracks=num_epi_tracks,
                num_bins=num_epi_bins,
                d_model=d_model,
                dropout=dropout,
            )

            # Fusion - selection based on fusion_type or legacy use_gate_fusion flag
            if fusion_type == 'gate' or use_gate_fusion:
                self.fusion = GatedAttentionFusion(
                    d_model=d_model,
                    dropout=dropout,
                )
            elif fusion_type == 'cross_attention':
                self.fusion = CrossAttentionFusion(
                    d_model=d_model,
                    dropout=dropout,
                )
            else:
                # Fallback to simple concatenation fusion
                self.fusion = ChromaGuideFusion(
                    d_model=d_model,
                    hidden_dim=d_model * 2,
                    dropout=dropout,
                    use_gate=False,
                )

            # Optional MI regularizer
            if use_mi_regularizer:
                self.mi_reg = NonRedundancyRegularizer(d_model=d_model)

        # Prediction head
        self.pred_head = BetaRegressionHead(
            d_model=d_model,
            dropout=dropout,
        )

        # Loss function
        self.loss_fn = BetaNLL()

    def forward(
        self,
        seq: torch.Tensor,
        epi_tracks: Optional[torch.Tensor] = None,
        epi_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            seq: One-hot encoded sequence (batch, 4, seq_len)
            epi_tracks: Epigenomic tracks (batch, num_tracks, num_bins)
            epi_mask: Assay availability mask (batch, num_tracks)

        Returns:
            Dict with predictions and intermediate embeddings:
              'mu': Mean efficacy prediction
              'phi': Concentration parameter
              'alpha', 'beta': Beta distribution parameters
              'z_s': Sequence embedding
              'z_e': Epigenomic embedding (if used)
              'z_f': Fused embedding
        """
        # Sequence encoding
        z_s = self.seq_encoder(seq)  # (batch, d_model)

        output = {'z_s': z_s}

        # Epigenomic encoding and fusion
        if self.use_epigenomics and epi_tracks is not None:
            z_e = self.epi_encoder(epi_tracks, epi_mask)  # (batch, d_model)
            z_f = self.fusion(z_s, z_e)  # (batch, d_model)
            output['z_e'] = z_e
        else:
            z_f = z_s  # Sequence-only mode

        output['z_f'] = z_f

        # Prediction
        pred = self.pred_head(z_f)
        output.update(pred)

        return output

    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.

        Args:
            output: Model output dict from forward()
            y: True efficacy values (batch, 1) in (0, 1)

        Returns:
            Dict with loss components:
              'total_loss': Combined loss
              'beta_nll': Beta NLL loss
              'mi_loss': MI regularization loss (if enabled)
        """
        beta_nll = self.loss_fn(output['alpha'], output['beta'], y)
        losses = {'beta_nll': beta_nll}

        total_loss = beta_nll

        # MI regularization
        if (
            self.use_mi_regularizer
            and self.use_epigenomics
            and 'z_e' in output
        ):
            mi_loss = self.mi_reg(output['z_s'], output['z_e'])
            total_loss = total_loss + self.mi_lambda * mi_loss
            losses['mi_loss'] = mi_loss

        losses['total_loss'] = total_loss
        return losses

    @torch.no_grad()
    def predict(self, seq, epi_tracks=None, epi_mask=None):
        """Convenience method for inference."""
        self.eval()
        output = self.forward(seq, epi_tracks, epi_mask)
        return {
            'efficacy': output['mu'].squeeze(-1),
            'uncertainty': (1.0 / output['phi']).squeeze(-1),
        }
