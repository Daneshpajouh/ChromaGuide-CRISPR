"""Epigenomic encoder module for ChromaGuide.

Processes chromatin accessibility and histone-mark tracks around the
target cut site to produce an epigenomic context embedding z_e.

For each assay (DNase/ATAC, H3K4me3, H3K27ac), extracts a window
(default +/- 5kb) around the cut site, bins to fixed-length features,
applies log1p and training-only standardization, and maps through an MLP.

Assay-availability indicator features are included so the model can
gracefully handle missing tracks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ENCODE cell-line mapping for DeepHF benchmark
# HEK293T, HCT116, and HeLa
ENCODE_CELL_LINES = {
    'HEK293T': {
        'DNase': 'ENCSR000EJR',
        'H3K4me3_ChIP': 'ENCSR000DTU',
        'H3K27ac_ChIP': 'ENCSR000FCJ',
    },
    'HCT116': {
        'DNase': 'ENCSR000ENO',
        'H3K4me3_ChIP': 'ENCSR000DWN',
        'ATAC_seq': 'ENCSR000EOJ',
    },
    'HeLa': {
        'DNase': 'ENCSR000ENP',
        'H3K4me3_ChIP': 'ENCSR000DWE',
        'H3K27ac_ChIP': 'ENCSR000FCG',
    },
}

# Default number of epigenomic tracks per cell line (per PhD proposal)
DEFAULT_NUM_TRACKS = 3  # e.g., DNase, H3K4me3, H3K27ac


class EpigenomicEncoder(nn.Module):
    """Encodes epigenomic context features into a fixed-size embedding.

    Input: Binned signal tracks around the cut site.
      Shape: (batch, num_tracks, num_bins)
      where num_tracks = 3 (DNase, H3K4me3, H3K27ac) as per PhD proposal
      and num_bins = number of bins in the window (e.g., 100 bins for 10kb window)

    The encoder also takes an assay-availability mask:
      Shape: (batch, num_tracks)
      Binary mask indicating which tracks are available for each sample.
      Missing tracks are set to zero and the indicator feature informs the model.

    Output: Epigenomic embedding z_e of shape (batch, d_model)
    """
    def __init__(
        self,
        num_tracks: int = DEFAULT_NUM_TRACKS,
        num_bins: int = 100,
        d_model: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.num_bins = num_bins
        self.d_model = d_model

        # Per-track 1D convolution for local pattern extraction
        self.track_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_dim // num_tracks, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim // num_tracks),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            for _ in range(num_tracks)
        ])

        # Assay availability indicator (binary features)
        # Concatenated with track features to signal missing data
        feature_dim = (hidden_dim // num_tracks) * num_tracks + num_tracks

        # MLP to project to d_model
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tracks: torch.Tensor,
        availability_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tracks: Binned epigenomic signals (batch, num_tracks, num_bins)
                    Values should be log1p-transformed and standardized.
            availability_mask: Binary mask (batch, num_tracks)
                              1.0 = track available, 0.0 = missing.
                              If None, assumes all tracks are available.

        Returns:
            Epigenomic embedding z_e (batch, d_model)
        """
        B = tracks.shape[0]

        if availability_mask is None:
            availability_mask = torch.ones(B, self.num_tracks, device=tracks.device)

        # Zero out unavailable tracks
        tracks = tracks * availability_mask.unsqueeze(-1)

        # Process each track through its own conv branch
        track_features = []
        for i, conv in enumerate(self.track_convs):
            t_i = tracks[:, i:i+1, :]  # (batch, 1, num_bins)
            feat_i = conv(t_i).squeeze(-1)  # (batch, hidden_dim // num_tracks)
            track_features.append(feat_i)

        # Concatenate track features with availability indicators
        h = torch.cat(track_features + [availability_mask], dim=-1)

        return self.mlp(h)  # (batch, d_model)


class EpigenomicPreprocessor:
    """Preprocessor for raw bigWig epigenomic signal files.

    Handles:
      - Extracting signal windows around cut sites
      - Binning to fixed resolution
      - log1p transformation
      - Training-set standardization
    """
    def __init__(
        self,
        window_size: int = 10000,  # +/- 5kb from cut site
        num_bins: int = 100,
        cell_line: str = 'HEK293T',
    ):
        self.window_size = window_size
        self.num_bins = num_bins
        self.bin_size = window_size // num_bins
        self.cell_line = cell_line
        self._mean = None
        self._std = None

    def extract_window(self, chrom: str, cut_pos: int, bigwig_file) -> torch.Tensor:
        """Extract and bin signal from a bigWig file around the cut site."""
        import pyBigWig
        half_window = self.window_size // 2
        start = max(0, cut_pos - half_window)
        end = cut_pos + half_window

        bw = pyBigWig.open(bigwig_file)
        values = bw.stats(chrom, start, end, nBins=self.num_bins, type='mean')
        bw.close()

        # Replace None with 0 (regions with no data)
        values = [v if v is not None else 0.0 for v in values]
        signal = torch.tensor(values, dtype=torch.float32)

        # log1p transform
        signal = torch.log1p(signal)
        return signal

    def fit(self, signals: torch.Tensor):
        """Compute training-set mean and std for standardization."""
        self._mean = signals.mean(dim=0, keepdim=True)
        self._std = signals.std(dim=0, keepdim=True).clamp(min=1e-8)

    def transform(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply standardization using training statistics."""
        if self._mean is not None:
            return (signal - self._mean) / self._std
        return signal
