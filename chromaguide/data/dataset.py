"""PyTorch Dataset classes for ChromaGuide.

Provides:
    - CRISPRDataset: On-target efficacy prediction (sequence + epigenomics → efficacy)
    - OffTargetDataset: Off-target cleavage prediction (guide-target pairs → binary)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


class CRISPRDataset(Dataset):
    """Dataset for on-target efficacy prediction.
    
    Each sample contains:
        - One-hot encoded sgRNA sequence (4, seq_len)
        - Binned epigenomic signals (n_tracks, n_bins)
        - Efficacy score (scalar in (0, 1))
        - Metadata (cell_line, gene, dataset)
    """
    
    DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
    
    def __init__(
        self,
        sequences: pd.Series | list[str],
        efficacy: np.ndarray,
        epigenomic: np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
        seq_len: int = 23,
        augment: bool = False,
    ):
        """
        Args:
            sequences: sgRNA sequences (strings).
            efficacy: Efficacy scores, shape (n,).
            epigenomic: Epigenomic signals, shape (n, n_tracks, n_bins).
            metadata: Optional metadata DataFrame.
            seq_len: Sequence length (20nt + 3nt PAM = 23).
            augment: Whether to apply data augmentation.
        """
        self.sequences = list(sequences)
        self.efficacy = efficacy.astype(np.float32)
        self.epigenomic = epigenomic
        self.metadata = metadata
        self.seq_len = seq_len
        self.augment = augment
        
        assert len(self.sequences) == len(self.efficacy)
        if self.epigenomic is not None:
            assert len(self.sequences) == len(self.epigenomic)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        # One-hot encode sequence
        seq = self.sequences[idx].upper()[:self.seq_len]
        one_hot = torch.zeros(4, self.seq_len, dtype=torch.float32)
        for i, base in enumerate(seq):
            if base in self.DNA_MAP and i < self.seq_len:
                one_hot[self.DNA_MAP[base], i] = 1.0
        
        # Efficacy target
        target = torch.tensor(self.efficacy[idx], dtype=torch.float32)
        
        # Epigenomic signals
        if self.epigenomic is not None:
            epi = torch.tensor(self.epigenomic[idx], dtype=torch.float32)
        else:
            epi = torch.zeros(3, 100, dtype=torch.float32)
        
        # Augmentation
        if self.augment and self.training_mode:
            epi = self._augment_epigenomic(epi)
        
        sample = {
            "sequence": one_hot,
            "sequence_str": seq,
            "epigenomic": epi,
            "efficacy": target,
        }
        
        if self.metadata is not None:
            sample["cell_line"] = self.metadata.iloc[idx]["cell_line"]
            sample["gene"] = self.metadata.iloc[idx]["gene"]
        
        return sample
    
    @property
    def training_mode(self):
        return self.augment
    
    def _augment_epigenomic(self, epi: torch.Tensor) -> torch.Tensor:
        """Light epigenomic augmentation: Gaussian noise + random scaling."""
        if torch.rand(1).item() < 0.3:
            epi = epi + torch.randn_like(epi) * 0.05
        if torch.rand(1).item() < 0.2:
            scale = 0.9 + 0.2 * torch.rand(1).item()
            epi = epi * scale
        return epi
    
    @classmethod
    def from_processed(
        cls,
        processed_dir: str | Path,
        split_file: str | Path | None = None,
        subset: str = "train",
        augment: bool = False,
    ) -> "CRISPRDataset":
        """Load dataset from preprocessed files.
        
        Args:
            processed_dir: Path to processed data directory.
            split_file: Path to .npz split file.
            subset: Which split subset ('train', 'cal', 'test').
            augment: Enable augmentation (train only).
        """
        processed_dir = Path(processed_dir)
        
        df = pd.read_parquet(processed_dir / "sequences.parquet")
        efficacy = np.load(processed_dir / "efficacy.npy")
        epigenomic = np.load(processed_dir / "epigenomic.npy")
        
        if split_file is not None:
            split = np.load(split_file)
            idx = split[subset]
            df = df.iloc[idx].reset_index(drop=True)
            efficacy = efficacy[idx]
            epigenomic = epigenomic[idx]
        
        return cls(
            sequences=df["sequence"],
            efficacy=efficacy,
            epigenomic=epigenomic,
            metadata=df[["cell_line", "gene", "dataset"]],
            augment=augment,
        )


class OffTargetDataset(Dataset):
    """Dataset for off-target cleavage prediction.
    
    Each sample contains:
        - Encoded guide-target alignment (12, seq_len)
        - Optional chromatin features at off-target site (3,)
        - Binary label (1 = cleavage, 0 = no cleavage)
    """
    
    DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
    
    def __init__(
        self,
        guides: list[str],
        targets: list[str],
        labels: np.ndarray,
        chromatin: np.ndarray | None = None,
        seq_len: int = 23,
    ):
        self.guides = guides
        self.targets = targets
        self.labels = labels.astype(np.float32)
        self.chromatin = chromatin
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return len(self.guides)
    
    def __getitem__(self, idx: int) -> dict:
        guide = self.guides[idx].upper()[:self.seq_len]
        target = self.targets[idx].upper()[:self.seq_len]
        
        # Encode alignment
        encoding = torch.zeros(12, self.seq_len, dtype=torch.float32)
        for i, (g, t) in enumerate(zip(guide, target)):
            if i >= self.seq_len:
                break
            # Guide one-hot
            if g in self.DNA_MAP:
                encoding[self.DNA_MAP[g], i] = 1.0
            # Target one-hot
            if t in self.DNA_MAP:
                encoding[4 + self.DNA_MAP[t], i] = 1.0
            # Match/mismatch
            encoding[8, i] = float(g == t)
            encoding[9, i] = float(g != t)
            # Position
            encoding[11, i] = i / self.seq_len
        
        sample = {
            "alignment": encoding,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        
        if self.chromatin is not None:
            sample["chromatin"] = torch.tensor(
                self.chromatin[idx], dtype=torch.float32
            )
        
        return sample


def create_dataloaders(
    dataset: CRISPRDataset,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=isinstance(dataset, CRISPRDataset) and dataset.augment,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
