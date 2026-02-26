"""Leakage-controlled data split construction.

Implements three split strategies:
    Split A: Gene-held-out (primary evaluation)
        - Train/cal/test by gene group — no gene appears in multiple splits
        - Prevents information leakage from correlated guides targeting same gene
    
    Split B: Dataset-held-out
        - Leave-one-dataset-out cross-validation
        - Tests generalization across experimental protocols
    
    Split C: Cell-line-held-out  
        - Leave-one-cell-line-out cross-validation
        - Tests generalization across biological contexts

All splits include:
    - Deduplication (identical sequences removed across splits)
    - Calibration set (15% of data) for conformal prediction
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from typing import Optional
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SplitBuilder:
    """Constructs leakage-controlled train/cal/test splits."""
    
    def __init__(self, df: pd.DataFrame, seed: int = 42):
        """
        Args:
            df: DataFrame with columns: sequence, efficacy, cell_line, 
                cas9_variant, gene, dataset
            seed: Random seed for reproducibility.
        """
        self.df = df.copy()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Deduplication
        n_before = len(self.df)
        self.df = self.df.drop_duplicates(subset=["sequence"], keep="first")
        n_after = len(self.df)
        if n_before > n_after:
            logger.info(f"Deduplication: {n_before} → {n_after} ({n_before - n_after} duplicates removed)")
    
    def build_split_a(
        self,
        train_ratio: float = 0.70,
        cal_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict[str, np.ndarray]:
        """Split A: Gene-held-out.
        
        Groups sgRNAs by gene, then assigns whole gene groups to
        train/cal/test. Ensures no gene-level leakage.
        
        Returns dict with 'train', 'cal', 'test' index arrays.
        """
        logger.info("Building Split A (gene-held-out)...")
        
        genes = self.df["gene"].values
        unique_genes = np.unique(genes)
        n_genes = len(unique_genes)
        
        logger.info(f"  {n_genes} unique genes, {len(self.df)} sgRNAs")
        
        # Shuffle genes
        gene_order = self.rng.permutation(unique_genes)
        
        # Split gene groups
        n_test_genes = max(1, int(n_genes * test_ratio))
        n_cal_genes = max(1, int(n_genes * cal_ratio))
        
        test_genes = set(gene_order[:n_test_genes])
        cal_genes = set(gene_order[n_test_genes:n_test_genes + n_cal_genes])
        train_genes = set(gene_order[n_test_genes + n_cal_genes:])
        
        # Map back to sample indices
        train_idx = np.where(self.df["gene"].isin(train_genes))[0]
        cal_idx = np.where(self.df["gene"].isin(cal_genes))[0]
        test_idx = np.where(self.df["gene"].isin(test_genes))[0]
        
        # Verify no leakage
        assert len(set(self.df.iloc[train_idx]["gene"]) & set(self.df.iloc[test_idx]["gene"])) == 0
        assert len(set(self.df.iloc[train_idx]["gene"]) & set(self.df.iloc[cal_idx]["gene"])) == 0
        assert len(set(self.df.iloc[cal_idx]["gene"]) & set(self.df.iloc[test_idx]["gene"])) == 0
        
        split = {"train": train_idx, "cal": cal_idx, "test": test_idx}
        self._log_split_stats(split, "A")
        
        return split
    
    def build_split_b(self) -> list[dict[str, np.ndarray]]:
        """Split B: Dataset-held-out (leave-one-dataset-out).
        
        For each dataset d:
            - test = dataset d
            - cal = 15% of remaining data
            - train = rest
        
        Returns list of split dicts (one per held-out dataset).
        """
        logger.info("Building Split B (dataset-held-out)...")
        
        datasets = self.df["dataset"].unique()
        splits = []
        
        for held_out in datasets:
            test_idx = np.where(self.df["dataset"] == held_out)[0]
            remain_idx = np.where(self.df["dataset"] != held_out)[0]
            
            # Split remaining into train/cal
            n_cal = max(1, int(len(remain_idx) * 0.15))
            self.rng.shuffle(remain_idx)
            cal_idx = remain_idx[:n_cal]
            train_idx = remain_idx[n_cal:]
            
            split = {"train": train_idx, "cal": cal_idx, "test": test_idx}
            splits.append(split)
            
            logger.info(f"  Split B ({held_out}): train={len(train_idx)}, cal={len(cal_idx)}, test={len(test_idx)}")
        
        return splits
    
    def build_split_c(self) -> list[dict[str, np.ndarray]]:
        """Split C: Cell-line-held-out (leave-one-cell-line-out).
        
        For each cell line c:
            - test = cell line c
            - cal = 15% of remaining data
            - train = rest
        
        Returns list of split dicts (one per held-out cell line).
        """
        logger.info("Building Split C (cell-line-held-out)...")
        
        cell_lines = self.df["cell_line"].unique()
        splits = []
        
        for held_out in cell_lines:
            test_idx = np.where(self.df["cell_line"] == held_out)[0]
            remain_idx = np.where(self.df["cell_line"] != held_out)[0]
            
            if len(remain_idx) == 0:
                logger.warning(f"  Skipping {held_out}: no remaining data")
                continue
            
            n_cal = max(1, int(len(remain_idx) * 0.15))
            self.rng.shuffle(remain_idx)
            cal_idx = remain_idx[:n_cal]
            train_idx = remain_idx[n_cal:]
            
            split = {"train": train_idx, "cal": cal_idx, "test": test_idx}
            splits.append(split)
            
            logger.info(f"  Split C ({held_out}): train={len(train_idx)}, cal={len(cal_idx)}, test={len(test_idx)}")
        
        return splits
    
    def build_kfold_splits(
        self, n_folds: int = 5, n_repeats: int = 2,
    ) -> list[dict[str, np.ndarray]]:
        """5×2 cross-validation splits for statistical testing.
        
        Returns list of 10 train/test splits (5 folds × 2 repeats).
        """
        logger.info(f"Building {n_repeats}×{n_folds}cv splits...")
        splits = []
        
        for repeat in range(n_repeats):
            # Gene-grouped k-fold
            genes = self.df["gene"].values
            unique_genes = np.unique(genes)
            self.rng.shuffle(unique_genes)
            
            fold_genes = np.array_split(unique_genes, n_folds)
            
            for fold_idx, test_gene_set in enumerate(fold_genes):
                test_gene_set = set(test_gene_set)
                test_idx = np.where(self.df["gene"].isin(test_gene_set))[0]
                train_idx = np.where(~self.df["gene"].isin(test_gene_set))[0]
                
                splits.append({
                    "train": train_idx,
                    "test": test_idx,
                    "fold": fold_idx,
                    "repeat": repeat,
                })
        
        return splits
    
    def _log_split_stats(self, split: dict, name: str) -> None:
        """Log split statistics."""
        for key, idx in split.items():
            if isinstance(idx, np.ndarray) and len(idx) > 0:
                subset = self.df.iloc[idx]
                eff = subset["efficacy"]
                logger.info(
                    f"  Split {name} {key}: n={len(idx)}, "
                    f"efficacy μ={eff.mean():.3f} ± {eff.std():.3f}, "
                    f"genes={subset['gene'].nunique()}, "
                    f"cell_lines={list(subset['cell_line'].unique())}"
                )


def build_all_splits(cfg: DictConfig) -> None:
    """Build and save all split configurations.
    
    Usage:
        chromaguide data --stage splits
    """
    processed_dir = Path(cfg.data.processed_dir)
    
    # Load processed data
    df = pd.read_parquet(processed_dir / "sequences.parquet")
    
    builder = SplitBuilder(df, seed=cfg.project.seed)
    
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Split A
    split_a = builder.build_split_a(
        train_ratio=cfg.splits.split_a.train_ratio,
        cal_ratio=cfg.splits.split_a.cal_ratio,
        test_ratio=cfg.splits.split_a.test_ratio,
    )
    np.savez(splits_dir / "split_a.npz", **split_a)
    
    # Split B
    splits_b = builder.build_split_b()
    for i, split in enumerate(splits_b):
        np.savez(splits_dir / f"split_b_{i}.npz", **split)
    
    # Split C
    splits_c = builder.build_split_c()
    for i, split in enumerate(splits_c):
        np.savez(splits_dir / f"split_c_{i}.npz", **split)
    
    # 5×2cv splits for statistical testing
    cv_splits = builder.build_kfold_splits(n_folds=5, n_repeats=2)
    for i, split in enumerate(cv_splits):
        np.savez(splits_dir / f"cv_5x2_{i}.npz", **{k: v for k, v in split.items() if isinstance(v, np.ndarray)})
    
    logger.info(f"\nAll splits saved to: {splits_dir}")
