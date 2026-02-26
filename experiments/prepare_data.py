#!/usr/bin/env python3
"""Data preparation script for ChromaGuide experiments.

Downloads REAL CRISPR datasets, preprocesses, generates splits.
Run ONCE on a login node (no GPU needed).

Data sources:
    - 9 CRISPR-FMC benchmark datasets (291K+ sgRNAs):
      WT, ESP, HF (Wang et al. 2019), xCas9, SpCas9-NG, Sniper (Kim et al. 2020),
      HCT116, HELA, HL60 (Hart 2015 / Chuai 2018)
    - From: https://github.com/xx0220/CRISPR-FMC

Usage:
    python prepare_data.py [--data-dir DATA_DIR] [--skip-download]
"""
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='ChromaGuide Data Preparation (Real Data)')
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data'),
                        help='Root data directory')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download step (use existing raw data)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: Download real CRISPR datasets
    # ================================================================
    if not args.skip_download:
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading REAL CRISPR datasets")
        logger.info("=" * 60)

        from chromaguide.data.acquire import download_crispr_fmc_datasets
        downloaded = download_crispr_fmc_datasets(raw_dir)
        logger.info(f"  Downloaded {len(downloaded)}/9 datasets")

        if len(downloaded) < 9:
            logger.warning("Some datasets failed to download. Check network.")
    else:
        logger.info("Skipping download (using existing data)")

    # ================================================================
    # Step 2: Preprocess real datasets
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing REAL CRISPR data")
    logger.info("=" * 60)

    from chromaguide.data.preprocess import (
        parse_all_benchmark_datasets,
        _generate_cellline_aware_epigenomic,
        normalize_efficacy,
    )

    # Parse all 9 benchmark datasets
    combined = parse_all_benchmark_datasets(raw_dir)
    logger.info(f"Combined dataset: {len(combined)} sgRNAs")
    logger.info(f"  Cell lines: {combined['cell_line'].value_counts().to_dict()}")
    logger.info(f"  Datasets: {combined['dataset'].value_counts().to_dict()}")
    logger.info(f"  Genes: {combined['gene'].nunique()}")

    # CRITICAL VALIDATION: Check that efficacy has real variance
    eff_var = combined['efficacy'].var()
    logger.info(f"  Efficacy variance: {eff_var:.6f} (MUST be >> 0)")
    if eff_var < 0.01:
        raise RuntimeError(
            f"CRITICAL: Efficacy variance = {eff_var:.6f}. "
            "This suggests synthetic/corrupted data. Check data files."
        )

    # Epsilon-clamp for Beta regression
    eps = 1e-6
    combined['efficacy'] = combined['efficacy'].clip(eps, 1.0 - eps)

    # Epigenomic signals (synthetic until ENCODE bigWig are set up)
    n_tracks = 3
    n_bins = 100
    epi_signals = _generate_cellline_aware_epigenomic(combined, n_tracks, n_bins)

    # Save processed data (CSV for portability, no pyarrow needed)
    combined.to_csv(processed_dir / 'sequences.csv', index=False)
    np.save(processed_dir / 'efficacy.npy', combined['efficacy'].values.astype(np.float32))
    np.save(processed_dir / 'epigenomic.npy', epi_signals)

    # Also save parquet if pyarrow is available
    try:
        combined.to_parquet(processed_dir / 'sequences.parquet', index=False)
        logger.info(f"Saved: sequences.parquet ({len(combined)} rows)")
    except ImportError:
        logger.info("pyarrow not available, skipping parquet")

    logger.info(f"Saved: {processed_dir}/sequences.csv ({len(combined)} rows)")
    logger.info(f"Saved: {processed_dir}/efficacy.npy {combined['efficacy'].values.shape}")
    logger.info(f"Saved: {processed_dir}/epigenomic.npy {epi_signals.shape}")

    # ================================================================
    # Step 3: Build splits
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Building leakage-controlled splits")
    logger.info("=" * 60)

    from chromaguide.data.splits import SplitBuilder

    builder = SplitBuilder(combined, seed=42)
    splits_dir = processed_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Split A: Gene-held-out (primary)
    split_a = builder.build_split_a(train_ratio=0.70, cal_ratio=0.15, test_ratio=0.15)
    np.savez(splits_dir / 'split_a.npz', **split_a)
    logger.info(f"Split A: train={len(split_a['train'])}, cal={len(split_a['cal'])}, test={len(split_a['test'])}")

    # Split B: Dataset-held-out
    splits_b = builder.build_split_b()
    for i, split in enumerate(splits_b):
        np.savez(splits_dir / f'split_b_{i}.npz', **split)
    logger.info(f"Split B: {len(splits_b)} folds")

    # Split C: Cell-line-held-out
    splits_c = builder.build_split_c()
    for i, split in enumerate(splits_c):
        np.savez(splits_dir / f'split_c_{i}.npz', **split)
    logger.info(f"Split C: {len(splits_c)} folds")

    # 5×2cv for statistical testing
    cv_splits = builder.build_kfold_splits(n_folds=5, n_repeats=2)
    for i, split in enumerate(cv_splits):
        np.savez(
            splits_dir / f'cv_5x2_{i}.npz',
            **{k: v for k, v in split.items() if isinstance(v, np.ndarray)}
        )
    logger.info(f"5×2cv: {len(cv_splits)} splits")

    # ================================================================
    # Summary
    # ================================================================
    logger.info("=" * 60)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Total samples: {len(combined)}")
    logger.info(f"Unique sequences: {combined['sequence'].nunique()}")
    logger.info(f"Splits directory: {splits_dir}")

    for f in sorted(splits_dir.glob('*.npz')):
        d = np.load(f)
        sizes = {k: len(d[k]) for k in d.files if k in ['train', 'cal', 'test']}
        logger.info(f"  {f.name}: {sizes}")

    logger.info("")
    logger.info("REAL DATA STATISTICS:")
    for ds in combined['dataset'].unique():
        subset = combined[combined['dataset'] == ds]
        logger.info(
            f"  {ds}: n={len(subset)}, efficacy μ={subset['efficacy'].mean():.3f} ± {subset['efficacy'].std():.3f}"
        )


if __name__ == '__main__':
    main()
