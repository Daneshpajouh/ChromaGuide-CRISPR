#!/usr/bin/env python3
"""Data preparation script for ChromaGuide experiments.

Downloads datasets, preprocesses, generates splits.
Run ONCE on a login node (no GPU needed).

Usage:
    python prepare_data.py [--data-dir DATA_DIR]
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
    parser = argparse.ArgumentParser(description='ChromaGuide Data Preparation')
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data'),
                        help='Root data directory')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download step (use existing raw data)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (for testing pipeline)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: Download datasets
    # ================================================================
    if not args.skip_download and not args.synthetic:
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading datasets")
        logger.info("=" * 60)
        
        try:
            from chromaguide.data.acquire import download_all
            from omegaconf import OmegaConf
            
            cfg = OmegaConf.create({
                'data': {
                    'raw_dir': str(raw_dir),
                    'processed_dir': str(processed_dir),
                }
            })
            download_all(cfg)
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            logger.warning("Will use synthetic data for development")
            args.synthetic = True
    
    # ================================================================
    # Step 2: Preprocess
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing data")
    logger.info("=" * 60)
    
    from chromaguide.data.preprocess import (
        parse_deephf, parse_crispron, 
        normalize_efficacy, extract_epigenomic_signals,
        _generate_synthetic_deephf, _generate_synthetic_crispron,
        _generate_synthetic_epigenomic
    )
    
    if args.synthetic:
        logger.info("Using SYNTHETIC data for pipeline testing")
        deephf_df = _generate_synthetic_deephf(n_per_condition=6000)
        crispron_df = _generate_synthetic_crispron(n=23902)
    else:
        logger.info("Parsing DeepHF...")
        deephf_df = parse_deephf(raw_dir)
        logger.info(f"  → {len(deephf_df)} sgRNAs")
        
        logger.info("Parsing CRISPRon...")
        crispron_df = parse_crispron(raw_dir)
        logger.info(f"  → {len(crispron_df)} sgRNAs")
    
    combined = pd.concat([deephf_df, crispron_df], ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} sgRNAs")
    logger.info(f"  Cell lines: {combined['cell_line'].value_counts().to_dict()}")
    logger.info(f"  Datasets: {combined['dataset'].value_counts().to_dict()}")
    logger.info(f"  Genes: {combined['gene'].nunique()}")
    
    # Normalize efficacy
    combined['efficacy_raw'] = combined['efficacy']
    combined['efficacy'] = normalize_efficacy(combined['efficacy'].values)
    
    # Epigenomic signals
    if args.synthetic:
        n_tracks = 3
        n_bins = 100
        epi_signals = _generate_synthetic_epigenomic(len(combined), n_tracks, n_bins)
    else:
        epi_signals = extract_epigenomic_signals(
            combined, raw_dir, n_bins=100, window_size=2000
        )
    
    # Save
    # Use CSV instead of parquet for better portability across clusters
    combined.to_csv(processed_dir / 'sequences.csv', index=False)
    np.save(processed_dir / 'efficacy.npy', combined['efficacy'].values.astype(np.float32))
    np.save(processed_dir / 'epigenomic.npy', epi_signals)
    
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
    logger.info(f"Splits directory: {splits_dir}")
    
    # List all split files
    for f in sorted(splits_dir.glob('*.npz')):
        d = np.load(f)
        sizes = {k: len(d[k]) for k in d.files if k in ['train', 'cal', 'test']}
        logger.info(f"  {f.name}: {sizes}")


if __name__ == '__main__':
    main()
