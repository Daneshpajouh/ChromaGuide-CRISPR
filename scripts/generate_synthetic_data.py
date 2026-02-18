#!/usr/bin/env python3
"""
Generate synthetic training data for PhD thesis experiments
Creates realistic mock datasets when real data is unavailable

This is for testing the training pipeline only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
import os

def generate_random_dna_sequence(length: int = 30) -> str:
    """Generate random DNA sequence of specified length."""
    return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))

def generate_genes(n_genes: int = 50) -> list:
    """Generate fake gene names."""
    return [f"GENE_{i:04d}" for i in range(n_genes)]

def generate_synthetic_data(cell_line: str, n_samples: int = 500) -> pd.DataFrame:
    """
    Generate synthetic CRISPR efficacy data.
    
    Args:
        cell_line: Name of cell line
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with columns: sequence, intensity, gene (if applicable)
    """
    
    genes = generate_genes(min(50, n_samples // 10))
    
    sequences = [generate_random_dna_sequence() for _ in range(n_samples)]
    intensities = np.random.beta(2, 5, n_samples)  # Beta distribution [0, 1]
    sequence_genes = [random.choice(genes) for _ in range(n_samples)]
    
    df = pd.DataFrame({
        'sequence': sequences,
        'intensity': intensities,
        'gene': sequence_genes,
        'cell_line': [cell_line] * n_samples
    })
    
    return df

def create_leakage_controlled_splits(train_df: pd.DataFrame, 
                                     val_df: pd.DataFrame,
                                      test_df: pd.DataFrame) -> tuple:
    """
    Create leakage-controlled split:
    - Each gene appears in only one of train/val/test
    """
    
    unique_genes_train = set(train_df['gene'].unique())
    unique_genes_val = set(val_df['gene'].unique())
    unique_genes_test = set(test_df['gene'].unique())
    
    # Verify no overlap
    assert len(unique_genes_train & unique_genes_val) == 0, "Train/Val gene overlap"
    assert len(unique_genes_train & unique_genes_test) == 0, "Train/Test gene overlap"
    assert len(unique_genes_val & unique_genes_test) == 0, "Val/Test gene overlap"
    
    return train_df, val_df, test_df

def main():
    """Generate all synthetic data."""
    
    print("=" * 80)
    print("SYNTHETIC TRAINING DATA GENERATION")
    print("=" * 80)
    
    # Create base directory structure
    data_dir = Path("/home/amird/chromaguide_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = data_dir / "raw" / "deeprf"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = data_dir / "splits" / "split_a_gene_held_out"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    cell_lines = ['HEK293T', 'HCT116', 'HeLa']
    
    # Generate raw data for each cell line
    print("\nGenerating raw synthetic datasets...")
    all_data = {}
    for cell_line in cell_lines:
        df = generate_synthetic_data(cell_line, n_samples=400)
        raw_path = raw_dir / f"{cell_line}.csv"
        df.to_csv(raw_path, index=False)
        all_data[cell_line] = df
        print(f"  ✓ {cell_line}: {len(df)} samples → {raw_path}")
    
    # Combine all data and create leakage-controlled splits
    print("\nCreating leakage-controlled splits...")
    combined_data = pd.concat(all_data.values(), ignore_index=True)
    print(f"  Total samples: {len(combined_data)}")
    
    # Get unique genes
    unique_genes = combined_data['gene'].unique()
    n_genes = len(unique_genes)
    print(f"  Total unique genes: {n_genes}")
    
    # Split genes (not sequences) to prevent leakage
    np.random.seed(42)
    gene_indices = np.random.permutation(n_genes)
    
    train_genes = set(unique_genes[gene_indices[:int(0.7 * n_genes)]])
    val_genes = set(unique_genes[gene_indices[int(0.7 * n_genes):int(0.85 * n_genes)]])
    test_genes = set(unique_genes[gene_indices[int(0.85 * n_genes):]])
    
    # Split data by genes (leakage-controlled)
    train_df = combined_data[combined_data['gene'].isin(train_genes)].reset_index(drop=True)
    val_df = combined_data[combined_data['gene'].isin(val_genes)].reset_index(drop=True)
    test_df = combined_data[combined_data['gene'].isin(test_genes)].reset_index(drop=True)
    
    print(f"  Train: {len(train_df)} samples ({len(train_genes)} genes)")
    print(f"  Val:   {len(val_df)} samples ({len(val_genes)} genes)")
    print(f"  Test:  {len(test_df)} samples ({len(test_genes)} genes)")
    
    # Save splits
    print(f"\nSaving splits to {splits_dir}...")
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "validation.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    
    print("  ✓ train.csv")
    print("  ✓ validation.csv")
    print("  ✓ test.csv")
    
    print("\n" + "=" * 80)
    print("✓ Synthetic data generation complete!")
    print("=" * 80)
    print(f"\nData structure:")
    print(f"  {data_dir}")
    print(f"  ├── raw/deeprf/")
    print(f"  │   ├── HEK293T.csv (synthetic)")
    print(f"  │   ├── HCT116.csv (synthetic)")
    print(f"  │   └── HeLa.csv (synthetic)")
    print(f"  └── splits/split_a_gene_held_out/")
    print(f"      ├── train.csv ({len(train_df)} samples)")
    print(f"      ├── validation.csv ({len(val_df)} samples)")
    print(f"      └── test.csv ({len(test_df)} samples)")
    print()

if __name__ == "__main__":
    main()
