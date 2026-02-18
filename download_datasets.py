#!/usr/bin/env python3
"""
Download and merge CRISPRon + major CRISPR datasets
Target: 50k+ samples with proper Z-score normalization
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
import urllib.request
import os

# ============================================================================
# DATASET URLS (Updated for 2025)
# ============================================================================

DATASETS = {
    "CRISPRon": {
        "url": "https://github.com/JasonChen-Spotec/CRISPRon/raw/master/data/all_data.csv",
        "backup_url": "https://raw.githubusercontent.com/JasonChen-Spotec/CRISPRon/master/data/all_data.csv",
        "samples": 24000,
        "columns": {
            "sequence": "sgRNA_seq",  # Column name mapping
            "efficiency": "label"
        }
    },
    "Wang2019": {
        "url": "https://github.com/Peppags/CNN-SVR/raw/master/data/Wang_dataset.csv",
        "backup_url": None,
        "samples": 13000,
        "columns": {
            "sequence": "sgRNA",
            "efficiency": "efficiency"
        }
    },
    "DeepSpCas9": {
        "url": "https://github.com/Jm-Kwak/DeepSpCas9/raw/master/data/DeepSpCas9_data.csv",
        "backup_url": None,
        "samples": 13000,
        "columns": {
            "sequence": "30mer",  # 30bp sequence
            "efficiency": "indel_freq"
        }
    }
}

# ============================================================================
# DOWNLOAD FUNCTION
# ============================================================================
def download_dataset(name, info, output_dir="./data/raw"):
    """Download a single dataset"""

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{name}.csv")

    if os.path.exists(output_file):
        print(f"✓ {name} already downloaded: {output_file}")
        return output_file

    print(f"\n📥 Downloading {name} (~{info['samples']} samples)...")

    try:
        urllib.request.urlretrieve(info['url'], output_file)
        print(f"✓ Downloaded to: {output_file}")
        return output_file

    except Exception as e:
        print(f"✗ Failed with primary URL: {e}")

        if info['backup_url']:
            print(f"Trying backup URL...")
            try:
                urllib.request.urlretrieve(info['backup_url'], output_file)
                print(f"✓ Downloaded from backup: {output_file}")
                return output_file
            except Exception as e2:
                print(f"✗ Backup also failed: {e2}")

        print(f"\n⚠️  Could not download {name}")
        print(f"Please manually download from GitHub:")
        print(f"   {info['url']}")
        print(f"   Save to: {output_file}")
        return None

# ============================================================================
# NORMALIZATION FUNCTION
# ============================================================================
def normalize_dataset(df, efficiency_col, name):
    """Apply Z-score normalization to efficiency scores"""

    print(f"\n📊 Normalizing {name}:")
    print(f"   Raw efficiency - Mean: {df[efficiency_col].mean():.4f}, Std: {df[efficiency_col].std():.4f}")

    # Z-score normalization
    df['efficiency_normalized'] = zscore(df[efficiency_col])

    print(f"   Normalized    - Mean: {df['efficiency_normalized'].mean():.4f}, Std: {df['efficiency_normalized'].std():.4f}")

    return df

# ============================================================================
# MERGE DATASETS
# ============================================================================
def merge_datasets(datasets_dict, output_file="./data/merged_crispr_data.csv"):
    """Download, normalize, and merge all datasets"""

    print("="*80)
    print("CRISPR DATASET CONSOLIDATION")
    print("="*80)

    all_frames = []
    total_samples = 0

    for name, info in datasets_dict.items():
        # Download
        file_path = download_dataset(name, info)

        if file_path is None:
            print(f"⚠️  Skipping {name} (download failed)")
            continue

        try:
            # Load
            df = pd.read_csv(file_path)
            print(f"\n✓ Loaded {name}: {len(df)} rows")
            print(f"   Columns: {df.columns.tolist()}")

            # Map columns
            seq_col = info['columns']['sequence']
            eff_col = info['columns']['efficiency']

            if seq_col not in df.columns or eff_col not in df.columns:
                print(f"⚠️  Missing required columns in {name}, skipping")
                continue

            # Extract and clean
            df_subset = df[[seq_col, eff_col]].copy()
            df_subset.columns = ['sequence', 'efficiency']
            df_subset = df_subset.dropna()

            # Normalize
            df_subset = normalize_dataset(df_subset, 'efficiency', name)

            # Add dataset source
            df_subset['dataset'] = name

            all_frames.append(df_subset)
            total_samples += len(df_subset)

            print(f"✓ Processed {len(df_subset)} samples from {name}")

        except Exception as e:
            print(f"✗ Error processing {name}: {e}")
            continue

    # Merge
    if not all_frames:
        print("\n❌ No datasets successfully loaded!")
        return None

    print(f"\n{'='*80}")
    print(f"MERGING DATASETS")
    print(f"{'='*80}")

    merged_df = pd.concat(all_frames, ignore_index=True)

    print(f"\n📦 MERGED DATASET:")
    print(f"   Total samples: {len(merged_df)}")
    print(f"   Datasets included: {merged_df['dataset'].unique().tolist()}")
    print(f"\n   Distribution:")
    print(merged_df['dataset'].value_counts())

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    print(f"\n✅ Saved merged dataset to: {output_file}")
    print(f"   Ready for training!")

    return merged_df

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    output_path = "/Users/studio/Desktop/PhD/Proposal/data/merged_crispr_data.csv"

    df = merge_datasets(DATASETS, output_path)

    if df is not None:
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS!")
        print(f"{'='*80}")
        print(f"\nNext steps:")
        print(f"1. Review: {output_path}")
        print(f"2. Run: python emergency_dnabert2_train.py")
        print(f"3. Target: Spearman ρ > 0.75")
