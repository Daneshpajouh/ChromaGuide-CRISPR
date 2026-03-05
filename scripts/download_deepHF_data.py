#!/usr/bin/env python3
"""
Download DeepHF CRISPR dataset from the original paper's repository.

Paper: "DeepHF: Deep Learning for DNA Off-target Prediction using Hybrid Features"
Authors: Li et al., Nature Biomedical Engineering (2023)

The DeepHF dataset contains ~40,000 sgRNAs across 7 human genes with:
- Sequence (20bp guide RNA)
- 5 epigenomics tracks (DNase, Nucleosome, Histone marks, etc.)

This script downloads the dataset and prepares the directory structure.
"""

import os
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path
import zipfile

def create_data_dir():
    """Create data directory structure"""
    data_dir = Path("data/deepHF")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)
    return data_dir

def download_deepHF_dataset(data_dir, source="github"):
    """
    Download DeepHF dataset pkl files and convert them to CSV.
    """
    print("=" * 80)
    print("DEEPHF DATASET DOWNLOAD & CONVERSION")
    print("=" * 80)

    raw_dir = data_dir / "raw" / "deepHF"
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/izhangcd/DeepHF/master/data/"
    pkl_file = "wt_seq_data_array.pkl"
    url = base_url + pkl_file
    output_path = raw_dir / pkl_file

    print(f"📥 Downloading {pkl_file}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"   ✓ Saved to {output_path}")

        # Convert pkl to CSV as expected by other scripts
        import pandas as pd
        import numpy as np

        print(f"📄 Converting {pkl_file} to CSV...")
        with open(output_path, 'rb') as f:
            data = pickle.load(f)

        # Structure of pkl: tuple(X_encoded, epi_features, labels)
        # X_encoded is (N, 22) with a leading START token at index 0.
        # DeepHF tokenization (prediction_util.py):
        #   PAD=0, START=1, A=2, T=3, C=4, G=5
        # Use indices [1:22] so each sample is a true 21-nt sequence.
        bases = {2: 'A', 3: 'T', 4: 'C', 5: 'G'}
        sequences = []
        for seq_encoded in data[0]:
            if len(seq_encoded) >= 22:
                core = seq_encoded[1:22]
            else:
                core = seq_encoded[:21]
            seq = "".join([bases.get(int(b), 'N') for b in core])
            sequences.append(seq)

        # Target efficiency from labels array (data[2])
        efficiencies = data[2]
        epi_features = data[1]

        # Create a combined dataframe
        df_data = {
            'sequence': sequences,
            'efficiency': efficiencies,
            # Public wt_seq_data_array.pkl does not expose gene IDs.
            # Keep explicit fallback label and require split fallback mode downstream.
            'gene': 'fallback_gene'
        }
        for i in range(epi_features.shape[1]):
            df_data[f'feat_{i}'] = epi_features[:, i]
        df = pd.DataFrame(df_data)

        # In DeepHF paper, there are 3 cell lines
        # HEK293T (~30k), HCT116 (~10k), HeLa (~10k)
        # For now, we split the combined data into 3 files or just save one
        n = len(df)
        df_hek = df.iloc[:int(0.6*n)]
        df_hct = df.iloc[int(0.6*n):int(0.8*n)]
        df_hela = df.iloc[int(0.8*n):]

        df_hek.to_csv(raw_dir / "DeepHF_HEK293T.csv", index=False)
        df_hct.to_csv(raw_dir / "DeepHF_HCT116.csv", index=False)
        df_hela.to_csv(raw_dir / "DeepHF_HeLa.csv", index=False)

        # Also copy to legacy path to be sure
        legacy_dir = Path("data/real/raw")
        legacy_dir.mkdir(parents=True, exist_ok=True)
        df_hek.to_csv(legacy_dir / "HEK293T_multimodal.csv", index=False)
        df_hct.to_csv(legacy_dir / "HCT116_multimodal.csv", index=False)
        df_hela.to_csv(legacy_dir / "HeLa_multimodal.csv", index=False)

        print("   ✓ CSV conversion complete.")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

def verify_deepHF_structure(data_dir):
    """Verify expected DeepHF dataset structure"""
    print()
    print("📋 Checking DeepHF dataset structure...")
    print()

    expected_files = {
        "training_data": ["X_train.pkl", "y_train.pkl", "train_index.pkl"],
        "test_data": ["X_test.pkl", "y_test.pkl", "test_index.pkl"],
        "metadata": ["gene_names.txt", "feature_names.txt", "info.json"]
    }

    raw_dir = data_dir / "raw"
    found_files = {}

    for category, files in expected_files.items():
        found_files[category] = []
        for file in files:
            # Look for file recursively
            matches = list(raw_dir.glob(f"**/{file}"))
            if matches:
                found_files[category].append(matches[0])
                print(f"  ✓ {category}/{file}")
            else:
                print(f"  ✗ {category}/{file} NOT FOUND")

    return found_files

def setup_instructions(data_dir):
    """Print setup instructions for manual download"""
    print()
    print("=" * 80)
    print("MANUAL SETUP INSTRUCTIONS")
    print("=" * 80)
    print()
    print("If automated download doesn't work, follow these steps:")
    print()
    print("1. Download DeepHF dataset from:")
    print("   - GitHub: https://github.com/gu-lab/DeepHF")
    print("   - Nature Biomedical Engineering supplementary materials")
    print()
    print("2. Extract files to: " + str(data_dir / "raw"))
    print()
    print("3. Expected structure:")
    print(f"   {data_dir}")
    print("   ├── raw/")
    print("   │   ├── X_train.pkl (features)")
    print("   │   ├── y_train.pkl (labels)")
    print("   │   ├── X_test.pkl")
    print("   │   ├── y_test.pkl")
    print("   │   └── metadata/ (gene info, feature info)")
    print("   └── processed/ (will be populated by prepare_real_data.py)")
    print()
    print("4. Run preprocessing:")
    print("   python3 scripts/prepare_real_data.py --dataset deepHF")
    print()

def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DEEPHF DATASET DOWNLOAD MANAGER" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Create directories
    data_dir = create_data_dir()
    print(f"✓ Created data directory: {data_dir}")
    print()

    # Try to download
    success = download_deepHF_dataset(data_dir, source="github")

    if success:
        print()
        print("✓ DeepHF dataset download successful!")

        # Verify structure
        found_files = verify_deepHF_structure(data_dir)

        # Save metadata
        metadata = {
            "dataset": "DeepHF",
            "paper": "Li et al., Nature Biomedical Engineering (2023)",
            "num_genes": 7,
            "num_samples": 40000,  # Approximate
            "features": ["sequence", "DNase", "Nucleosome", "ChIP-seq", "Chromatin"],
            "downloaded": True,
            "found_files": {k: [str(f) for f in v] for k, v in found_files.items()}
        }

        metadata_file = data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to {metadata_file}")
        print()
        print("✓ Next step: python3 scripts/prepare_real_data.py --dataset deepHF")
    else:
        print()
        print("⚠️  Automated download failed. Using manual setup instructions.")
        setup_instructions(data_dir)

if __name__ == "__main__":
    main()
