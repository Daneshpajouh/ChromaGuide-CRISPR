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
    Download DeepHF dataset from available sources.
    
    Sources:
    1. GitHub (primary): https://github.com/gu-lab/DeepHF
    2. Zenodo (backup): https://zenodo.org/record/... (if available)
    3. Supplementary materials: Nature Biomedical Engineering
    """
    
    print("=" * 80)
    print("DEEPHF DATASET DOWNLOAD")
    print("=" * 80)
    print()
    
    raw_dir = data_dir / "raw"
    
    # Option 1: GitHub repository
    if source == "github":
        print("📥 Attempting to clone DeepHF repository from GitHub...")
        print("   Repository: https://github.com/gu-lab/DeepHF")
        print()
        
        github_url = "https://github.com/gu-lab/DeepHF.git"
        repo_path = raw_dir / "DeepHF_repo"
        
        try:
            import subprocess
            result = subprocess.run(
                ["git", "clone", github_url, str(repo_path)],
                capture_output=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"✓ Repository cloned to {repo_path}")
                
                # Find dataset files
                data_files = list(repo_path.glob("**/*.pkl"))
                data_files += list(repo_path.glob("**/*.pickle"))
                data_files += list(repo_path.glob("**/*.npy"))
                data_files += list(repo_path.glob("**/*.csv"))
                
                if data_files:
                    print(f"✓ Found {len(data_files)} data files")
                    for f in data_files[:5]:
                        print(f"   - {f.relative_to(repo_path)}")
                    return True
                else:
                    print("⚠️  No data files found in repository")
                    return False
            else:
                print(f"✗ Git clone failed: {result.stderr.decode()}")
                return False
                
        except FileNotFoundError:
            print("✗ Git not found. Install with: brew install git")
            return False
        except Exception as e:
            print(f"✗ Error cloning repository: {e}")
            return False
    
    # Option 2: Direct download from data repository
    elif source == "direct":
        print("📥 Attempting direct download of DeepHF data...")
        print()
        
        # Example URLs (update with actual DeepHF dataset URLs)
        dataset_urls = {
            "DeepHF_training": "https://...",  # Update with actual URL
            "DeepHF_test": "https://...",      # Update with actual URL
        }
        
        print("⚠️  Please provide direct download URLs for DeepHF dataset")
        print("   These are typically provided in the paper's supplementary materials")
        return False
    
    else:
        print(f"✗ Unknown source: {source}")
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
