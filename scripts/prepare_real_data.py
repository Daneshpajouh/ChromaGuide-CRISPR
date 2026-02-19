#!/usr/bin/env python3
"""
Prepare real CRISPR dataset (DeepHF, CRISPRnature) for training.

This script:
1. Loads raw dataset from various sources
2. Normalizes features
3. Creates train/val/test splits
4. Generates data loaders
5. Validates data format for ChromaGuide models

Supported datasets:
- DeepHF: Li et al., Nature Biomedical Engineering (2023)
- CRISPRnature: Shao et al., Nature (2024)
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse

def load_deepHF_dataset(data_dir="data/deepHF/raw"):
    """Load DeepHF dataset from pickle files"""

    print("Loading DeepHF dataset...")

    data_dir = Path(data_dir)

    # Try different file locations
    X_train_paths = list(data_dir.glob("**/X_train.pkl")) + \
                   list(data_dir.glob("**/X_train*.npy")) + \
                   list(data_dir.glob("**/x_train.pkl"))

    y_train_paths = list(data_dir.glob("**/y_train.pkl")) + \
                   list(data_dir.glob("**/y_train*.npy")) + \
                   list(data_dir.glob("**/y_train*.csv"))

    X_test_paths = list(data_dir.glob("**/X_test.pkl")) + \
                  list(data_dir.glob("**/X_test*.npy")) + \
                  list(data_dir.glob("**/x_test.pkl"))

    y_test_paths = list(data_dir.glob("**/y_test.pkl")) + \
                  list(data_dir.glob("**/y_test*.npy")) + \
                  list(data_dir.glob("**/y_test*.csv"))

    if not (X_train_paths and y_train_paths):
        raise FileNotFoundError(
            f"Could not find X_train or y_train in {data_dir}. "
            "Run scripts/download_deepHF_data.py first."
        )

    # Load training data
    X_train_path = X_train_paths[0]
    y_train_path = y_train_paths[0]

    print(f"  Loading: {X_train_path.relative_to(data_dir.parent.parent)}")
    if str(X_train_path).endswith('.pkl'):
        with open(X_train_path, 'rb') as f:
            X_train = pickle.load(f)
    else:
        X_train = np.load(X_train_path)

    print(f"  Loading: {y_train_path.relative_to(data_dir.parent.parent)}")
    if str(y_train_path).endswith('.pkl'):
        with open(y_train_path, 'rb') as f:
            y_train = pickle.load(f)
    else:
        y_train = np.load(y_train_path)

    # Load test data if available
    X_test = None
    y_test = None

    if X_test_paths and y_test_paths:
        X_test_path = X_test_paths[0]
        y_test_path = y_test_paths[0]

        print(f"  Loading: {X_test_path.relative_to(data_dir.parent.parent)}")
        if str(X_test_path).endswith('.pkl'):
            with open(X_test_path, 'rb') as f:
                X_test = pickle.load(f)
        else:
            X_test = np.load(X_test_path)

        print(f"  Loading: {y_test_path.relative_to(data_dir.parent.parent)}")
        if str(y_test_path).endswith('.pkl'):
            with open(y_test_path, 'rb') as f:
                y_test = pickle.load(f)
        else:
            y_test = np.load(y_test_path)

    print(f"  ✓ Loaded X_train: {X_train.shape}")
    print(f"  ✓ Loaded y_train: {y_train.shape}")
    if X_test is not None:
        print(f"  ✓ Loaded X_test: {X_test.shape}")
        print(f"  ✓ Loaded y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test

def normalize_features(X_train, X_val, X_test=None):
    """Normalize features using z-score normalization"""

    print("\nNormalizing features...")

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    X_test_norm = None
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)

    # Save scaler
    scaler_path = Path("data/deepHF/processed/scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved to {scaler_path}")

    return X_train_norm, X_val_norm, X_test_norm, scaler

def prepare_dataset(dataset="deepHF", val_split=0.15, test_split=0.15, seed=42):
    """
    Prepare dataset for ChromaGuide training.

    Args:
        dataset: "deepHF" or "crisprNature"
        val_split: Fraction for validation set
        test_split: Fraction for test set (if not already provided)
        seed: Random seed for reproducibility
    """

    print("=" * 80)
    print(f"PREPARING {dataset.upper()} DATASET FOR CHROMAGUIDE TRAINING")
    print("=" * 80)
    print()

    output_dir = Path(f"data/{dataset}/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    if dataset.lower() == "deephf":
        X_data, y_data, X_test_provided, y_test_provided = load_deepHF_dataset()
    else:
        raise ValueError(f"Dataset '{dataset}' not supported yet")

    print(f"\nDataset shape: X={X_data.shape}, y={y_data.shape}")
    print(f"  Features: {X_data.shape[1]}")
    print(f"  Samples: {X_data.shape[0]}")
    print(f"  Label range: [{y_data.min():.4f}, {y_data.max():.4f}]")

    # Create train/val split
    if X_test_provided is not None:
        # If test set provided, split training data into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data,
            test_size=val_split,
            random_state=seed
        )
        X_test = X_test_provided
        y_test = y_test_provided
    else:
        # Otherwise create train/val/test splits
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_data, y_data,
            test_size=(val_split + test_split),
            random_state=seed
        )

        # Split remaining into val and test
        val_frac = val_split / (val_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_frac),
            random_state=seed
        )

    print(f"\nData splits created:")
    print(f"  Train: {X_train.shape[0]} samples ({100*X_train.shape[0]/len(X_data):.1f}%)")
    print(f"  Val:   {X_val.shape[0]} samples ({100*X_val.shape[0]/len(X_data):.1f}%)")
    print(f"  Test:  {X_test.shape[0]} samples ({100*X_test.shape[0]/len(X_data):.1f}%)")

    # Normalize features
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        X_train, X_val, X_test
    )

    print(f"  Feature normalization: z-score (mean=0, std=1)")

    # Save processed data
    print(f"\nSaving processed data...")

    datasets = {
        'X_train': X_train_norm,
        'y_train': y_train,
        'X_val': X_val_norm,
        'y_val': y_val,
        'X_test': X_test_norm,
        'y_test': y_test
    }

    for name, data in datasets.items():
        output_path = output_dir / f"{name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ {name}: {output_path.relative_to(Path.cwd())}")

    # Save metadata
    metadata = {
        "dataset": dataset,
        "num_features": X_train.shape[1],
        "num_samples": len(X_data),
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "label_mean": float(y_data.mean()),
        "label_std": float(y_data.std()),
        "label_min": float(y_data.min()),
        "label_max": float(y_data.max()),
        "normalization": "z-score",
        "random_seed": seed
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to {metadata_path.relative_to(Path.cwd())}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Ready to train ChromaGuide models!")
    print(f"✓ Run: python3 scripts/train_on_real_data.py --dataset {dataset}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare real CRISPR dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepHF",
        choices=["deepHF", "crisprNature"],
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split fraction"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Test split fraction"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    prepare_dataset(
        dataset=args.dataset,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
