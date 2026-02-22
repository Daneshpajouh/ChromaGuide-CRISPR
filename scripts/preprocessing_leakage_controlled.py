"""
LEAKAGE-CONTROLLED DATASET SPLITS
PhD thesis requirement: Rigorous separation of train/test to prevent leakage.
Split A: Gene-held-out (primary) - each target gene in only train or test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeakageControlledSplitter:
    def __init__(self, data_dir: Path, random_seed: int = 42):
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.splits = {}

    def load_raw_data(self):
        logger.info("Loading raw datasets...")
        datasets = {}
        data_sources = {
            'HEK293T': 'DeepHF_HEK293T.csv',
            'HCT116': 'DeepHF_HCT116.csv',
            'HeLa': 'DeepHF_HeLa.csv'
        }
        raw_dir = self.data_dir / "deepHF" / "raw" / "deepHF"
        for cell_line, filename in data_sources.items():
            full_path = raw_dir / filename
            if full_path.exists():
                df = pd.read_csv(full_path)
                logger.info(f"  Loaded {cell_line}: {len(df)} samples")
                datasets[cell_line] = df
            else:
                logger.warning(f"  Expected {full_path} not found")
        return datasets

    def create_split_a(self, datasets, val_size=0.1, test_size=0.2):
        logger.info("\n=== SPLIT A: GENE-HELD-OUT (PRIMARY) ===")
        split_a = {}
        for cell_line, df in datasets.items():
            genes = df['gene'].unique() if 'gene' in df.columns else []
            if len(genes) < 5:
                logger.warning(f"Too few genes ({len(genes)}) in {cell_line}, using sequence split instead")
                df = df.drop_duplicates(subset=['sequence']).sample(frac=1, random_state=self.random_seed)
                n = len(df)
                n_test = int(n * test_size)
                n_val = int(n * val_size)
                split_a[cell_line] = {
                    'train': df.iloc[n_val+n_test:],
                    'validation': df.iloc[n_test:n_test+n_val],
                    'test': df.iloc[:n_test]
                }
            else:
                np.random.shuffle(genes)
                n_test = max(1, int(len(genes) * test_size))
                n_val = max(1, int(len(genes) * val_size))
                test_genes = set(genes[:n_test])
                val_genes = set(genes[n_test:n_test+n_val])
                train_genes = set(genes[n_test+n_val:])

                split_a[cell_line] = {
                    'train': df[df['gene'].isin(train_genes)],
                    'validation': df[df['gene'].isin(val_genes)],
                    'test': df[df['gene'].isin(test_genes)]
                }
            logger.info(f"  {cell_line}: Train={len(split_a[cell_line]['train'])}, Val={len(split_a[cell_line]['validation'])}, Test={len(split_a[cell_line]['test'])}")
        self.splits['split_a'] = split_a
        return split_a

    def create_split_b(self, datasets, val_size=0.1, test_size=0.2):
        logger.info("\n=== SPLIT B: RANDOM SEQUENCE SPLIT ===")
        split_b = {}
        for cell_line, df in datasets.items():
            df = df.sample(frac=1, random_state=self.random_seed + 1)
            n = len(df)
            n_test = int(n * test_size)
            n_val = int(n * val_size)
            split_b[cell_line] = {
                'train': df.iloc[n_val+n_test:],
                'validation': df.iloc[n_test:n_test+n_val],
                'test': df.iloc[:n_test]
            }
            logger.info(f"  {cell_line}: Train={len(split_b[cell_line]['train'])}, Val={len(split_b[cell_line]['validation'])}, Test={len(split_b[cell_line]['test'])}")
        self.splits['split_b'] = split_b
        return split_b

    def save_splits(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in self.splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for cell_line, subsets in split_data.items():
                for subset_name, df in subsets.items():
                    path = split_dir / f"{cell_line}_{subset_name}.csv"
                    df.to_csv(path, index=False)
        logger.info(f"✓ Splits saved to {output_dir}")

def main():
    data_dir = Path("data")
    output_dir = Path("data/processed")
    splitter = LeakageControlledSplitter(data_dir)
    datasets = splitter.load_raw_data()
    splitter.create_split_a(datasets)
    splitter.create_split_b(datasets)
    splitter.save_splits(output_dir)

if __name__ == "__main__":
    main()
