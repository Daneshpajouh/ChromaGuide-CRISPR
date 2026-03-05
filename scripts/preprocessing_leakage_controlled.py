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
import argparse

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

    def create_split_a(self, datasets, val_size=0.1, test_size=0.2, allow_sequence_fallback=False):
        logger.info("\n=== SPLIT A: GENE-HELD-OUT (PRIMARY) ===")
        split_a = {}
        for cell_line, df in datasets.items():
            genes = df['gene'].unique() if 'gene' in df.columns else []
            if len(genes) < 5:
                if not allow_sequence_fallback:
                    raise ValueError(
                        f"Split A requires real gene annotations. {cell_line} has only {len(genes)} unique genes."
                    )
                logger.warning(
                    f"Too few genes ({len(genes)}) in {cell_line}; using explicit sequence-fallback mode."
                )
                df = df.drop_duplicates(subset=['sequence']).sample(frac=1, random_state=self.random_seed)
                n = len(df)
                n_test = int(n * test_size)
                n_val = int(n * val_size)
                split_a[cell_line] = {
                    'train': df.iloc[n_val + n_test:],
                    'validation': df.iloc[n_test:n_test + n_val],
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

    def create_split_c(self, datasets, test_cell_line='HeLa', val_size=0.1):
        logger.info("\n=== SPLIT C: CELL-LINE-HELD-OUT ===")
        if test_cell_line not in datasets:
            raise ValueError(f"Requested held-out cell line '{test_cell_line}' not found in datasets")

        train_pool = []
        for cell_line, df in datasets.items():
            if cell_line != test_cell_line:
                train_pool.append(df.copy())
        if not train_pool:
            raise ValueError("No training cell lines available for Split C.")

        train_all = pd.concat(train_pool).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        n_val = max(1, int(len(train_all) * val_size))
        validation = train_all.iloc[:n_val].reset_index(drop=True)
        train = train_all.iloc[n_val:].reset_index(drop=True)
        test = datasets[test_cell_line].copy().reset_index(drop=True)

        split_c = {
            test_cell_line: {
                'train': train,
                'validation': validation,
                'test': test,
            }
        }
        logger.info(
            f"  Held-out={test_cell_line}: Train={len(train)}, Val={len(validation)}, Test={len(test)}"
        )
        self.splits['split_c'] = split_c
        return split_c

    def save_splits(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_dir_name = {
            'split_a': 'split_a_gene_held_out',
            'split_b': 'split_b_dataset_held_out',
            'split_c': 'split_c_cellline_held_out',
        }
        for split_name, split_data in self.splits.items():
            split_dir = output_dir / split_dir_name.get(split_name, split_name)
            split_dir.mkdir(parents=True, exist_ok=True)
            for cell_line, subsets in split_data.items():
                for subset_name, df in subsets.items():
                    path = split_dir / f"{cell_line}_{subset_name}.csv"
                    df.to_csv(path, index=False)
            with open(split_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "split_name": split_name,
                        "random_seed": self.random_seed,
                        "files": sorted([p.name for p in split_dir.glob("*.csv")]),
                    },
                    f,
                    indent=2,
                )
        logger.info(f"✓ Splits saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create leakage-controlled splits")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--test_cell_line", type=str, default="HeLa")
    parser.add_argument("--allow_sequence_fallback", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    splitter = LeakageControlledSplitter(data_dir, random_seed=args.random_seed)
    datasets = splitter.load_raw_data()
    splitter.create_split_a(datasets, allow_sequence_fallback=args.allow_sequence_fallback)
    splitter.create_split_b(datasets)
    splitter.create_split_c(datasets, test_cell_line=args.test_cell_line)
    splitter.save_splits(output_dir)

if __name__ == "__main__":
    main()
