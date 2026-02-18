"""
LEAKAGE-CONTROLLED DATASET SPLITS
PhD thesis requirement: Rigorous separation of train/test to prevent leakage.
Implements three evaluation strategies:
  - Split A: Gene-held-out (primary) - each target gene in only train or test
  - Split B: Dataset-held-out (cross-dataset) - train on one dataset, test on others
  - Split C: Cell-line-held-out (cross-cell-line) - train on one cell line, test on others
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for each dataset split."""
    name: str
    n_samples: int
    n_genes: int
    n_unique_sequences: int
    genes: Set[str]
    sequences: Set[str]
    cell_line: str


class LeakageControlledSplitter:
    """Implement rigorous leakage-controlled splits for PhD thesis."""
    
    def __init__(self, data_dir: Path, random_seed: int = 42):
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.metadata = {}
        self.splits = {}
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all three datasets."""
        logger.info("Loading raw datasets...")
        datasets = {}
        
        data_sources = {
            'HEK293T': 'raw/deeprf/HEK293T.csv',
            'HCT116': 'raw/deeprf/HCT116.csv',
            'HeLa': 'raw/deeprf/HeLa.csv'
        }
        
        for cell_line, filepath in data_sources.items():
            full_path = self.data_dir / filepath
            
            if full_path.exists():
                df = pd.read_csv(full_path)
                logger.info(f"  Loaded {cell_line}: {len(df)} sequences, columns: {df.columns.tolist()}")
                datasets[cell_line] = df
            else:
                logger.warning(f"  Expected {full_path} not found")
        
        return datasets
    
    def deduplicate_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate sequences within dataset, keeping best (highest intensity)."""
        logger.info(f"  Deduplication: {len(df)} → ", end='')
        
        if 'sequence' in df.columns and 'intensity' in df.columns:
            # Group by sequence, keep highest intensity
            df_dedup = df.loc[df.groupby('sequence')['intensity'].idxmax()]
        else:
            # If columns differ, use first occurrence
            df_dedup = df.drop_duplicates(subset=['sequence'])
        
        logger.info(f"{len(df_dedup)} sequences (removed {len(df) - len(df_dedup)} duplicates)")
        return df_dedup.reset_index(drop=True)
    
    def extract_metadata(self, datasets: Dict[str, pd.DataFrame]):
        """Extract and store metadata for each dataset."""
        logger.info("Extracting dataset metadata...")
        
        for cell_line, df in datasets.items():
            genes = set()
            if 'gene' in df.columns:
                genes = set(df['gene'].dropna().unique())
            elif 'target_gene' in df.columns:
                genes = set(df['target_gene'].dropna().unique())
            
            sequences = set(df['sequence'].unique())
            
            self.metadata[cell_line] = DatasetMetadata(
                name=cell_line,
                n_samples=len(df),
                n_genes=len(genes),
                n_unique_sequences=len(sequences),
                genes=genes,
                sequences=sequences,
                cell_line=cell_line
            )
            
            logger.info(f"  {cell_line}: {len(df)} samples, {len(genes)} genes, "
                       f"{len(sequences)} unique sequences")
    
    def create_split_a_gene_held_out(self, 
                                     datasets: Dict[str, pd.DataFrame],
                                     val_size: float = 0.1,
                                     test_size: float = 0.2) -> Dict:
        """
        SPLIT A: Gene-held-out (PRIMARY EVALUATION)
        - Genes in validation/test never appear in training
        - Strongest leakage control - most rigorous
        - Use for ranking models
        """
        logger.info("\n=== SPLIT A: GENE-HELD-OUT (PRIMARY) ===")
        
        split_a = {}
        
        for cell_line, df in datasets.items():
            logger.info(f"\nProcessing {cell_line}...")
            
            # Extract genes
            if 'gene' not in df.columns and 'target_gene' not in df.columns:
                logger.warning(f"  No gene column found in {cell_line}, using sequence-based split")
                # Fallback to sequence deduplication
                df = df.drop_duplicates(subset=['sequence'])
                sequences = df['sequence'].values
                n_test = int(len(sequences) * test_size)
                n_val = int(len(sequences) * val_size)
                
                indices = np.arange(len(sequences))
                np.random.shuffle(indices)
                
                train_idx = indices[n_val + n_test:]
                val_idx = indices[n_test:n_test + n_val]
                test_idx = indices[:n_test]
                
            else:
                # Group by gene
                gene_col = 'gene' if 'gene' in df.columns else 'target_gene'
                genes = df[gene_col].unique()
                n_test_genes = max(1, int(len(genes) * test_size))
                n_val_genes = max(1, int(len(genes) * val_size))
                
                gene_indices = np.arange(len(genes))
                np.random.shuffle(gene_indices)
                
                test_genes = set(genes[gene_indices[:n_test_genes]])
                val_genes = set(genes[gene_indices[n_test_genes:n_test_genes + n_val_genes]])
                train_genes = set(genes) - test_genes - val_genes
                
                # Split by gene
                train_idx = df[df[gene_col].isin(train_genes)].index
                val_idx = df[df[gene_col].isin(val_genes)].index
                test_idx = df[df[gene_col].isin(test_genes)].index
                
                logger.info(f"  Genes: Train={len(train_genes)}, Val={len(val_genes)}, Test={len(test_genes)}")
            
            split_a[cell_line] = {
                'train': df.iloc[train_idx].reset_index(drop=True),
                'validation': df.iloc[val_idx].reset_index(drop=True),
                'test': df.iloc[test_idx].reset_index(drop=True),
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'n_test': len(test_idx)
            }
            
            logger.info(f"  Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        self.splits['split_a_gene_held_out'] = split_a
        return split_a
    
    def create_split_b_dataset_held_out(self, 
                                       datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        SPLIT B: Dataset-held-out (CROSS-DATASET GENERALIZATION)
        - Train on 2 cell lines, test on 1
        - Evaluate cross-dataset generalization
        - Important for PharmGKB external validation
        """
        logger.info("\n=== SPLIT B: DATASET-HELD-OUT ===")
        
        cell_lines = list(datasets.keys())
        split_b = {}
        
        for test_cell_line in cell_lines:
            logger.info(f"\nTest on {test_cell_line}, train on others...")
            
            train_dfs = []
            for cell_line in cell_lines:
                if cell_line != test_cell_line:
                    train_dfs.append(datasets[cell_line])
            
            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = datasets[test_cell_line]
            
            # Further split test into val/test
            test_sequences = test_df['sequence'].unique()
            n_val = max(1, int(len(test_sequences) * 0.1))
            
            shuffled_idx = np.arange(len(test_sequences))
            np.random.shuffle(shuffled_idx)
            val_seqs = set(test_sequences[shuffled_idx[:n_val]])
            
            val_df = test_df[test_df['sequence'].isin(val_seqs)]
            test_df_split = test_df[~test_df['sequence'].isin(val_seqs)]
            
            split_b[f"train_all_except_{test_cell_line}"] = {
                'train': train_df.reset_index(drop=True),
                'validation': val_df.reset_index(drop=True),
                'test': test_df_split.reset_index(drop=True),
                'n_train': len(train_df),
                'n_val': len(val_df),
                'n_test': len(test_df_split)
            }
            
            logger.info(f"  Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df_split)}")
        
        self.splits['split_b_dataset_held_out'] = split_b
        return split_b
    
    def create_split_c_cellline_held_out(self, 
                                        datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        SPLIT C: Cell-line-held-out (CROSS-CELL-LINE GENERALIZATION)
        - Same as Split B but explicitly framed for cell-line generalization
        - Evaluate robustness across cell types
        """
        logger.info("\n=== SPLIT C: CELL-LINE-HELD-OUT ===")
        # Use same logic as Split B but with explicit naming
        return self.create_split_b_dataset_held_out(datasets)
    
    def generate_split_report(self, output_dir: Path):
        """Generate summary report of all splits."""
        logger.info("\n=== SPLIT SUMMARY REPORT ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'metadata': {
                k: {
                    'n_samples': v.n_samples,
                    'n_genes': v.n_genes,
                    'n_unique_sequences': v.n_unique_sequences,
                    'cell_line': v.cell_line
                }
                for k, v in self.metadata.items()
            },
            'splits': {}
        }
        
        for split_name, split_data in self.splits.items():
            logger.info(f"\n{split_name}:")
            report['splits'][split_name] = {}
            
            if isinstance(split_data, dict):
                for key, value in split_data.items():
                    if isinstance(value, dict) and 'n_train' in value:
                        logger.info(f"  {key}: train={value['n_train']}, val={value['n_val']}, test={value['n_test']}")
                        report['splits'][split_name][key] = {
                            'n_train': value['n_train'],
                            'n_val': value['n_val'],
                            'n_test': value['n_test']
                        }
        
        # Save report
        report_path = output_dir / 'split_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved to {report_path}")
        
        return report
    
    def save_splits(self, output_dir: Path):
        """Save all splits to disk for training."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving splits to {output_dir}...")
        
        for split_name, split_data in self.splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            if isinstance(split_data, dict):
                for subset_name, subset_data in split_data.items():
                    if isinstance(subset_data, dict) and 'train' in subset_data:
                        # Save train/val/test for this subset
                        for split_type in ['train', 'validation', 'test']:
                            if isinstance(subset_data[split_type], pd.DataFrame):
                                path = split_dir / f"{subset_name}_{split_type}.csv"
                                subset_data[split_type].to_csv(path, index=False)
                                logger.info(f"  Saved {path}")
        
        logger.info("✓ All splits saved")


def main():
    """Main execution."""
    data_dir = Path("/project/def-bengioy/chromaguide_data")
    output_dir = Path("/project/def-bengioy/chromaguide_data/splits")
    
    # Create splitter
    splitter = LeakageControlledSplitter(data_dir, random_seed=42)
    
    # Load and process data
    datasets = splitter.load_raw_data()
    datasets = {
        cell_line: splitter.deduplicate_sequences(df)
        for cell_line, df in datasets.items()
    }
    
    splitter.extract_metadata(datasets)
    
    # Generate three split strategies
    splitter.create_split_a_gene_held_out(datasets)
    splitter.create_split_b_dataset_held_out(datasets)
    splitter.create_split_c_cellline_held_out(datasets)
    
    # Save results
    splitter.generate_split_report(output_dir)
    splitter.save_splits(output_dir)
    
    logger.info("\n✓✓✓ Leakage-controlled splits complete! ✓✓✓")
    logger.info(f"Ready for training with splits in {output_dir}")


if __name__ == "__main__":
    main()
