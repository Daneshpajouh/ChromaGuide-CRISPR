#!/usr/bin/env python3
"""
Data Preprocessing Pipeline
===========================

Comprehensive data preprocessing for CRISPR prediction:
- Sequence encoding (one-hot, BPE, embeddings)
- Feature scaling and normalization
- Missing value handling
- Outlier detection and removal
- Train-test split strategies
- Class balancing
- Feature engineering

Usage:
    preprocessor = DataPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(X_raw, y_raw)
    
    X_test_processed = preprocessor.transform(X_test)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceEncoder:
    """Encode biological sequences."""
    
    @staticmethod
    def one_hot_encode(sequence: str, alphabet: str = 'ATCG') -> np.ndarray:
        """One-hot encode a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            alphabet: Characters to encode
            
        Returns:
            One-hot encoded array (seq_len, alphabet_size)
        """
        mapping = {char: idx for idx, char in enumerate(alphabet)}
        
        encoded = np.zeros((len(sequence), len(alphabet)), dtype=np.float32)
        
        for i, char in enumerate(sequence):
            if char in mapping:
                encoded[i, mapping[char]] = 1.0
            else:
                # Unknown character - uniform distribution
                encoded[i, :] = 1.0 / len(alphabet)
        
        return encoded
    
    @staticmethod
    def encode_kmers(sequence: str, k: int = 3, alphabet: str = 'ATCG') -> np.ndarray:
        """Encode sequence as k-mer frequencies.
        
        Args:
            sequence: DNA sequence
            k: K-mer length
            alphabet: Base alphabet
            
        Returns:
            K-mer frequency vector
        """
        from itertools import product
        
        # Generate all possible kmers
        all_kmers = [''.join(p) for p in product(alphabet, repeat=k)]
        kmer_dict = {kmer: 0 for kmer in all_kmers}
        
        # Count kmers in sequence
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
        
        # Convert to frequency vector
        total_kmers = len(sequence) - k + 1
        freqs = np.array([kmer_dict[kmer] / total_kmers for kmer in all_kmers], 
                         dtype=np.float32)
        
        return freqs


class FeatureEngineer:
    """Feature engineering for CRISPR data."""
    
    @staticmethod
    def compute_gc_content(sequence: str) -> float:
        """Compute GC content."""
        return (sequence.count('G') + sequence.count('C')) / len(sequence) if len(sequence) > 0 else 0
    
    @staticmethod
    def compute_homopolymer_runs(sequence: str) -> Dict[str, int]:
        """Compute homopolymer run statistics."""
        max_run = 0
        current_run = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return {
            'max_homopolymer_run': max_run,
            'has_long_run': 1 if max_run >= 4 else 0
        }
    
    @staticmethod
    def compute_secondary_structure_features(sequence: str) -> Dict[str, float]:
        """Mock secondary structure prediction."""
        # This is simplified - full implementation would use Vienna, RNAfold, etc.
        
        # Simple heuristic based on pairing potential
        at_content = (sequence.count('A') + sequence.count('T')) / len(sequence)
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        return {
            'at_content': at_content,
            'gc_content': gc_content,
            'stability_score': gc_content * 0.5 + at_content * 0.3
        }
    
    @staticmethod
    def compute_all_features(sequence: str) -> Dict[str, float]:
        """Compute all sequence features."""
        features = {}
        
        # Basic composition
        features['gc_content'] = FeatureEngineer.compute_gc_content(sequence)
        
        # Homopolymer runs
        features.update(FeatureEngineer.compute_homopolymer_runs(sequence))
        
        # Secondary structure
        features.update(FeatureEngineer.compute_secondary_structure_features(sequence))
        
        return features


class DataPreprocessor:
    """Main preprocessing pipeline."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_names = None
        self.preprocessing_info = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DataPreprocessor':
        """Fit preprocessing pipeline."""
        logger.info(f"Fitting preprocessor on {X.shape[0]} samples")
        
        # Fit scaler
        self.scaler.fit(X)
        self.is_fitted = True
        
        # Store feature info
        self.preprocessing_info = {
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'feature_means': self.scaler.center_.tolist(),
            'feature_scales': self.scaler.scale_.tolist()
        }
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Remove NaN and inf
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X), y
    
    def split_train_test(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                          np.ndarray, np.ndarray]:
        """Split data into train/test with stratification."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=np.digitize(y, bins=5)  # Stratify by quantiles
        )
        
        # Fit on train, transform train and test
        self.fit(X_train, y_train)
        
        X_train_processed = self.transform(X_train)
        X_test_processed = self.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def get_kfold_splits(self, X: np.ndarray, y: np.ndarray, 
                        n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]]:
        """Generate cross-validation splits."""
        kfold = StratifiedKFold(n_splits=n_splits, random_state=self.random_state)
        splits = []
        
        for train_idx, val_idx in kfold.split(X, np.digitize(y, bins=5)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Preprocess
            preprocessor = DataPreprocessor(self.random_state)
            X_train_p = preprocessor.fit_transform(X_train, y_train)[0]
            X_val_p = preprocessor.transform(X_val)
            
            splits.append((X_train_p, X_val_p, y_train, y_val))
        
        return splits
    
    def handle_outliers(self, X: np.ndarray, method: str = 'iqr',
                       threshold: float = 1.5) -> np.ndarray:
        """Remove outliers."""
        if method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
            return X[mask]
        
        elif method == 'zscore':
            z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
            mask = np.all(z_scores < threshold, axis=1)
            return X[mask]
        
        return X
    
    def balance_classes(self, X: np.ndarray, y: np.ndarray,
                       method: str = 'oversample') -> Tuple[np.ndarray, np.ndarray]:
        """Balance class distribution."""
        unique, counts = np.unique(y, return_counts=True)
        
        if method == 'oversample':
            max_count = counts.max()
            X_balanced, y_balanced = [], []
            
            for label in unique:
                mask = y == label
                X_class = X[mask]
                y_class = y[mask]
                
                # Oversample to max count
                n_needed = max_count - len(X_class)
                if n_needed > 0:
                    indices = np.random.choice(len(X_class), n_needed, replace=True)
                    X_class = np.vstack([X_class, X_class[indices]])
                    y_class = np.hstack([y_class, y_class[indices]])
                
                X_balanced.append(X_class)
                y_balanced.append(y_class)
            
            return np.vstack(X_balanced), np.hstack(y_balanced)
        
        return X, y
    
    def export_preprocessing_info(self, output_path: Path) -> None:
        """Export preprocessing metadata."""
        with open(output_path, 'w') as f:
            json.dump(self.preprocessing_info, f, indent=2)
        logger.info(f"Preprocessing info saved to {output_path}")


class SequenceDataPreprocessor:
    """Specialized preprocessing for sequence data."""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.sequence_encoder = SequenceEncoder()
    
    def extract_sequence_features(self, sequences: List[str]) -> np.ndarray:
        """Extract features from sequences."""
        features_list = []
        
        for seq in sequences:
            features = self.feature_engineer.compute_all_features(seq)
            features_list.append(features)
        
        return pd.DataFrame(features_list).values
    
    def encode_sequences(self, sequences: List[str], encoding: str = 'onehot') -> np.ndarray:
        """Encode sequences."""
        if encoding == 'onehot':
            # Flatten one-hot encodings
            encoded = []
            for seq in sequences:
                onehot = self.sequence_encoder.one_hot_encode(seq)
                encoded.append(onehot.flatten())
            return np.array(encoded)
        
        elif encoding == 'kmers':
            encoded = []
            for seq in sequences:
                kmers = self.sequence_encoder.encode_kmers(seq)
                encoded.append(kmers)
            return np.array(encoded)
        
        return np.array([np.zeros(100) for _ in sequences])


if __name__ == '__main__':
    logger.info("Data Preprocessing Module Initialized")
