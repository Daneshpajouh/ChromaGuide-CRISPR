import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
import torch
import os
import sys

# Assume we run this after generating predictions.
class NucleomerExtractor:
    def __init__(self, max_depth=3):
        self.tree = DecisionTreeRegressor(max_depth=max_depth)
        self.feature_names = None

    def fit(self, sequences, scores):
        """
        sequences: List of strings (e.g. 23bp)
        scores: List of floats (predictions)
        """
        # 1. Featurize Sequences (One-Hot Flattened)
        # Features: Pos{i}_{NT}
        X = []
        self.feature_names = []
        bases = ['A', 'C', 'G', 'T']

        # Generate feature names once
        seq_len = len(sequences[0])
        for i in range(seq_len):
            for b in bases:
                self.feature_names.append(f"Pos{i+1}_{b}")

        for seq in sequences:
            # Flatten one-hot
            vec = []
            for i, char in enumerate(seq):
                for b in bases:
                    vec.append(1 if char == b else 0)
            X.append(vec)

        X = np.array(X)
        y = np.array(scores)

        # 2. Train Tree
        self.tree.fit(X, y)
        print("Surrogate Tree Trained.")

    def export_rules(self):
        if not self.feature_names:
            return "Tree not trained."
        r = export_text(self.tree, feature_names=self.feature_names)
        return r

if __name__ == "__main__":
    # Test
    seqs = ["ACGT"*5 + "AAA", "TGCA"*5 + "TTT"] # 23bp approx
    scores = [0.9, 0.1]
    extractor = NucleomerExtractor(max_depth=3)
    extractor.fit(seqs, scores)
    print(extractor.export_rules())
