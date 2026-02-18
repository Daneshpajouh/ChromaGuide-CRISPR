
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
from src.model.crispr_rag_tta import CRISPR_RAG_Head

class Wasserstein_RAG_Head(CRISPR_RAG_Head):
    """
    RAG Head using Optimal Transport (Wasserstein Distance) for retrieval.
    Replaces Euclidean L2 distance with Earth Mover's Distance approximation.
    """
    def __init__(self, d_model, k_neighbors=50, epsilon=0.1, max_iter=50):
        super().__init__(d_model, k_neighbors)
        self.epsilon = epsilon  # Regularization for Sinkhorn
        self.max_iter = max_iter

    def retrieve_neighbors(self, query_embeddings):
        """
        Retrieval using 1D Wasserstein Approximation (Sliced Wasserstein)
        or Sinkhorn logic.

        For speed in RAG (querying 150k items), we use the Sliced Wasserstein
        approximation: sorting features and computing L1 distance between sorted vectors.
        This approximates the cost of transporting the mass of one embedding distribution
        to another.
        """
        batch_size = query_embeddings.size(0)
        memory_size = self.memory_keys.size(0)

        # Sliced Wasserstein Approximation (Sort dimensions)
        # 1. Sort query and keys along feature dimension
        # (batch, d_model) -> (batch, d_model)
        query_sorted, _ = torch.sort(query_embeddings, dim=1)

        # (memory, d_model) -> (memory, d_model)
        # Note: In production, memory_keys usually pre-sorted, but we do it here for compatibility
        keys_sorted, _ = torch.sort(self.memory_keys, dim=1)

        # 2. Complete L1 distance between sorted vectors
        # dist_matrix: (batch, memory)
        # We need efficient broadcast.
        # unsqueeze: (batch, 1, d_model) - (1, memory, d_model)

        # Optimization: Don't compute full matrix if memory is huge.
        # But for 150k, torch can handle it on GPU.
        # Let's try PyTorch cdist with p=1 on sorted vectors.
        # This IS the 1D Wasserstein distance between the feature distributions.

        dists = torch.cdist(query_sorted, keys_sorted, p=1)

        # 3. Top-k
        distances, indices = torch.topk(dists, k=self.k, largest=False)

        neighbor_keys = self.memory_keys[indices]
        neighbor_vals = self.memory_values[indices]

        return neighbor_keys, neighbor_vals, distances

def test_wasserstein():
    print("Testing Wasserstein RAG Head...")
    d_model = 128
    head = Wasserstein_RAG_Head(d_model, k_neighbors=5)

    # Mock data
    train = torch.randn(100, d_model)
    labels = torch.randn(100, 1)
    head.initialize_memory(train, labels)

    # Query
    q = torch.randn(2, d_model)
    keys, vals, dists = head.retrieve_neighbors(q)

    print(f"Query shape: {q.shape}")
    print(f"Neighbors shape: {keys.shape}")
    print(f"Distances: {dists[0]}")
    print("✓ Wasserstein Retrieval successful (1D Approx)")

if __name__ == "__main__":
    test_wasserstein()
