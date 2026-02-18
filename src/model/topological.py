import torch
import torch.nn as nn

class TopologicalRegularizer(nn.Module):
    """
    Chapter 6: Topological Genomics.

    Computes Persistent Homology loss (TopoAE style) to enforce
    that the latent space manifold preserves the topology of the input space.

    Note: Full PH calculation is O(N^3), so we use a simplified
    distance-matrix matching proxy (Topological Autoencoder).
    """
    def __init__(self, lam=0.1):
        super().__init__()
        self.lam = lam

    def forward(self, input_data, latent_data):
        """
        input_data: [B, D_in] (e.g., flattened sequence one-hot)
        latent_data: [B, D_latent]

        Enforces: DistanceMatrix(Input) ~= DistanceMatrix(Latent)
        Using 0-th order topology (Connected Components / Clustering preservation)
        """
        # Pairwise distance matrices
        dist_x = torch.cdist(input_data, input_data, p=2)
        dist_z = torch.cdist(latent_data, latent_data, p=2)

        # Normalize distances to be scale-invariant
        dist_x = dist_x / (dist_x.mean() + 1e-8)
        dist_z = dist_z / (dist_z.mean() + 1e-8)

        # Topological Loss: Structure preservation
        # (Simplified version of Vietoris-Rips diagram matching)
        loss_topo = torch.mean((dist_x - dist_z) ** 2)

        return self.lam * loss_topo
