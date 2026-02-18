import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

class MechanisticProbes:
    """
    Chapter 9: The Conscious Microscope.

    Tools for 'Circuit Discovery' in Mamba models.
    Identifies which latent dimensions encode specific biological features.
    """
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.probes = {} # feature_name -> sklearn_model

    def extract_activations(self, data_loader, layer_idx=2):
        """
        Runs the model and collects hidden states at a specific layer.
        """
        activations = []
        labels_pam = []
        labels_seed = []

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                seq = batch['sequence'].to(self.device)

                # We need to hook the model or modify forward to return states
                # For now, we assume model.encoder returns list of states if prompted
                # Or we just use the final pooled state for simplicity in this MVP

                # Run partial forward (mock)
                emb = self.model.dna_emb(seq)
                if hasattr(self.model, 'epi_fusion') and 'epigenetics' in batch:
                     epi = batch['epigenetics'].to(self.device)
                     emb = self.model.epi_fusion(emb, epi)

                # Assume encoder has a method to get intermediate layers
                # For MVP, let's just grab the final encoding before pooling
                hidden = self.model.encoder(emb) # [B, L, D]

                # Pool
                hidden_pooled = hidden.mean(dim=1) # [B, D]

                activations.append(hidden_pooled.cpu().numpy())

                # Create synthetic biological labels for probing
                # PAM: Does the sequence end in GG?
                # Seed: Is there a mismatch in first 5 bp?
                # We need raw string sequences or decode tensors
                # This is a placeholder logic

        return np.concatenate(activations, axis=0)

    def train_probe(self, activations, binary_labels, feature_name):
        """
        Trains a linear classifier to predict a biological feature from latent state.
        Accuracy > 90% implies the model 'knows' this feature.
        """
        print(f"Training Mechanistic Probe for: {feature_name}")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(activations, binary_labels)

        preds = clf.predict(activations)
        acc = accuracy_score(binary_labels, preds)

        print(f"Probe Accuracy: {acc:.4f}")

        # Identify "Neurons" (Dimensions) with high coefficients
        top_neurons = np.argsort(np.abs(clf.coef_[0]))[-5:]
        print(f"Key Neurons for {feature_name}: {top_neurons}")

        self.probes[feature_name] = {
            'model': clf,
            'accuracy': acc,
            'neurons': top_neurons
        }
        return acc
