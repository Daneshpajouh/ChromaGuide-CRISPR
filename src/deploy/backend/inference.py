import torch
import os
import sys
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.model.crispro import CRISPROModel

class ModelService:
    def __init__(self, model_path: str, device: str = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = None
        self.model_path = model_path
        self.vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
        # Load automatically on init
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Warning: Model file not found at {self.model_path}")
            # Initialize a dummy model for development if file is missing
            self.model = CRISPROModel()
            return

        print(f"Loading model from {self.model_path}...")
        try:
            self.model = CRISPROModel()
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = CRISPROModel() # Fallback

    def _tokenize(self, sequence: str) -> torch.Tensor:
        # Simple tokenization for inference
        seq_idx = [self.vocab.get(base.upper(), 5) for base in sequence]
        return torch.tensor([seq_idx], dtype=torch.long, device=self.device)

    def predict(self, sequence: str, epigenetics: torch.Tensor = None) -> float:
        if self.model is None:
            return 0.0

        sequence_tensor = self._tokenize(sequence)

        # Helper for epigenetics if not provided (dummy zeros for now if model requires it)
        # The model allows epi_tracks=None, assuming it handles it or we pass a zero vector if needed.
        # Based on crispro.py: if epi_tracks is not None: ... else: skip fusion.

        with torch.no_grad():
            score = self.model(sequence_tensor, epigenetics)
            return float(score.item())
