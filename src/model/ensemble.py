import torch
import torch.nn as nn
from src.model.deepmens import DeepMEnsExact

class DeepMEnsEnsemble(nn.Module):
    """
    Validation-time ensemble wrapper for DeepMEns.
    Holds 5 trained models and averages their predictions.
    """
    def __init__(self, models_list):
        super(DeepMEnsEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x_seq, x_shape, x_pos):
        predictions = []
        for model in self.models:
            predictions.append(model(x_seq, x_shape, x_pos))

        # stack and average
        stacked = torch.stack(predictions, dim=1) # (B, 5, 1)
        mean_pred = torch.mean(stacked, dim=1)    # (B, 1)
        std_pred = torch.std(stacked, dim=1)      # (B, 1) - Uncertainty proxy!

        return mean_pred, std_pred
