
import torch
import torch.nn as nn
import torch.nn.functional as F

class RankNetLoss(nn.Module):
    """
    RankNet Pairwise Ranking Loss.

    Optimizes for the relative order of predictors rather than their absolute values.
    Crucial for Spearman correlation maximization in CRISPR efficiency prediction.

    Paper: Burges et al., "Learning to Rank using Gradient Descent", ICML 2005.

    Math:
    P_ij = 1 / (1 + exp(-(s_i - s_j)))  [Predicted probability that i > j]
    P_bar_ij = 1 if y_i > y_j else 0    [Actual probability, simplified to binary]
    L = -P_bar_ij * log(P_ij) - (1 - P_bar_ij) * log(1 - P_ij) [Cross Entropy]
    """
    def __init__(self, sigma=1.0):
        super(RankNetLoss, self).__init__()
        self.sigma = sigma

    def forward(self, preds, targets):
        """
        Args:
            preds: (B, 1) or (B,) Predicted scores
            targets: (B, 1) or (B,) True efficiency scores
        """
        # Flatten
        preds = preds.view(-1, 1)
        targets = targets.view(-1, 1)

        # Create all pairs (B^2 complexity) - acceptable for batch_size <= 64
        # pairwise difference matrices
        # P_diff[i][j] = preds[i] - preds[j]
        pred_diff = preds - preds.t()

        # T_diff[i][j] = targets[i] - targets[j]
        target_diff = targets - targets.t()

        # Signed Indicator Matrix:
        # S_ij = 1 if T_i > T_j
        # S_ij = -1 if T_i < T_j
        # S_ij = 0 if T_i = T_j
        S_ij = torch.sign(target_diff)

        # We only care about pairs where T_i != T_j (i.e. one is better than other)
        # Pairs with equal targets don't contribute to ranking info
        mask = (S_ij != 0) & (torch.abs(target_diff) > 0.1) # Optimized: Only pair samples with distinct efficiency

        if mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        # Target Probability (1 if i > j, 0 if j > i)
        # For RankNet, we can effectively use BCE on the logistic of the diff
        # P_ij_true = 0.5 * (1 + S_ij) => 1 or 0
        P_ij_true = 0.5 * (1 + S_ij[mask])

        # Predicted Logits
        pred_logits = pred_diff[mask]

        # BCE with Logits
        # Input: Logits (pred_diff), Target (Probabilities 0 or 1)
        loss = F.binary_cross_entropy_with_logits(pred_logits, P_ij_true, reduction='mean')

        return loss

class HybridRankLoss(nn.Module):
    """
    Combines MSE (for stability/calibration) + RankNet (for ordering).
    """
    def __init__(self, rank_weight=0.8, mse_weight=0.2):
        super(HybridRankLoss, self).__init__()
        self.rank_loss = RankNetLoss()
        self.mse_loss = nn.MSELoss()
        self.rank_weight = rank_weight
        self.mse_weight = mse_weight

    def forward(self, preds, targets):
        L_rank = self.rank_loss(preds, targets)
        L_mse = self.mse_loss(preds, targets)

        return self.rank_weight * L_rank + self.mse_weight * L_mse
