import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for Imbalanced Classification.
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Predictions (logits or probabilities) -> Logic handles logs
        targets: Binary targets (0 or 1)
        """
        # We assume inputs are probabilities (from Sigmoid) if strictly positive,
        # but to be safe and numerically stable, usually logits are better with BCEWithLogitsLoss.
        # However, our model output has Sigmoid() at the end of cls_head.
        # So inputs are Probabilities [0, 1].

        # Clip to avoid log(0)
        p = torch.clamp(inputs, min=1e-7, max=1-1e-7)

        # Calculate log_p for the '1' class
        log_p = torch.log(p)
        # Calculate log_1_minus_p for the '0' class
        log_1_minus_p = torch.log(1 - p)

        # Targets
        targets = targets.float()

        # Loss terms
        # If target=1: -alpha * (1-p)^gamma * log(p)
        # If target=0: -(1-alpha) * p^gamma * log(1-p)

        loss_pos = -self.alpha * torch.pow(1 - p, self.gamma) * log_p * targets
        loss_neg = -(1 - self.alpha) * torch.pow(p, self.gamma) * log_1_minus_p * (1 - targets)

        loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Dice Loss for Binary Classification.
        Loss = 1 - (2 * Intersection + Smooth) / (Union + Smooth)
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: Probabilities [0, 1]
        # targets: Binary [0, 1] - same shape as inputs or squeezable

        # Flatten strictly
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Intersection
        intersection = (inputs * targets).sum()

        # Dice Coeff: (2*Intersection + Smooth) / (Sum(Preds) + Sum(Targets) + Smooth)
        # SOTA Note: This is soft dice, differentiable
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        dice = self.dice(inputs, targets)
        focal = self.focal(inputs, targets)
        return (self.dice_weight * dice) + (self.focal_weight * focal)

class SpearmanLoss(nn.Module):
    """
    Differentiable Spearman Loss using Pearson Correlation as proxy on values.
    Loss = 1 - Correlation
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # Center
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()

        # Covariance
        cov = (pred_mean * target_mean).sum()

        # Std Dev
        pred_std = torch.sqrt((pred_mean ** 2).sum() + self.eps)
        target_std = torch.sqrt((target_mean ** 2).sum() + self.eps)

        # Correlation
        corr = cov / (pred_std * target_std + self.eps)

        return 1.0 - corr

class HybridLoss(nn.Module):
    """
    Hybrid Regression Loss: MSE + Spearman
    """
    def __init__(self, mse_weight=0.3, spearman_weight=0.6, rc_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.spearman_weight = spearman_weight
        self.rc_weight = rc_weight

        self.mse = nn.SmoothL1Loss(reduction='mean')
        self.spearman = SpearmanLoss()

    def forward(self, pred, target):
        # Calculate individual losses
        l_mse = self.mse(pred, target)

        # Spearman requires batch variance, if batch size < 2 return 0
        if pred.numel() > 1:
            l_spearman = self.spearman(pred, target)
        else:
            l_spearman = torch.tensor(0.0, device=pred.device)

        # RC Loss not implemented yet (requires RC pairs), placeholder
        l_rc = torch.tensor(0.0, device=pred.device)

        return (self.mse_weight * l_mse) + (self.spearman_weight * l_spearman) + (self.rc_weight * l_rc)
