import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Penalizes the model heavier for misclassifying difficult examples.
    Highly effective for imbalanced time series data.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def get_loss_function(loss_name="crossentropy"):
    if loss_name.lower() == "crossentropy":
        return nn.CrossEntropyLoss()
    elif loss_name.lower() == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Loss function {loss_name} not supported.")