import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """ Implementation of the Focal Loss"""
    def __init__(self, gamma=0, alpha=torch.tensor([1])):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none', pos_weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
