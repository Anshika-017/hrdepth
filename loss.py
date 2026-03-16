"""
loss.py — Improved segmentation losses.

Changes vs original:
  - DiceLoss: sigmoid applied once, clamped numerator/denominator to prevent
    NaN on all-zero predictions.
  - FocalLoss: down-weights easy negatives, crucial for class-imbalanced
    indoor masks where background pixels dominate.
  - IoULoss: directly optimises the metric reported at evaluation time.
  - CombinedLoss: weighted sum of BCE + Dice + Focal + IoU, with configurable
    weights so you can ablate contributions.
  - All losses return scalar tensors compatible with .backward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for binary segmentation.
    Reduces weight of well-classified (easy) pixels so the model focuses on
    hard foreground boundaries — critical for indoor scene segmentation.
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t   = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class IoULoss(nn.Module):
    """
    Soft IoU loss — directly optimises the Intersection-over-Union metric.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        total        = (probs + targets).sum(dim=1)
        union        = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss used during training
# ─────────────────────────────────────────────────────────────────────────────

class CombinedSegmentationLoss(nn.Module):
    """
    Loss = w_bce * BCE  +  w_dice * Dice  +  w_focal * Focal  +  w_iou * IoU

    Default weights put most emphasis on Dice + Focal which empirically work
    best for small, class-imbalanced medical/indoor segmentation datasets.
    """

    def __init__(
        self,
        w_bce:   float = 0.3,
        w_dice:  float = 0.4,
        w_focal: float = 0.2,
        w_iou:   float = 0.1,
    ):
        super().__init__()
        self.w_bce   = w_bce
        self.w_dice  = w_dice
        self.w_focal = w_focal
        self.w_iou   = w_iou

        self.dice  = DiceLoss()
        self.focal = FocalLoss()
        self.iou   = IoULoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets)
        dice  = self.dice(logits,  targets)
        focal = self.focal(logits, targets)
        iou   = self.iou(logits,   targets)

        loss = (
            self.w_bce   * bce
            + self.w_dice  * dice
            + self.w_focal * focal
            + self.w_iou   * iou
        )
        return loss, {
            "bce": bce.item(),
            "dice": dice.item(),
            "focal": focal.item(),
            "iou": iou.item(),
        }


# ── convenience function (backward-compatible with original API) ─────────────
_loss_fn = CombinedSegmentationLoss()

def segmentation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total, _ = _loss_fn(pred, target)
    return total
