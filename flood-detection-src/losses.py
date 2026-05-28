"""
Loss functions for 3-class flood/permanent-water segmentation
(DeepSARFlood paper 2025 — extended for multi-class).

Class layout:
    0 = non-water (background)
    1 = flood
    2 = permanent water
   -1 = ignore (masked-out pixels)

Implemented losses (all averaged across the foreground classes 1 and 2):
  - DiceLoss
  - JaccardLoss (IoU loss)
  - TverskyLoss         (alpha=0.6  → penalise false negatives more)
  - FocalTverskyLoss    (alpha=0.6, gamma=0.75)
  - MTLLoss             (weighted combination: 0.9 * seg_loss + 0.1 * MAE on MNDWI)

Usage:
    criterion = MTLLoss(seg_loss=TverskyLoss())
    loss = criterion(flood_logits, mndwi_pred, flood_labels, mndwi_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1.0   # paper uses epsilon=1 to avoid division by zero
FOREGROUND_CLASSES = (1, 2)   # class IDs to average loss over (skip background)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert (B, C, H, W) logits → (B, C, H, W) softmax probabilities."""
    return torch.softmax(logits, dim=1)


def _per_class_targets(targets: torch.Tensor, class_id: int) -> torch.Tensor:
    """Binary one-hot for a specific class id, as float (B, H, W)."""
    return (targets == class_id).float()


def _valid_mask(targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """Boolean mask of pixels we should evaluate (B, H, W)."""
    return targets != ignore_index


def _flatten(pred: torch.Tensor, target: torch.Tensor):
    """Flatten 1D after masking: (N,), (N,)."""
    return pred.reshape(-1), target.reshape(-1)


# ---------------------------------------------------------------------------
# Per-class loss primitives
# ---------------------------------------------------------------------------

def _dice_term(pred_c: torch.Tensor, target_c: torch.Tensor) -> torch.Tensor:
    intersection = (pred_c * target_c).sum()
    cardinality  = pred_c.pow(2).sum() + target_c.pow(2).sum()
    return 1.0 - (2.0 * intersection + EPSILON) / (cardinality + EPSILON)


def _jaccard_term(pred_c: torch.Tensor, target_c: torch.Tensor) -> torch.Tensor:
    intersection = (pred_c * target_c).sum()
    union = pred_c.pow(2).sum() + target_c.pow(2).sum() - intersection
    return 1.0 - (intersection + EPSILON) / (union + EPSILON)


def _tversky_term(pred_c: torch.Tensor, target_c: torch.Tensor,
                  alpha: float) -> torch.Tensor:
    tp = (pred_c * target_c).sum()
    fn = (target_c * (1 - pred_c)).sum()
    fp = ((1 - target_c) * pred_c).sum()
    return 1.0 - (tp + EPSILON) / (tp + alpha * fn + (1 - alpha) * fp + EPSILON)


def _multiclass_loss(logits: torch.Tensor,
                     targets: torch.Tensor,
                     term_fn,
                     classes: tuple = FOREGROUND_CLASSES) -> torch.Tensor:
    """
    Compute a segmentation loss per foreground class and average.

    Args:
        logits  : (B, C, H, W) raw logits
        targets : (B, H, W) integer class map with -1 = ignore
        term_fn : callable(pred_c, target_c) -> scalar loss for one class
        classes : iterable of class ids to include (excludes background 0)
    """
    probs = _softmax_probs(logits)            # (B, C, H, W)
    valid = _valid_mask(targets)              # (B, H, W) bool

    losses = []
    for c in classes:
        pred_c   = probs[:, c]                # (B, H, W)
        target_c = _per_class_targets(targets, c)
        pred_c, target_c = _flatten(pred_c[valid], target_c[valid])
        losses.append(term_fn(pred_c, target_c))

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _multiclass_loss(logits, targets, _dice_term)


class JaccardLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _multiclass_loss(logits, targets, _jaccard_term)


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _multiclass_loss(
            logits, targets,
            lambda p, t: _tversky_term(p, t, self.alpha),
        )


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.6, gamma: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma

        def _term(p, t):
            return _tversky_term(p, t, self.alpha) ** gamma

        return _multiclass_loss(logits, targets, _term)


# ---------------------------------------------------------------------------
# MTL Combined Loss
# ---------------------------------------------------------------------------

class MTLLoss(nn.Module):
    """
    Multi-Task Learning loss:
      total = w_seg * seg_loss(logits, label) + w_reg * MAE(mndwi_pred, mndwi_target)

    Paper weights: w_seg=0.9, w_reg=0.1
    MNDWI target is clipped to [0, 0.5] (values 0–0.5 indicate water).
    """

    def __init__(self,
                 seg_loss: nn.Module | None = None,
                 w_seg: float = 0.9,
                 w_reg: float = 0.1):
        super().__init__()
        self.seg_loss = seg_loss if seg_loss is not None else TverskyLoss()
        self.w_seg = w_seg
        self.w_reg = w_reg

    def forward(self,
                flood_logits: torch.Tensor,
                mndwi_pred: torch.Tensor,
                flood_labels: torch.Tensor,
                mndwi_target: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            flood_logits  : (B, C, H, W) raw logits (C=3 by default)
            mndwi_pred    : (B, 1, H, W) predicted MNDWI in [0,1]
            flood_labels  : (B, H, W) integer class map with -1 = ignore
            mndwi_target  : (B, 1, H, W) MNDWI ground truth (optional)
        """
        seg = self.seg_loss(flood_logits, flood_labels) * self.w_seg

        if mndwi_target is not None:
            mndwi_clipped = mndwi_target.clamp(0.0, 0.5)
            reg = F.l1_loss(mndwi_pred, mndwi_clipped) * self.w_reg
        else:
            reg = torch.tensor(0.0, device=flood_logits.device)

        return seg + reg


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

LOSS_REGISTRY = {
    "dice":          DiceLoss,
    "jaccard":       JaccardLoss,
    "tversky":       TverskyLoss,
    "focal_tversky": FocalTverskyLoss,
}


def get_seg_loss(name: str) -> nn.Module:
    """Return segmentation loss by name. Used for model soup training runs."""
    name = name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]()


def get_mtl_loss(seg_loss_name: str = "tversky",
                 w_seg: float = 0.9,
                 w_reg: float = 0.1) -> MTLLoss:
    """Convenience: build MTLLoss wrapping a named seg loss."""
    return MTLLoss(seg_loss=get_seg_loss(seg_loss_name), w_seg=w_seg, w_reg=w_reg)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, H, W, C = 2, 64, 64, 3
    logits = torch.randn(B, C, H, W)
    labels = torch.randint(0, C, (B, H, W))
    labels[0, :5, :5] = -1   # exercise the ignore mask
    mndwi_pred = torch.sigmoid(torch.randn(B, 1, H, W))
    mndwi_gt   = torch.rand(B, 1, H, W) * 0.5

    for name in LOSS_REGISTRY:
        loss = get_mtl_loss(name)(logits, mndwi_pred, labels, mndwi_gt)
        assert torch.isfinite(loss), f"{name} produced non-finite loss"
        print(f"MTL({name:14s}) loss = {loss.item():.4f}")
