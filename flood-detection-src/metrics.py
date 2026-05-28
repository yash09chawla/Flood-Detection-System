"""
Evaluation metrics for 3-class flood/permanent-water segmentation.

Class layout matches dataset.py / losses.py:
    0 = non-water (background)
    1 = flood
    2 = permanent water
   -1 = ignore

Reports per-class IoU/F1/precision/recall plus a `mean` over the foreground
classes {1, 2}. The trainer reads `metrics["iou"]` for early stopping; that
key is kept as an alias for `iou_mean`.
"""

import torch


EPSILON = 1e-7
FOREGROUND_CLASSES = (1, 2)
CLASS_NAMES = {1: "flood", 2: "permanent"}


# ---------------------------------------------------------------------------
# Probability tensor → class map
# ---------------------------------------------------------------------------

def _to_class_map(probs: torch.Tensor) -> torch.Tensor:
    """
    Accepts either:
      - (B, C, H, W) softmax probabilities → argmax over channels
      - (B, H, W)    already a class map
    Returns (B, H, W) long tensor of predicted class ids.
    """
    if probs.dim() == 4:
        return probs.argmax(dim=1)
    return probs.long()


# ---------------------------------------------------------------------------
# Aggregated multi-class metric tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """
    Accumulates TP/FP/FN counts per foreground class across batches, then
    computes IoU/F1/precision/recall per class plus mean.

    update() expects probs as (B, C, H, W) softmax — pass the model softmax
    directly. Pixels with target == ignore_index are filtered out.
    """

    def __init__(self,
                 classes: tuple = FOREGROUND_CLASSES,
                 ignore_index: int = -1):
        self.classes = classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = {c: 0.0 for c in self.classes}
        self.fp = {c: 0.0 for c in self.classes}
        self.fn = {c: 0.0 for c in self.classes}

    @torch.no_grad()
    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            probs   : (B, C, H, W) softmax probabilities OR (B, H, W) class map
            targets : (B, H, W) integer class map with -1 = ignore
        """
        pred = _to_class_map(probs)
        valid = targets != self.ignore_index
        pred = pred[valid]
        targets = targets[valid]

        for c in self.classes:
            p_c = pred == c
            t_c = targets == c
            self.tp[c] += (p_c & t_c).sum().item()
            self.fp[c] += (p_c & ~t_c).sum().item()
            self.fn[c] += (~p_c & t_c).sum().item()

    def compute(self) -> dict[str, float]:
        out = {}
        ious, f1s = [], []
        for c in self.classes:
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            iou       = (tp + EPSILON) / (tp + fp + fn + EPSILON)
            precision = (tp + EPSILON) / (tp + fp + EPSILON)
            recall    = (tp + EPSILON) / (tp + fn + EPSILON)
            f1        = 2.0 * precision * recall / (precision + recall + EPSILON)

            name = CLASS_NAMES.get(c, f"class{c}")
            out[f"iou_{name}"]       = iou
            out[f"f1_{name}"]        = f1
            out[f"precision_{name}"] = precision
            out[f"recall_{name}"]    = recall

            ious.append(iou)
            f1s.append(f1)

        out["iou_mean"] = sum(ious) / max(len(ious), 1)
        out["f1_mean"]  = sum(f1s)  / max(len(f1s),  1)
        # Back-compat alias used by train.py for early-stopping/scheduler.
        out["iou"] = out["iou_mean"]
        out["f1"]  = out["f1_mean"]
        return out


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W, C = 4, 64, 64, 3
    probs   = torch.softmax(torch.randn(B, C, H, W), dim=1)
    targets = torch.randint(0, C, (B, H, W))
    targets[0, :5, :5] = -1   # ignored pixels

    tracker = MetricTracker()
    tracker.update(probs, targets)
    metrics = tracker.compute()
    for k, v in metrics.items():
        print(f"  {k:18s} = {v:.4f}")

    for required in ("iou_flood", "iou_permanent", "iou_mean", "iou"):
        assert required in metrics, f"missing required metric '{required}'"
