"""
Training script for the ResNet50 + UNet++ flood segmentation model.

Implements (from DeepSARFlood paper 2025):
  1. Single-model training with MTL (flood seg + MNDWI regression)
  2. Four training runs with different loss functions (for model soup)
  3. Greedy model soup: weight-averages checkpoints that improve val IoU
  4. Early stopping + LR scheduling (RMSprop + ReduceLROnPlateau)

Usage (from notebook or CLI):
    python train.py --data_root /path/to/sen1floods11/data \
                    --splits_root /path/to/sen1floods11/v1.1/splits \
                    --checkpoint_dir /path/to/checkpoints \
                    --epochs 400 --batch_size 12
"""

import os
import sys
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow imports from src/ when running as script
sys.path.insert(0, os.path.dirname(__file__))

from model   import build_model
from dataset import build_datasets
from losses  import get_mtl_loss
from metrics import MetricTracker


# ---------------------------------------------------------------------------
# Config defaults (overridden by argparse / notebook)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data_root":       "data/Sen1Floods11/v1.1/data/flood_events",
    "splits_root":     "data/Sen1Floods11/v1.1/splits/flood_handlabeled",
    "checkpoint_dir":  "checkpoints",
    "patch_size":      512,
    "batch_size":      12,
    "num_workers":     4,
    "lr":              1e-3,
    "max_epochs":      400,
    "patience_reduce": 8,     # epochs before LR halved
    "patience_stop":   20,    # epochs before early stopping
    "min_lr":          1e-7,
    "loss_names":      ["dice", "jaccard", "tversky", "focal_tversky"],
    "pretrained":      True,
}

LOSS_NAMES = ["dice", "jaccard", "tversky", "focal_tversky"]


# ---------------------------------------------------------------------------
# Training loop for a single model
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    tracker = MetricTracker()

    for batch in tqdm(loader, desc="  Train", leave=False):
        images  = batch["image"].to(device)
        labels  = batch["label"].to(device)
        mndwi   = batch["mndwi"].to(device)

        optimizer.zero_grad()

        if scaler is not None:   # AMP (mixed precision)
            with torch.amp.autocast('cuda'):
                logits, mndwi_pred = model(images)
                loss = criterion(logits, mndwi_pred, labels, mndwi)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            logits, mndwi_pred = model(images)
            loss = criterion(logits, mndwi_pred, labels, mndwi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)   # (B, C, H, W)
            tracker.update(probs, labels)

    metrics = tracker.compute()
    return total_loss / max(len(loader), 1), metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    tracker = MetricTracker()

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        mndwi  = batch["mndwi"].to(device)

        logits, mndwi_pred = model(images)
        loss = criterion(logits, mndwi_pred, labels, mndwi)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        tracker.update(probs, labels)

    metrics = tracker.compute()
    return total_loss / max(len(loader), 1), metrics


# ---------------------------------------------------------------------------
# Full training run for one loss function
# ---------------------------------------------------------------------------

def run_training(cfg: dict,
                 loss_name: str,
                 datasets: dict,
                 device: torch.device) -> tuple[nn.Module, float]:
    """
    Train one model with the specified loss function.

    Returns:
        model     : best model state (loaded from checkpoint)
        best_iou  : best validation IoU achieved
    """
    print(f"\n{'='*60}")
    print(f"  Training run: loss = {loss_name.upper()}")
    print(f"{'='*60}")

    # DataLoaders
    train_loader = DataLoader(
        datasets["train"], batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = build_model(in_channels=6, pretrained=cfg["pretrained"]).to(device)
    criterion = get_mtl_loss(loss_name)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg["lr"])



    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=cfg["patience_reduce"],
        min_lr=cfg["min_lr"],
    )


    # AMP scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_iou   = -1.0
    best_state = None
    no_improve = 0

    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"best_{loss_name}.pth")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, cfg["max_epochs"] + 1):
        tr_loss, tr_m = train_one_epoch(model, train_loader, criterion, optimizer,
                                        device, scaler)
        val_loss, val_m = validate(model, val_loader, criterion, device)

        iou      = val_m["iou"]                       # alias for iou_mean
        iou_fl   = val_m.get("iou_flood",     float("nan"))
        iou_pm   = val_m.get("iou_permanent", float("nan"))
        f1       = val_m["f1"]
        lr       = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:3d}/{cfg['max_epochs']} | "
              f"tr_loss={tr_loss:.4f} | val_IoU={iou:.4f} "
              f"(flood={iou_fl:.4f}, perm={iou_pm:.4f}) | "
              f"val_F1={f1:.4f} | lr={lr:.2e}")

        scheduler.step(iou)

        # Save best
        if iou > best_iou:
            best_iou = iou
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_state,
                        "epoch": epoch,
                        "iou": best_iou,
                        "loss": loss_name}, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= cfg["patience_stop"] or lr <= cfg["min_lr"]:
            print(f"  Early stopping at epoch {epoch} (no improvement for {no_improve} epochs)")
            break

    # Load best weights
    model.load_state_dict(best_state)
    print(f"  Best val IoU for {loss_name}: {best_iou:.4f}")
    return model, best_iou


# ---------------------------------------------------------------------------
# Greedy Model Soup
# ---------------------------------------------------------------------------

def _recalibrate_bn(model: nn.Module, loader: DataLoader, 
                    device: torch.device, n_batches: int = 50):
    """Recalibrate BatchNorm stats after weight averaging using train data."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.num_batches_tracked.zero_()
            m.train()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            model(batch['image'].to(device))
    model.eval()


def greedy_model_soup(models_with_scores, val_loader, train_loader,
                      criterion, device, checkpoint_dir):
    """
    Greedy weight-averaged ensemble (Wortsman et al. 2022 "Model Soups").

    All parameters are averaged uniformly — including BatchNorm running_mean
    and running_var. We deliberately do NOT reset and re-collect BN stats from
    a small sample, because that destroys the calibration the trained weights
    rely on (50 batches is far too few to recover stats accumulated over the
    full training run).

    `train_loader` is accepted for API stability but is unused.
    """
    del train_loader  # kept in signature for backwards-compat with main()

    print("\n" + "="*60)
    print("  Building Greedy Model Soup")
    print("="*60)

    ranked = sorted(models_with_scores, key=lambda x: x[1], reverse=True)

    soup_model = build_model(in_channels=6, pretrained=False).to(device)
    soup_state = copy.deepcopy(ranked[0][0].state_dict())
    soup_model.load_state_dict(soup_state)

    _, base_metrics = validate(soup_model, val_loader, criterion, device)
    current_iou = base_metrics["iou"]
    print(f"  Base model IoU: {current_iou:.4f}  ({ranked[0][1]:.4f} at training)")

    n_members = 1

    for model, score in ranked[1:]:
        new_state = model.state_dict()
        candidate_state = {}
        for key in soup_state:
            # num_batches_tracked is an integer counter — keep the soup's value
            if "num_batches_tracked" in key:
                candidate_state[key] = soup_state[key]
                continue
            # Average everything else (weights + BN running stats) uniformly
            candidate_state[key] = (
                soup_state[key].float() * n_members + new_state[key].float()
            ) / (n_members + 1)
            # Preserve original dtype (BN running stats are float, weights are float)
            candidate_state[key] = candidate_state[key].to(soup_state[key].dtype)

        soup_model.load_state_dict(candidate_state)

        _, cand_metrics = validate(soup_model, val_loader, criterion, device)
        cand_iou = cand_metrics["iou"]

        print(f"  + model (train IoU={score:.4f}) → candidate soup IoU={cand_iou:.4f}", end="")

        if cand_iou >= current_iou:
            soup_state  = candidate_state
            current_iou = cand_iou
            n_members  += 1
            print("  ✓ ADDED")
        else:
            soup_model.load_state_dict(soup_state)
            print("  ✗ skipped")

    print(f"\n  Final soup: {n_members} members | val IoU = {current_iou:.4f}")

    soup_path = os.path.join(checkpoint_dir, "model_soup.pth")
    torch.save({"model_state": soup_model.state_dict(),
                "n_members":   n_members,
                "iou":         current_iou}, soup_path)
    print(f"  Soup saved → {soup_path}")
    return soup_model


# ---------------------------------------------------------------------------
# Final evaluation on test set
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test(model: nn.Module,
                  test_loader: DataLoader,
                  device: torch.device) -> dict:
    model.eval()
    tracker = MetricTracker()
    for batch in tqdm(test_loader, desc="  Test "):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(images)
        probs = torch.softmax(logits, dim=1)
        tracker.update(probs, labels)
    return tracker.compute()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets
    datasets = build_datasets(
        data_root   = cfg["data_root"],
        splits_root = cfg["splits_root"],
        patch_size  = cfg["patch_size"],
    )

    train_loader = DataLoader(
        datasets["train"], batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
    )

    # Train 4 models (one per loss function)
    models_with_scores = []
    for loss_name in cfg["loss_names"]:
        model, best_iou = run_training(cfg, loss_name, datasets, device)
        models_with_scores.append((model, best_iou))

    # Build greedy soup (train_loader is needed for BatchNorm recalibration
    # after weight averaging — see _recalibrate_bn)
    dummy_criterion = get_mtl_loss("tversky")
    soup_model = greedy_model_soup(
        models_with_scores, val_loader, train_loader, dummy_criterion,
        device, cfg["checkpoint_dir"],
    )

    # Test set evaluation
    print("\n" + "="*60)
    print("  Final Test Set Evaluation (Model Soup)")
    print("="*60)
    test_metrics = evaluate_test(soup_model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"  {k:12s} = {v:.4f}")

    # Also evaluate individual models
    print("\n  Individual model test results:")
    for (model, _), loss_name in zip(models_with_scores, cfg["loss_names"]):
        m = evaluate_test(model, test_loader, device)
        print(f"  [{loss_name:14s}] IoU={m['iou']:.4f}  "
              f"(flood={m.get('iou_flood', float('nan')):.4f}, "
              f"perm={m.get('iou_permanent', float('nan')):.4f})  "
              f"F1={m['f1']:.4f}")

    return soup_model, test_metrics


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50+UNet++ flood model")
    parser.add_argument("--data_root",      default=DEFAULT_CONFIG["data_root"])
    parser.add_argument("--splits_root",    default=DEFAULT_CONFIG["splits_root"])
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CONFIG["checkpoint_dir"])
    parser.add_argument("--patch_size",     type=int, default=DEFAULT_CONFIG["patch_size"])
    parser.add_argument("--batch_size",     type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num_workers",    type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--lr",             type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--max_epochs",     type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--no_pretrained",  action="store_true")
    parser.add_argument("--loss",           nargs="+", default=LOSS_NAMES,
                        choices=LOSS_NAMES, help="Which loss functions to use")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg.update({
        "data_root":      args.data_root,
        "splits_root":    args.splits_root,
        "checkpoint_dir": args.checkpoint_dir,
        "patch_size":     args.patch_size,
        "batch_size":     args.batch_size,
        "num_workers":    args.num_workers,
        "lr":             args.lr,
        "max_epochs":     args.max_epochs,
        "pretrained":     not args.no_pretrained,
        "loss_names":     args.loss,
    })

    main(cfg)
