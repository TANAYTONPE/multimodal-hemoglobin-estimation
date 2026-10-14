"""
train.py
--------
Model definition, training loop, evaluation, and checkpointing
for nail-bed hemoglobin regression using PyTorch + EfficientNet-B0.

Usage
-----
    python train.py                         # train with defaults
    python train.py --epochs 50 --lr 3e-4  # override hyperparams
"""

import os
import math
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from preprocessing import get_dataloaders

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ──────────────────────────────────────────────────────────────────────
class HemoglobinNet(nn.Module):
    """
    EfficientNet-B0 backbone fine-tuned for scalar hemoglobin regression.

    Architecture
    ------------
    EfficientNet-B0 (pretrained on ImageNet)
        → Global average pooling  (1280-d feature vector)
        → Dropout (p=0.3)
        → FC(1280 → 256) + ReLU + BN
        → Dropout (p=0.2)
        → FC(256 → 1)             (raw g/dL output)
    """

    def __init__(self, dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()               # strip original head

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 1280)
        out      = self.head(features)       # (B, 1)
        return out.squeeze(1)               # (B,)


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² for regression."""
    mae  = np.mean(np.abs(preds - targets))
    rmse = math.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-8)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ── Train / eval loops ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for step, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        if step % 10 == 0 or step == len(loader):
            print(f"  Epoch [{epoch}/{total_epochs}] Step [{step}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_metrics(np.array(all_preds), np.array(all_targets))
    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images)
        loss  = criterion(preds, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_metrics(np.array(all_preds), np.array(all_targets))
    return epoch_loss, metrics


# ── Training orchestrator ──────────────────────────────────────────────────────
def train(args):
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir  = os.path.join("checkpoints", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Hemoglobin Regression Training  |  Run: {run_id}")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {args.epochs}  |  LR : {args.lr}  |  Batch : {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Data ──
    loaders = get_dataloaders(
        csv_path=args.csv,
        root_dir=args.root,
        batch_size=args.batch_size,
    )

    # ── Model ──
    model = HemoglobinNet(dropout=args.dropout).to(DEVICE)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total_params:,}  trainable={trainable_params:,}\n")

    # ── Loss & optimiser ──
    criterion = nn.HuberLoss(delta=1.0)     # robust to outlier labels
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ──
    best_val_mae  = float("inf")
    history       = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, epoch, args.epochs
        )
        val_loss, val_metrics = evaluate(model, loaders["val"], criterion)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:03d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}  MAE: {train_metrics['MAE']:.3f}  "
            f"RMSE: {train_metrics['RMSE']:.3f}  R2: {train_metrics['R2']:.3f}\n"
            f"{'':>30}Val   Loss: {val_loss:.4f}  MAE: {val_metrics['MAE']:.3f}  "
            f"RMSE: {val_metrics['RMSE']:.3f}  R2: {val_metrics['R2']:.3f}"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        })

        # ── Checkpoint ──
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            ckpt_path    = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_mae":    best_val_mae,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  [BEST] New best saved -> {ckpt_path}  (MAE={best_val_mae:.3f})")

        print()

    # ── Final test evaluation ──
    print(f"\n{'='*60}")
    print("  Final evaluation on TEST set")
    print(f"{'='*60}")
    test_loss, test_metrics = evaluate(model, loaders["test"], criterion)
    print(
        f"  Test Loss : {test_loss:.4f}\n"
        f"  MAE       : {test_metrics['MAE']:.3f} g/dL\n"
        f"  RMSE      : {test_metrics['RMSE']:.3f} g/dL\n"
        f"  R2        : {test_metrics['R2']:.3f}\n"
    )

    # ── Save history as CSV ──
    import pandas as pd
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(ckpt_dir, "history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"  Training history saved -> {hist_path}")
    print(f"  Best model saved       -> {ckpt_dir}/best_model.pth\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train hemoglobin regression model")
    p.add_argument("--csv",        default="data/metadata.csv", help="Path to metadata CSV")
    p.add_argument("--root",       default=".",                 help="Project root dir")
    p.add_argument("--epochs",     type=int,   default=30,      help="Number of epochs")
    p.add_argument("--lr",         type=float, default=3e-4,    help="Learning rate")
    p.add_argument("--wd",         type=float, default=1e-4,    help="Weight decay")
    p.add_argument("--batch_size", type=int,   default=32,      help="Batch size")
    p.add_argument("--dropout",    type=float, default=0.3,     help="Dropout in head")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
