"""
train_unet.py — Supervised UNet Training for Brain Tumor Segmentation
======================================================================
Loss     : Combined Dice + Binary Cross-Entropy (alpha=0.5 each)
Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
Early stop: patience=15 epochs on val Dice
Checkpoint: saves best model as best_unet.pth

This is the canonical training script for the project (train.py has been
removed — it was an earlier duplicate of this file).

Directory layout expected (from split_dataset.py):
    Dataset/
        train/images/  train/masks/
        val/images/    val/masks/
        test/images/   test/masks/

Run:
    python train_unet.py
    python train_unet.py --epochs 100 --batch 8 --img_size 256
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from model import UNetGenerator

# ───────────────────────── CLI ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", default="Dataset/train")
parser.add_argument("--val_dir", default="Dataset/val")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--out", default="best_unet.pth")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device : {DEVICE}")
print(f"[INFO] Epochs : {args.epochs} | Batch : {args.batch} | Size : {args.img_size}")


# ───────────────────────── DATASET ──────────────────────────────────────────
class BrainSegDataset(Dataset):
    """Loads RGB MRI + binary mask pairs with optional augmentation."""

    def __init__(self, root_dir: str, img_size: int = 256, augment: bool = False):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # ── Augmentation (training only) ──────────────────────────────────
        if self.augment:
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if torch.rand(1) > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if torch.rand(1) > 0.5:
                angle = torch.randint(-15, 15, (1,)).item()
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

            # Random brightness / contrast on image only
            img = TF.adjust_brightness(img, brightness_factor=0.8 + torch.rand(1).item() * 0.4)
            img = TF.adjust_contrast(img, contrast_factor=0.8 + torch.rand(1).item() * 0.4)

        img = T.ToTensor()(img)          # [3, H, W] float32 in [0,1]
        mask = T.ToTensor()(mask)        # [1, H, W]
        mask = (mask > 0.5).float()      # binarise

        return img, mask


# ───────────────────────── LOSSES ───────────────────────────────────────────
def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    pred   : [B,1,H,W] — raw sigmoid output (already in [0,1])
    target : [B,1,H,W] — binary mask

    NOTE: computed globally over the flattened batch (standard for a training
    loss). The *reported* dice_score() below is computed per-sample and then
    averaged — that's intentional (it matches how Dice is usually reported),
    but it means this loss value and the printed train_dice metric are not
    numerically identical. Don't be surprised if they diverge slightly.
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


bce_fn = nn.BCELoss()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Weighted sum of BCE and Dice losses."""
    return alpha * bce_fn(pred, target) + (1 - alpha) * dice_loss(pred, target)


# ───────────────────────── METRIC ───────────────────────────────────────────
@torch.no_grad()
def dice_score(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred_bin = (pred > thresh).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    denom = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean().item()


# ───────────────────────── DATA LOADERS ─────────────────────────────────────
train_ds = BrainSegDataset(args.train_dir, args.img_size, augment=True)
val_ds = BrainSegDataset(args.val_dir, args.img_size, augment=False)

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                           num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                         num_workers=0, pin_memory=True)

print(f"[INFO] Train samples : {len(train_ds)} | Val samples : {len(val_ds)}")

# ───────────────────────── MODEL / OPTIM ────────────────────────────────────
model = UNetGenerator().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=5, factor=0.5
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] Trainable params : {total_params / 1e6:.2f} M")

# ───────────────────────── TRAIN LOOP ───────────────────────────────────────
best_dice = 0.0
no_improve = 0
history = []

for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    # ── Training ──────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_dice = 0.0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)  # sigmoid already applied in forward()
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients early in training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_score(preds, masks)

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # ── Validation ────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += combined_loss(preds, masks).item()
            val_dice += dice_score(preds, masks)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    scheduler.step(val_dice)
    elapsed = time.time() - t0

    log = {
        "epoch": epoch,
        "train_loss": round(train_loss, 5),
        "train_dice": round(train_dice, 4),
        "val_loss": round(val_loss, 5),
        "val_dice": round(val_dice, 4),
        "lr": optimizer.param_groups[0]["lr"],
    }
    history.append(log)

    print(
        f"Epoch [{epoch:03d}/{args.epochs}] "
        f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
        f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
        f"lr={optimizer.param_groups[0]['lr']:.2e} "
        f"({elapsed:.1f}s)"
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    if val_dice > best_dice:
        best_dice = val_dice
        no_improve = 0
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_dice": best_dice,
            "history": history,
            "args": vars(args),
        }, args.out)
        print(f"  ✅ New best val_dice={best_dice:.4f} → saved to {args.out}")
    else:
        no_improve += 1
        if no_improve >= args.patience:
            print(f"\n[INFO] Early stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
            break

print(f"\n[DONE] Training complete. Best val Dice = {best_dice:.4f}")
print(f"[DONE] Checkpoint saved to : {args.out}")
