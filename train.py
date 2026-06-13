"""
train.py — Fixed Brain Tumor Segmentation Training
===================================================
Fixes applied:
  1. Removed double sigmoid in dice_loss and dice_score
  2. Removed GAN (discriminator was hurting segmentation learning)
  3. Removed dataset cap of 1000
  4. Increased image size to 256x256
  5. Switched BCEWithLogitsLoss → BCELoss (model outputs sigmoid already)
  6. Added LR scheduler and early stopping
  7. Saves best_unet.pth in the same format as train_unet.py
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import UNetGenerator

# ─── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Dataset ─────────────────────────────────────────────────────────────────
class BrainDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.files    = sorted(os.listdir(img_dir))   # ✅ use ALL files, no cap
        self.augment  = augment

        self.transform = T.Compose([
            T.Resize((256, 256)),   # ✅ 256 instead of 128
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.img_dir,  self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        # Resize both together
        image = image.resize((256, 256), Image.BILINEAR)
        mask  = mask.resize((256, 256),  Image.NEAREST)

        # Augmentation (training only)
        if self.augment:
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if torch.rand(1) > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)
            if torch.rand(1) > 0.5:
                angle = torch.randint(-15, 15, (1,)).item()
                image = TF.rotate(image, angle)
                mask  = TF.rotate(mask,  angle)

        image = T.ToTensor()(image)           # [3, H, W] float32 in [0,1]
        mask  = T.ToTensor()(mask)            # [1, H, W]
        mask  = (mask > 0.5).float()          # binarise

        return image, mask


# ─── Data loaders ────────────────────────────────────────────────────────────
train_dataset = BrainDataset("Dataset/train/images", "Dataset/train/masks", augment=True)
val_dataset   = BrainDataset("Dataset/val/images",   "Dataset/val/masks",   augment=False)

train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# ─── Model ───────────────────────────────────────────────────────────────────
# ✅ Only UNetGenerator — no discriminator needed for supervised segmentation
gen = UNetGenerator().to(device)

# ─── Losses ──────────────────────────────────────────────────────────────────
# ✅ BCELoss (NOT BCEWithLogitsLoss) because model already applies sigmoid
bce_fn = nn.BCELoss()

def dice_loss(pred, target, smooth=1e-6):
    # ✅ NO sigmoid here — model output is already in [0,1]
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    inter  = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target, alpha=0.5):
    return alpha * bce_fn(pred, target) + (1 - alpha) * dice_loss(pred, target)

# ─── Metric ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def dice_score(pred, target, thresh=0.5):
    # ✅ NO sigmoid here — model output is already in [0,1]
    pred_bin = (pred > thresh).float()
    inter    = (pred_bin * target).sum(dim=(1, 2, 3))
    denom    = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean().item()

# ─── Optimizer & Scheduler ───────────────────────────────────────────────────
optimizer = torch.optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=5, factor=0.5
)

# ─── Training loop ───────────────────────────────────────────────────────────
EPOCHS    = 50
PATIENCE  = 15
best_dice = 0.0
no_improve = 0

for epoch in range(1, EPOCHS + 1):

    # ── Train ────────────────────────────────────────────────────────────
    gen.train()
    train_loss = 0.0
    train_dice = 0.0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = gen(imgs)                        # sigmoid already applied
        loss  = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_score(preds, masks)

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # ── Validate ─────────────────────────────────────────────────────────
    gen.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds     = gen(imgs)
            val_loss += combined_loss(preds, masks).item()
            val_dice += dice_score(preds, masks)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    scheduler.step(val_dice)

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] "
        f"train_loss={train_loss:.4f}  train_dice={train_dice:.4f}  "
        f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  "
        f"lr={optimizer.param_groups[0]['lr']:.2e}"
    )

    # ── Save best ────────────────────────────────────────────────────────
    if val_dice > best_dice:
        best_dice  = val_dice
        no_improve = 0
        torch.save({
            "epoch":      epoch,
            "state_dict": gen.state_dict(),
            "best_dice":  best_dice,
        }, "best_unet.pth")
        print(f"  ✅ New best val_dice={best_dice:.4f} → saved to best_unet.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n[INFO] Early stopping after {epoch} epochs")
            break

print(f"\nTraining complete. Best val Dice = {best_dice:.4f}")