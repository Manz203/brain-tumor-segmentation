"""
evaluate_unet.py — Full Test-Set Evaluation for Trained UNet
=============================================================
Metrics : Dice, IoU, Pixel Accuracy, SSIM, PSNR
Outputs : Console summary table + visualisation grid saved as eval_results.png

This is the canonical evaluation script (evaluate.py has been removed — it
was an earlier duplicate with a double-sigmoid bug; this file does not have
that bug, since model.forward() already applies sigmoid internally).

Usage:
    python evaluate_unet.py
    python evaluate_unet.py --checkpoint best_unet.pth --test_dir Dataset/test --img_size 256
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from torch.utils.data import DataLoader, Dataset

from model import UNetGenerator

# ───────────────────────── CLI ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="best_unet.pth")
parser.add_argument("--test_dir", default="Dataset/test")
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--vis_n", type=int, default=6,
                     help="Number of samples to include in the visual grid")
parser.add_argument("--out_img", default="eval_results.png")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device     : {DEVICE}")
print(f"[INFO] Checkpoint : {args.checkpoint}")
print(f"[INFO] Test dir   : {args.test_dir}")


# ───────────────────────── DATASET ──────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        self.img_size = img_size
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img_t = self.to_tensor(img)
        mask_t = (self.to_tensor(mask) > 0.5).float()

        return img_t, mask_t, fname


# ───────────────────────── LOAD MODEL ───────────────────────────────────────
model = UNetGenerator().to(DEVICE)

if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(
        f"Checkpoint not found: {args.checkpoint}\n"
        "Train first with: python train_unet.py"
    )

ckpt = torch.load(args.checkpoint, map_location=DEVICE)

# Support both raw state_dict saves and wrapped saves from train_unet.py
if "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(best val Dice = {ckpt.get('best_dice', float('nan')):.4f})")
else:
    model.load_state_dict(ckpt, strict=False)
    print("[INFO] Loaded raw state_dict.")

model.eval()

# ───────────────────────── EVALUATE ─────────────────────────────────────────
test_ds = TestDataset(args.test_dir, args.img_size)
test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

print(f"[INFO] Test samples : {len(test_ds)}\n")

metrics_per_sample = []  # list of dicts
vis_samples = []         # (img_np, mask_np, pred_np) for first vis_n samples

with torch.no_grad():
    for imgs, masks, fnames in test_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs)  # [B,1,H,W] in [0,1]

        preds_np = preds.cpu().numpy()
        masks_np = masks.numpy()
        imgs_np = imgs.cpu().numpy()

        for b in range(preds_np.shape[0]):
            pred_2d = preds_np[b, 0]
            mask_2d = masks_np[b, 0]
            img_2d = np.transpose(imgs_np[b], (1, 2, 0))

            pred_bin = (pred_2d > args.threshold).astype(np.uint8)
            mask_bin = mask_2d.astype(np.uint8)

            # ── Dice ──────────────────────────────────────────────────
            inter = (pred_bin & mask_bin).sum()
            denom = pred_bin.sum() + mask_bin.sum()
            dice = (2 * inter + 1e-6) / (denom + 1e-6)

            # ── IoU ───────────────────────────────────────────────────
            union = (pred_bin | mask_bin).sum()
            iou = (inter + 1e-6) / (union + 1e-6)

            # ── Pixel Accuracy ────────────────────────────────────────
            acc = (pred_bin == mask_bin).mean()

            # ── SSIM ──────────────────────────────────────────────────
            s = ssim_fn(mask_2d, pred_2d, data_range=1.0)

            # ── PSNR ──────────────────────────────────────────────────
            p = psnr_fn(mask_2d, pred_2d, data_range=1.0)

            metrics_per_sample.append({
                "dice": dice, "iou": iou, "acc": acc, "ssim": s, "psnr": p
            })

            if len(vis_samples) < args.vis_n:
                vis_samples.append((img_2d, mask_2d, pred_2d, pred_bin))

# ───────────────────────── AGGREGATE ────────────────────────────────────────
keys = ["dice", "iou", "acc", "ssim", "psnr"]
means = {k: np.mean([m[k] for m in metrics_per_sample]) for k in keys}
stds = {k: np.std([m[k] for m in metrics_per_sample]) for k in keys}

print("=" * 60)
print("  EVALUATION RESULTS (test set)")
print("=" * 60)
print(f"  Samples evaluated : {len(metrics_per_sample)}")
print("-" * 60)
print(f"  {'Metric':<18} {'Mean':>10} {'Std':>10}")
print("-" * 60)
labels = {"dice": "Dice", "iou": "IoU", "acc": "Pixel Accuracy",
          "ssim": "SSIM", "psnr": "PSNR (dB)"}
for k in keys:
    print(f"  {labels[k]:<18} {means[k]:>10.4f} {stds[k]:>10.4f}")
print("=" * 60)

# ───────────────────────── VISUALISATION ────────────────────────────────────
n = len(vis_samples)
cols = 4  # MRI | GT Mask | Pred Mask | Overlay
fig = plt.figure(figsize=(cols * 3.2, n * 3.2))
fig.patch.set_facecolor("#1a1814")
gs = gridspec.GridSpec(n, cols, figure=fig, hspace=0.04, wspace=0.04)

col_titles = ["MRI Scan", "Ground Truth", "Predicted Mask", "Overlay"]
for col, title in enumerate(col_titles):
    ax = fig.add_subplot(gs[0, col])
    ax.set_title(title, color="#b5883e", fontsize=11, fontweight="bold", pad=8)

for row, (img, gt_mask, pred_prob, pred_bin) in enumerate(vis_samples):
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.imshow(img)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[row, 1])
    ax1.imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[row, 2])
    ax2.imshow(pred_prob, cmap="hot", vmin=0, vmax=1)
    ax2.axis("off")

    # Overlay — TP=green, FP=red, FN=blue
    gt_b = gt_mask.astype(bool)
    pr_b = pred_bin.astype(bool)
    overlay = img.copy()
    overlay[gt_b & pr_b] = overlay[gt_b & pr_b] * 0.4 + np.array([0.0, 1.0, 0.0]) * 0.6
    overlay[pr_b & ~gt_b] = overlay[pr_b & ~gt_b] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6
    overlay[gt_b & ~pr_b] = overlay[gt_b & ~pr_b] * 0.4 + np.array([0.0, 0.4, 1.0]) * 0.6
    overlay = np.clip(overlay, 0, 1)

    ax3 = fig.add_subplot(gs[row, 3])
    ax3.imshow(overlay)
    ax3.axis("off")

summary = (
    f"Dice {means['dice']:.3f} | IoU {means['iou']:.3f} | "
    f"Acc {means['acc']:.3f} | SSIM {means['ssim']:.3f} | PSNR {means['psnr']:.1f} dB"
)
fig.text(0.5, 0.01, summary, ha="center", fontsize=11,
          color="#c9a05c", fontfamily="monospace")

plt.savefig(args.out_img, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[INFO] Visualisation saved → {args.out_img}")
print("[DONE] Evaluation complete.")
