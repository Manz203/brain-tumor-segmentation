"""
inference.py — Single-image inference for Brain Tumor Segmentation
====================================================================
Loads a trained checkpoint and runs the model on ONE MRI scan, saving the
predicted mask and an overlay image. No ground truth required.

(Previous version of this file was an accidental duplicate of
evaluate_unet.py and did not actually support single-image inference —
this is the fix for that.)

Usage:
    python inference.py --image path/to/scan.png
    python inference.py --image scan.png --checkpoint best_unet.pth --out_dir results/
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from model import UNetGenerator

# ───────────────────────── CLI ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to a single MRI scan image")
parser.add_argument("--checkpoint", default="best_unet.pth")
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--out_dir", default=".", help="Directory to save mask/overlay outputs")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device     : {DEVICE}")
print(f"[INFO] Checkpoint : {args.checkpoint}")
print(f"[INFO] Image      : {args.image}")

if not os.path.exists(args.image):
    raise FileNotFoundError(f"Image not found: {args.image}")

if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(
        f"Checkpoint not found: {args.checkpoint}\n"
        "Train first with: python train_unet.py"
    )

os.makedirs(args.out_dir, exist_ok=True)

# ───────────────────────── LOAD MODEL ───────────────────────────────────────
model = UNetGenerator().to(DEVICE)
ckpt = torch.load(args.checkpoint, map_location=DEVICE)

if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(best val Dice = {ckpt.get('best_dice', float('nan')):.4f})")
else:
    model.load_state_dict(ckpt, strict=False)
    print("[INFO] Loaded raw state_dict.")

model.eval()

# ───────────────────────── PRE-PROCESS ──────────────────────────────────────
orig_image = Image.open(args.image).convert("RGB")
image_resized = orig_image.resize((args.img_size, args.img_size), Image.BILINEAR)

img_np = np.array(image_resized).astype(np.float32) / 255.0
img_t = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

# ───────────────────────── INFERENCE ────────────────────────────────────────
with torch.no_grad():
    # model.forward() already applies sigmoid — output is in [0, 1]
    pred = model(img_t).cpu().numpy()[0, 0]  # [H, W]

mask = (pred > args.threshold).astype(np.uint8)

# ── Honest statistics — no ground truth needed ───────────────────────────
tumor_area_pct = float(mask.mean() * 100)
mean_confidence = float(pred.mean())
tumor_confidence = float(pred[mask == 1].mean()) if mask.sum() > 0 else 0.0

print("\n" + "=" * 50)
print("  PREDICTION SUMMARY")
print("=" * 50)
print(f"  Tumour area        : {tumor_area_pct:.2f}% of image")
print(f"  Mean confidence     : {mean_confidence:.4f}")
print(f"  Tumour-region conf. : {tumor_confidence:.4f}")
print("=" * 50)
print(
    "  NOTE: Dice/IoU/Accuracy/SSIM/PSNR are supervised metrics that\n"
    "  require a ground-truth mask and are not computed here. Run\n"
    "  evaluate_unet.py on a labelled test set for those."
)

# ───────────────────────── SAVE OUTPUTS ─────────────────────────────────────
base = os.path.splitext(os.path.basename(args.image))[0]

mask_path = os.path.join(args.out_dir, f"{base}_mask.png")
overlay_path = os.path.join(args.out_dir, f"{base}_overlay.png")

Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

overlay_np = np.array(image_resized).copy()
overlay_np[mask == 1] = [255, 0, 0]
Image.fromarray(overlay_np.astype(np.uint8)).save(overlay_path)

print(f"\n[INFO] Mask saved    → {mask_path}")
print(f"[INFO] Overlay saved → {overlay_path}")
print("[DONE] Inference complete.")
