"""
api.py — FastAPI inference backend for Brain Tumor Segmentation
===============================================================
Supports checkpoints saved by train_unet.py (wrapped dict)
and legacy raw state_dict saves (best_generator.pth style).

Run:
    uvicorn api:app --reload --port 8000

NOTE ON METRICS
---------------
Dice, IoU, Accuracy, SSIM and PSNR are *supervised* metrics — they require a
ground-truth mask to be meaningful.  At inference time we have no ground truth,
so those metrics are NOT returned here (they were previously computed by
comparing the prediction against its own thresholded version, which is
self-referential and always gives inflated / meaningless numbers).

Real evaluation against labelled test data lives in evaluate_unet.py.

What IS returned:
  • tumor_area_pct   — percentage of pixels predicted as tumour (real)
  • mean_confidence  — average sigmoid probability across ALL pixels (model certainty)
  • tumor_confidence — average sigmoid probability inside the tumour mask (region confidence)
  • mask_b64         — binary segmentation mask as PNG
  • overlay_b64      — original image with tumour region highlighted in red
"""

import io
import base64
import os

import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model import UNetGenerator

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model loading — try new checkpoint first, fall back to legacy ────────────
CHECKPOINT_PATH = "best_unet.pth"
LEGACY_PATH     = "best_generator.pth"

model = UNetGenerator().to(DEVICE)

ckpt_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else LEGACY_PATH

if not os.path.exists(ckpt_path):
    raise RuntimeError(
        "No checkpoint found. Train the model first with:  python train_unet.py"
    )

ckpt = torch.load(ckpt_path, map_location=DEVICE)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
    print(f"[INFO] Loaded {ckpt_path}  (epoch {ckpt.get('epoch', '?')}, "
          f"best val Dice {ckpt.get('best_dice', 0):.4f})")
else:
    model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded raw state_dict from {ckpt_path}")

model.eval()

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="NeuroSeg API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _encode(img: Image.Image) -> str:
    """Encode a PIL image to a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Inference endpoint ───────────────────────────────────────────────────────
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a brain MRI image (any common format).
    Returns the tumour segmentation mask, an overlay, and honest confidence
    statistics derived purely from the model's own sigmoid output.
    """
    try:
        # ── Pre-process ───────────────────────────────────────────────────
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_256 = image.resize((256, 256), Image.BILINEAR)

        img_np = np.array(image_256).astype(np.float32) / 255.0
        img_t  = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # ── Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            # model.forward() already applies sigmoid — output is in [0, 1]
            pred = model(img_t).cpu().numpy()[0, 0]   # shape: [H, W]

        # ── Threshold → binary mask ───────────────────────────────────────
        THRESH = 0.5
        mask = (pred > THRESH).astype(np.uint8)

        # ── Honest statistics (no ground truth required) ──────────────────
        # 1. Tumour area: fraction of pixels classified as tumour
        tumor_area_pct = float(mask.mean() * 100)

        # 2. Mean confidence: average raw sigmoid value over ALL pixels.
        #    Values near 0.5 = uncertain; near 0 or 1 = confident.
        mean_confidence = float(pred.mean())

        # 3. Tumour-region confidence: average sigmoid value INSIDE the
        #    predicted tumour mask. High value = model is sure those pixels
        #    are tumour; close to 0.5 = borderline.
        if mask.sum() > 0:
            tumor_confidence = float(pred[mask == 1].mean())
        else:
            tumor_confidence = 0.0

        # ── Output images ─────────────────────────────────────────────────
        # Binary mask image
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_b64 = _encode(mask_img)

        # Overlay: highlight predicted tumour region in red on the input image
        overlay_np = np.array(image_256).copy()
        overlay_np[mask == 1] = [255, 0, 0]
        overlay_b64 = _encode(Image.fromarray(overlay_np.astype(np.uint8)))

        return {
            # ── Segmentation output ──────────────────────────────────────
            "mask_b64":          mask_b64,
            "overlay_b64":       overlay_b64,

            # ── Honest inference-time statistics ─────────────────────────
            "tumor_area_pct":    round(tumor_area_pct,   3),
            "mean_confidence":   round(mean_confidence,  4),
            "tumor_confidence":  round(tumor_confidence, 4),

            # ── Info ──────────────────────────────────────────────────────
            # Supervised metrics (Dice, IoU, Accuracy, SSIM, PSNR) are NOT
            # returned here because they require a ground-truth mask.
            # Run evaluate_unet.py on your labelled test set for those.
            "note": (
                "Supervised metrics (Dice/IoU/Accuracy/SSIM/PSNR) require "
                "ground-truth masks and are therefore only available via "
                "evaluate_unet.py on the labelled test set."
            ),
        }

    except Exception as exc:
        return {"error": str(exc)}


# ─── Evaluation endpoint (requires ground-truth mask upload) ─────────────────
@app.post("/api/evaluate")
async def evaluate(
    image_file: UploadFile = File(...),
    mask_file:  UploadFile = File(...),
):
    """
    Accepts an MRI image AND its ground-truth binary mask.
    Returns real Dice, IoU, and Pixel Accuracy computed against the GT mask.

    This is the only endpoint that returns supervised metrics — because here
    we actually have the ground truth to compare against.
    """
    try:
        # ── Load & pre-process image ──────────────────────────────────────
        image = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
        image = image.resize((256, 256), Image.BILINEAR)
        img_np = np.array(image).astype(np.float32) / 255.0
        img_t  = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # ── Load & binarise ground-truth mask ─────────────────────────────
        gt_pil  = Image.open(io.BytesIO(await mask_file.read())).convert("L")
        gt_pil  = gt_pil.resize((256, 256), Image.NEAREST)
        gt_np   = (np.array(gt_pil) / 255.0 > 0.5).astype(np.uint8)   # [H, W] binary

        # ── Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            pred = model(img_t).cpu().numpy()[0, 0]   # [H, W] float in [0,1]

        THRESH   = 0.5
        pred_bin = (pred > THRESH).astype(np.uint8)

        # ── Real supervised metrics ───────────────────────────────────────
        smooth = 1e-6

        # Dice coefficient
        inter = int((pred_bin & gt_np).sum())
        denom = int(pred_bin.sum() + gt_np.sum())
        dice  = (2 * inter + smooth) / (denom + smooth)

        # Intersection over Union (Jaccard index)
        union = int((pred_bin | gt_np).sum())
        iou   = (inter + smooth) / (union + smooth)

        # Pixel accuracy
        pixel_acc = float((pred_bin == gt_np).mean())

        # Tumour area (predicted)
        tumor_area_pct = float(pred_bin.mean() * 100)

        # ── Output images ─────────────────────────────────────────────────
        mask_b64    = _encode(Image.fromarray((pred_bin * 255).astype(np.uint8)))
        overlay_np  = np.array(image).copy()
        overlay_np[pred_bin == 1] = [255, 0, 0]
        overlay_b64 = _encode(Image.fromarray(overlay_np.astype(np.uint8)))

        return {
            "mask_b64":       mask_b64,
            "overlay_b64":    overlay_b64,
            "tumor_area_pct": round(tumor_area_pct, 3),
            "dice":           round(float(dice),      4),
            "iou":            round(float(iou),       4),
            "pixel_accuracy": round(pixel_acc,        4),
        }

    except Exception as exc:
        return {"error": str(exc)}


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint": ckpt_path,
    }
