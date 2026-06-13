"""
app.py — Minimal FastAPI wrapper (real model, no fake metrics)
==============================================================
This file is a lightweight alternative to api.py.
It loads the real trained UNet and returns honest outputs only.

For the full API (including /api/evaluate with ground-truth Dice/IoU),
use api.py instead.

Run:
    uvicorn app:app --reload --port 8000
"""

import io
import base64
import os

import numpy as np
from PIL import Image

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model import UNetGenerator

app = FastAPI(title="NeuroSeg", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load real model ──────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "best_unet.pth"
LEGACY_PATH     = "best_generator.pth"
ckpt_path       = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else LEGACY_PATH

if not os.path.exists(ckpt_path):
    raise RuntimeError(
        f"No checkpoint found at '{ckpt_path}'.\n"
        "Train first with:  python train_unet.py"
    )

model = UNetGenerator().to(DEVICE)
ckpt  = torch.load(ckpt_path, map_location=DEVICE)

if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
    print(f"[INFO] Loaded {ckpt_path}  "
          f"(epoch {ckpt.get('epoch','?')}, "
          f"val Dice {ckpt.get('best_dice', 0):.4f})")
else:
    model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded raw state_dict from {ckpt_path}")

model.eval()
print(f"[INFO] Device : {DEVICE}")


# ─── Real inference helper ────────────────────────────────────────────────────
def model_predict(image_bytes: bytes) -> dict:
    """
    Run the real UNet on raw image bytes.

    Returns
    -------
    dict with keys:
        tumor_area_pct    – % of pixels predicted as tumour (0–100)
        mean_confidence   – mean sigmoid value over all pixels
        tumor_confidence  – mean sigmoid value inside predicted tumour mask
        overlay_b64       – base64 PNG of image with tumour highlighted red
        mask_b64          – base64 PNG of binary segmentation mask
    """
    # Pre-process
    img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_256 = img.resize((256, 256), Image.BILINEAR)
    img_np  = np.array(img_256).astype(np.float32) / 255.0
    img_t   = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        pred = model(img_t).cpu().numpy()[0, 0]   # [H, W] float in [0,1]

    # Binary mask
    THRESH = 0.5
    mask   = (pred > THRESH).astype(np.uint8)

    # Honest statistics — NO ground truth needed
    tumor_area_pct   = float(mask.mean() * 100)
    mean_confidence  = float(pred.mean())
    tumor_confidence = float(pred[mask == 1].mean()) if mask.sum() > 0 else 0.0

    # Overlay image
    overlay         = np.array(img_256).copy()
    overlay[mask == 1] = [255, 0, 0]

    def to_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return {
        "tumor_area_pct":   round(tumor_area_pct,   3),
        "mean_confidence":  round(mean_confidence,  4),
        "tumor_confidence": round(tumor_confidence, 4),
        "overlay_b64":      to_b64(overlay),
        "mask_b64":         to_b64(mask * 255),
    }


# ─── Route ───────────────────────────────────────────────────────────────────
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a brain MRI image → get tumour segmentation + confidence stats.

    Supervised metrics (Dice, IoU, Accuracy, SSIM, PSNR) are NOT included
    because they require a ground-truth mask.  Use /api/evaluate in api.py
    or run evaluate_unet.py on your labelled test set.
    """
    image_bytes = await file.read()
    try:
        result = model_predict(image_bytes)
        return result
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "checkpoint": ckpt_path}
