# Brain Tumor Segmentation using U-Net

A deep learning system for automatic brain tumor segmentation from MRI scans, using a U-Net encoder-decoder trained with a supervised Dice + BCE loss. Includes a single-image inference script, a FastAPI backend, and a web-based interface for uploading scans and viewing predicted segmentation masks.

## Overview

Brain tumor segmentation is a critical task in medical imaging, helping clinicians localize and quantify tumor regions from MRI scans. This project implements an encoder-decoder U-Net architecture and trains it directly against ground-truth masks with a combined Dice + Binary Cross-Entropy loss.

A PatchGAN discriminator (`PatchDiscriminator` in `model.py`) is included in the codebase as a starting point for future adversarial-training experiments, but **the model that produced the results below was trained without it** — plain supervised training performed better in practice (see `train_unet.py`'s docstring). The "Future Work" section below covers re-introducing adversarial training properly.

## Architecture

**Generator (UNetGenerator)** — A 4-level U-Net with skip connections:
- Encoder: 4 downsampling blocks (64 → 128 → 256 → 512 channels), each with two Conv-BatchNorm-ReLU layers followed by max pooling
- Bottleneck: 1024-channel convolutional block
- Decoder: 4 upsampling blocks using transposed convolutions, with skip connections concatenating corresponding encoder features
- Output: 1-channel sigmoid activation producing a binary segmentation mask
- **Total trainable parameters: ~31.04M**

**Discriminator (PatchDiscriminator)** — A PatchGAN-style convolutional discriminator that takes the concatenated image and mask (4 channels) and outputs a patch-wise real/fake prediction. Defined in `model.py` but not used by the current training pipeline.

## Dataset

- Training samples: 2,799
- Validation samples: 1,119
- Images resized to 256×256
- Custom train/val/test split via `split_dataset.py` (seeded for reproducibility; writes `split_manifest.json` so the exact split can be audited later)

## Training

Trained for 50 epochs on GPU (CUDA) with a learning rate of 1e-3, Dice+BCE loss, `ReduceLROnPlateau` scheduling, and early stopping on validation Dice.

Validation Dice score progression (selected epochs):

| Epoch | Train Dice | Val Dice |
|-------|-----------|----------|
| 1  | 0.2086 | 0.2788 |
| 5  | 0.4571 | 0.4775 |
| 10 | 0.5585 | 0.5848 |
| 12 | 0.5827 | **0.6154** |
| 14 | 0.5926 | **0.6103** |

Best validation Dice achieved: **~0.62**.

## Project Structure

```
brain-tumor-segmentation/
├── model.py            # U-Net generator (used) and PatchGAN discriminator (unused, future work)
├── train_unet.py        # Training script — the canonical training entrypoint
├── evaluate_unet.py      # Full test-set evaluation with Dice/IoU/Acc/SSIM/PSNR + visual grid
├── inference.py          # Run inference on a single new MRI image
├── split_dataset.py      # Train/val/test dataset splitting (seeded, writes a manifest)
├── api.py                # Full FastAPI backend — /api/predict + /api/evaluate
├── app.py                # Minimal FastAPI backend — /api/predict only
├── index.html            # Frontend interface (JS is inline in this file)
├── Notebooks/             # Exploratory notebooks and experiments
└── requirements.txt
```

## Setup & Usage

### Install dependencies
```
pip install -r requirements.txt
```

### Split the dataset
```
python split_dataset.py --base_dir Dataset
```

### Train the model
```
python train_unet.py
```

### Run inference on a new image
```
python inference.py --image path/to/scan.png
```
Saves `<name>_mask.png` and `<name>_overlay.png` next to the input (or to `--out_dir` if given). Prints tumour area and model confidence — no ground truth needed.

### Evaluate on a labelled test set
```
python evaluate_unet.py --checkpoint best_unet.pth --test_dir Dataset/test
```
Prints Dice / IoU / Pixel Accuracy / SSIM / PSNR and saves a visual grid to `eval_results.png`.

### Run the web app
```
python app.py
```
or, for the full API with the `/api/evaluate` endpoint:
```
uvicorn api:app --reload --port 8000
```
Then open `index.html` in your browser to upload MRI scans and view predicted segmentation masks.

## Model Weights

Trained model checkpoints (`best_unet.pth`, `final_generator.pth`, `best_generator.pth`) are not included in this repository due to size (~125MB each), but can be provided on request or via [Google Drive link].

## Future Work

- Re-introduce adversarial training: wire `PatchDiscriminator` back into the training loop with a proper adversarial loss weight, and A/B test it against the current supervised-only baseline (0.62 Dice) instead of assuming it will help
- Extend training beyond 50 epochs with a more aggressive LR schedule
- Add multi-class segmentation for tumor sub-regions (edema, enhancing tumor, necrotic core)
- Deploy as a hosted web demo (e.g. Hugging Face Spaces)
