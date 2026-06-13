import torch
import numpy as np
from PIL import Image
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from model import UNetGenerator
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetGenerator().to(device)
model.load_state_dict(torch.load("best_generator.pth", map_location=device))
model.eval()

transform = T.ToTensor()


# folders
IMG_DIR = "dataset/images"
MASK_DIR = "dataset/masks"

dice_scores = []
ssim_scores = []
psnr_scores = []

def dice(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    return (2 * (pred * gt).sum()) / (pred.sum() + gt.sum() + 1e-8)


for file in os.listdir(IMG_DIR):

    img = Image.open(os.path.join(IMG_DIR, file)).convert("RGB")
    gt = Image.open(os.path.join(MASK_DIR, file)).convert("L")

    img = img.resize((256, 256))
    gt = gt.resize((256, 256))

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    gt = np.array(gt) / 255.0
    gt = (gt > 0.5).astype(np.uint8)

    pred_bin = (pred > 0.5).astype(np.uint8)

    # Dice
    d = dice(pred_bin, gt)
    dice_scores.append(d)

    # SSIM
    s = ssim(gt, pred, data_range=1)
    ssim_scores.append(s)

    # PSNR
    p = psnr(gt, pred)
    psnr_scores.append(p)


print("===== FINAL EVALUATION =====")
print("Dice:", np.mean(dice_scores))
print("SSIM:", np.mean(ssim_scores))
print("PSNR:", np.mean(psnr_scores))