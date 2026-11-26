"""Evaluate a frozen checkpoint, preserving the held-out patient split."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import validate, MRIDataset
from .model import load_checkpoint, scores
from .metrics import patient_summary

def evaluate_checkpoint(checkpoint, data, out, device=None, batch=8, workers=2):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = validate(data)
    model, ck = load_checkpoint(checkpoint, device)
    cfg = ck["config"]
    digest = hashlib.sha256((data/"dataset_manifest.csv").read_bytes()).hexdigest()
    if cfg["manifest_sha256"] != digest:
        raise ValueError("Checkpoint was trained with a different split manifest")
    loader = DataLoader(MRIDataset(data, df, "test", cfg["size"]), batch_size=batch,
                        shuffle=False, num_workers=workers)
    rows, examples = [], []
    threshold = cfg.get("threshold", .5)
    with torch.inference_mode():
        for x, y, names, groups, types in loader:
            logits = model(x.to(device)).float()
            metrics = scores(logits, y.to(device), threshold)
            pred = (logits.sigmoid() > threshold).cpu().numpy()
            for i, name in enumerate(names):
                row = {"filename": name, "patient_group": groups[i], "tumor_type": types[i]}
                row.update({k: float(v[i]) for k,v in metrics.items()})
                rows.append(row)
                examples.append((row["dice"], x[i,0].numpy(), y[i,0].numpy(), pred[i,0], name))
    result = pd.DataFrame(rows)
    result.to_csv(out/"per_sample_metrics.csv", index=False)
    summary = {"architecture": cfg.get("architecture", "unet"), "best_epoch": ck["epoch"],
               "best_validation_dice": ck["val_dice"], "test_images": len(result),
               "test_patients": result.patient_group.nunique(), "manifest_sha256": digest,
               "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               **patient_summary(result),
               "limitations": "Tumor-positive 2D slices only; no external or clinical validation. Bootstrap intervals are descriptive."}
    (out/"test_metrics.json").write_text(json.dumps(summary, indent=2))
    examples.sort(key=lambda x:x[0])
    selected = examples[:3] + examples[-3:]
    fig, axes = plt.subplots(len(selected), 4, figsize=(12,3*len(selected)), squeeze=False)
    for ax,(dice,img,gt,pred,name) in zip(axes,selected):
        ax[0].imshow(img,cmap="gray"); ax[1].imshow(gt,cmap="gray"); ax[2].imshow(pred,cmap="gray")
        ax[3].imshow(img,cmap="gray")
        ax[3].imshow(np.ma.masked_where(~pred,pred),cmap="autumn",alpha=.45)
        for a,t in zip(ax,[name,"Ground truth",f"Prediction Dice={dice:.3f}","Overlay"]):
            a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(out/"segmentation_examples.png",dpi=120); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    evaluate_checkpoint(args.checkpoint,args.data,args.out)

if __name__=="__main__": main()
