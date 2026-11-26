"""Model inference on de-identified 2D image exports, not DICOM volumes."""
import base64
import io
import numpy as np
import torch
from PIL import Image
from .data import normalize

NOTICE="Research only. Not for diagnosis. An empty predicted mask does not rule out disease."
LIMIT=10*1024*1024

def decode(raw, mask=False):
    with Image.open(io.BytesIO(raw)) as im:
        if im.format not in {"PNG","JPEG","TIFF"} or im.width*im.height>16_000_000 or getattr(im,"n_frames",1)!=1:
            raise ValueError("Use a single supported 2D image up to 16 million pixels")
        if mask: return np.array(im.convert("L"))
        return np.array(im if im.mode in {"L","I","I;16","F"} else im.convert("L"))

def encode(arr):
    buf=io.BytesIO()
    Image.fromarray(arr).save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def predict(model, cfg, device, raw, source="raw", ground_truth=None):
    img=decode(raw)
    if source=="preprocessed":
        if img.dtype!=np.uint8:
            raise ValueError("Preprocessed input must be 8-bit grayscale")
        norm=img
    elif source=="raw":
        norm=normalize(img)
    else: raise ValueError("Unknown input preprocessing mode")
    size=cfg["size"]
    resized=np.array(Image.fromarray(norm).resize((size,size),Image.Resampling.BILINEAR))
    x=torch.from_numpy(resized.copy()).float()[None,None].to(device)/255
    with torch.inference_mode():
        p=model(x).float().sigmoid()[0,0].cpu().numpy()
    mask=np.asarray(Image.fromarray((p>cfg.get("threshold",.5)).astype(np.uint8)*255)
                    .resize((norm.shape[1],norm.shape[0]),Image.Resampling.NEAREST))
    result={"image_b64":encode(norm),"mask_b64":encode(mask),
            "predicted_foreground_area_pct":float((mask>0).mean()*100),
            "notice":NOTICE}
    if ground_truth is not None:
        gt=decode(ground_truth,mask=True)
        if gt.shape!=mask.shape: raise ValueError("Ground truth must match original MRI dimensions")
        if not np.isin(gt,[0,1,255]).all(): raise ValueError("Ground truth must be a binary PNG/TIFF mask (0/1 or 0/255)")
        t=gt>0; pred=mask>0
        overlap=int((t&pred).sum()); union=int((t|pred).sum()); denom=int(t.sum()+pred.sum())
        result["ground_truth_b64"]=encode(t.astype(np.uint8)*255)
        result["comparison"]={"dice":2*overlap/denom if denom else 1.0,"iou":overlap/union if union else 1.0,
                              "empty_both":not bool(union)}
    return result
