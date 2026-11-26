"""Model inference on de-identified 2D image exports, not DICOM volumes."""
import base64
import io
import time
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
    if x.is_cuda: torch.cuda.synchronize(x.device)
    start=time.perf_counter()
    with torch.inference_mode():
        p=model(x).float().sigmoid()[0,0].cpu().numpy()
    if x.is_cuda: torch.cuda.synchronize(x.device)
    inference_ms=(time.perf_counter()-start)*1000
    mask=np.asarray(Image.fromarray((p>cfg.get("threshold",.5)).astype(np.uint8)*255)
                    .resize((norm.shape[1],norm.shape[0]),Image.Resampling.NEAREST))
    result={"image_b64":encode(norm),"mask_b64":encode(mask),
            "predicted_foreground_area_pct":float((mask>0).mean()*100),
            "notice":NOTICE,"inference_ms":inference_ms}
    if ground_truth is not None:
        gt=decode(ground_truth,mask=True)
        if gt.shape!=mask.shape: raise ValueError("Ground truth must match original MRI dimensions")
        if not np.isin(gt,[0,1,255]).all(): raise ValueError("Ground truth must be a binary PNG/TIFF mask (0/1 or 0/255)")
        t=gt>0; pred=mask>0
        comparison,error_map=compare_masks(pred,t)
        result["ground_truth_b64"]=encode(t.astype(np.uint8)*255)
        result["comparison"]=comparison
        result["error_map_b64"]=encode(error_map)
    return result

def compare_masks(pred,t):
    """Native uploaded-image resolution. Undefined ratios are JSON null, not fake zeros."""
    pred=np.asarray(pred,dtype=bool); t=np.asarray(t,dtype=bool)
    if pred.shape!=t.shape: raise ValueError("Mask dimensions differ")
    tp=int((pred&t).sum()); fp=int((pred&~t).sum())
    fn=int((~pred&t).sum()); tn=int((~pred&~t).sum())
    def ratio(a,b): return a/b if b else None
    union=tp+fp+fn
    scores={"dice":2*tp/(2*tp+fp+fn) if union else 1.0,
            "iou":tp/union if union else 1.0,
            "precision":ratio(tp,tp+fp),"recall":ratio(tp,tp+fn),
            "specificity":ratio(tn,tn+fp),"empty_both":not bool(union),
            "tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "ground_truth_area_pct":100*(tp+fn)/pred.size}
    error=np.zeros((*pred.shape,3),dtype=np.uint8)
    error[pred&t]=(80,220,130)
    error[pred&~t]=(255,87,105)
    error[~pred&t]=(89,169,255)
    return scores,error
