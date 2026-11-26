import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import torch
from PIL import Image,UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from .model import UNet
from .data import normalize

NOTICE="Research prototype only. Not a diagnostic tool. A predicted empty mask does not rule out disease."

@asynccontextmanager
async def lifespan(app):
    app.state.model=None
    path=Path(os.getenv("NEUROSEG_CHECKPOINT","neuroseg_results/best_unet.pth"))
    if path.is_file():
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck=torch.load(path,map_location=device,weights_only=True)
        model=UNet(ck["config"]["width"]).to(device)
        model.load_state_dict(ck["model"],strict=True); model.eval()
        app.state.model=model; app.state.device=device; app.state.cfg=ck["config"]
        app.state.info={"epoch":ck["epoch"],"validation_dice":ck["val_dice"]}
    yield

app=FastAPI(title="NeuroSeg V2",lifespan=lifespan)

def encode(arr):
    buf=io.BytesIO(); Image.fromarray(arr).save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def predict(raw):
    if app.state.model is None: raise HTTPException(503,"Model checkpoint not installed")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in {"PNG","JPEG","TIFF"} or im.width*im.height>16_000_000:
                raise ValueError("Unsupported format or dimensions")
            image=np.array(im if im.mode in {"L","I","I;16","F"} else im.convert("L"))
        norm=normalize(image)
    except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError):
        raise HTTPException(400,"Invalid or unsupported MRI image")
    cfg=app.state.cfg; size=cfg["size"]
    resized=np.array(Image.fromarray(norm).resize((size,size),Image.Resampling.BILINEAR))
    x=torch.from_numpy(resized.copy()).float()[None,None].to(app.state.device)/255
    with torch.inference_mode(): p=app.state.model(x).sigmoid()[0,0].cpu().numpy()
    mask=Image.fromarray((p>cfg["threshold"]).astype(np.uint8)*255).resize((norm.shape[1],norm.shape[0]),Image.Resampling.NEAREST)
    mask=np.asarray(mask); overlay=np.repeat(norm[:,:,None],3,axis=2).copy()
    overlay[mask>0]=(overlay[mask>0]*.55+np.array([255,0,0])*.45).astype(np.uint8)
    return {"predicted_foreground_area_pct":float((mask>0).mean()*100),"mask_b64":encode(mask),"overlay_b64":encode(overlay),"notice":NOTICE}

@app.get("/health")
def health():
    return {"ready":app.state.model is not None}

@app.get("/api/model/info")
def info():
    if app.state.model is None: raise HTTPException(503,"Model checkpoint not installed")
    return app.state.info

@app.post("/api/predict")
async def predict_endpoint(file:UploadFile=File(...)):
    raw=await file.read(10*1024*1024+1)
    if len(raw)>10*1024*1024: raise HTTPException(413,"Upload exceeds 10 MB")
    if not raw: raise HTTPException(400,"Empty file")
    return await run_in_threadpool(predict,raw)

@app.get("/",response_class=HTMLResponse)
def home():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")
