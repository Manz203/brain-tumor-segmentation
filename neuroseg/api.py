import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .model import load_checkpoint
from .predictor import predict, LIMIT
from .monitoring import Monitor

@asynccontextmanager
async def lifespan(app):
    app.state.models={}
    app.state.monitor=Monitor()
    root=Path(os.getenv("NEUROSEG_RESULTS","neuroseg_results"))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device=device
    paths=[]
    explicit=os.getenv("NEUROSEG_CHECKPOINT")
    if explicit: paths.append(Path(explicit))
    else:
        paths.append(root/"best_model.pth" if (root/"best_model.pth").exists() else root/"best_unet.pth")
        if os.getenv("NEUROSEG_LOAD_CANDIDATES","0")=="1":
            paths.extend(sorted(root.glob("candidates/*/best_unet.pth")))
    for path in paths:
        if not path.is_file(): continue
        model,ck=load_checkpoint(path,device)
        cfg=ck["config"]
        key=cfg.get("architecture","unet")
        # Keep the highest validation-Dice checkpoint per architecture.
        existing=app.state.models.get(key)
        if existing and existing[2]["validation_dice"]>=ck["val_dice"]: continue
        info={"architecture":key,"epoch":ck["epoch"],"validation_dice":ck["val_dice"],
              "size":cfg["size"],"threshold":cfg.get("threshold",.5),
              "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        app.state.models[key]=(model,cfg,info)
    app.state.default=next(iter(app.state.models),None)
    app.state.report={}
    app.state.sample_root=Path(__file__).with_name("samples")
    metrics=root/"test_metrics.json"
    if metrics.is_file():
        report=json.loads(metrics.read_text())
        key=report.get("architecture")
        if key in app.state.models and report.get("checkpoint_sha256")==app.state.models[key][2]["checkpoint_sha256"]:
            app.state.report=report
    yield

app=FastAPI(title="NeuroSeg V2",lifespan=lifespan)
app.mount("/static",StaticFiles(directory=Path(__file__).with_name("static")),name="static")

@app.get("/health")
def health():
    return {"ready":bool(app.state.models)}

@app.get("/api/model/info")
def info():
    if not app.state.models: raise HTTPException(503,"Model checkpoint not installed")
    return {"default":app.state.default,"models":[v[2] for v in app.state.models.values()],
            **app.state.models[app.state.default][2]}

@app.get("/api/model/performance")
def performance(architecture:str|None=None):
    if not app.state.report or (architecture and architecture!=app.state.report.get("architecture")): raise HTTPException(404,"No verified evaluation report for installed checkpoint")
    return app.state.report

@app.get("/api/samples")
def samples():
    manifest=app.state.sample_root/"manifest.json"
    return json.loads(manifest.read_text()) if manifest.is_file() else {"samples":[]}

@app.get("/api/samples/{sample_id}/{kind}")
def sample_file(sample_id:str,kind:str):
    allowed={s["id"] for s in samples()["samples"]}
    if sample_id not in allowed or kind not in {"image","mask"}:
        raise HTTPException(404,"Sample unavailable")
    path=app.state.sample_root/f"{sample_id}_{kind}.png"
    if not path.is_file(): raise HTTPException(404,"Sample unavailable")
    return FileResponse(path,media_type="image/png")

@app.get("/api/monitoring")
def monitoring(): return app.state.monitor.snapshot()

@app.get("/metrics",response_class=PlainTextResponse)
def metrics(): return app.state.monitor.prometheus()

@app.post("/api/predict")
async def predict_endpoint(file:UploadFile=File(...),ground_truth:UploadFile|None=File(None),
                           architecture:str|None=Form(None),source:str=Form("raw")):
    started=time.perf_counter()
    ok=False
    try:
        key=architecture or app.state.default
        if key not in app.state.models: raise HTTPException(503,"Requested model is not installed")
        raw=await file.read(LIMIT+1)
        gt=await ground_truth.read(LIMIT+1) if ground_truth else None
        if len(raw)>LIMIT or (gt is not None and len(gt)>LIMIT): raise HTTPException(413,"Each upload must be 10 MB or less")
        if not raw or (gt is not None and not gt): raise HTTPException(400,"Empty file")
        model,cfg,model_info=app.state.models[key]
        try:
            result=await run_in_threadpool(predict,model,cfg,app.state.device,raw,source,gt)
        except (ValueError,OSError,Image.DecompressionBombError) as exc:
            raise HTTPException(400,str(exc)) from exc
        ok=True
        return {**result,"model":model_info,"request_ms":(time.perf_counter()-started)*1000}
    finally:
        app.state.monitor.observe(time.perf_counter()-started,ok)
        await file.close()
        if ground_truth: await ground_truth.close()

@app.get("/",response_class=HTMLResponse)
def home():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")
