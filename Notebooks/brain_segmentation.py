"""
api.py — FastAPI Backend for Brain Tumor Segmentation

Endpoints:
  POST   /api/predict              Upload MRI → prediction + metrics
  POST   /api/predict-with-mask    Upload MRI + GT mask → full metrics
  POST   /api/train/start          Start training in background
  POST   /api/train/stop           Stop training
  GET    /api/train/status         Current training state
  GET    /api/history              Full training history
  GET    /api/history/{epoch}      Single epoch record
  GET    /api/model/info           Model architecture summary
  GET    /api/results              Paginated prediction results
  GET    /api/results/{id}         Single prediction result
  DELETE /api/results/{id}         Delete result
  WS     /ws/training              Live training progress stream

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import (
    FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import (
    Column, DateTime, Float, Integer, JSON, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..inference import load_model, predict, b64_to_bytes
from ..train import Trainer, TrainConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)



DB_URL = "sqlite:///./brain_tumor.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class PredictionResult(Base):
    __tablename__ = "predictions"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename     = Column(String)
    created_at   = Column(DateTime, default=datetime.utcnow)
    tumor_pct    = Column(Float)
    has_gt       = Column(Integer, default=0)   # 1 if ground truth provided
    dice         = Column(Float, nullable=True)
    iou          = Column(Float, nullable=True)
    ssim         = Column(Float, nullable=True)
    psnr         = Column(Float, nullable=True)
    overlay_b64  = Column(Text)
    mask_b64     = Column(Text)
    extra        = Column(JSON, default=dict)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at= Column(DateTime, nullable=True)
    status     = Column(String, default="running")   # running | completed | stopped | failed
    config     = Column(JSON, default=dict)
    best_dice  = Column(Float, nullable=True)
    epochs_done= Column(Integer, default=0)
    history    = Column(JSON, default=list)


Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


training_state = {
    "status": "idle",          # idle | running | stopping
    "run_id": None,
    "epoch": None,
    "best_dice": 0.0,
    "latest_metrics": {},
}
training_stop_event = threading.Event()
ws_broadcast_queue: asyncio.Queue = None   # set in lifespan


def _broadcast(data: dict):
    """Thread-safe push to WebSocket queue."""
    global ws_broadcast_queue
    if ws_broadcast_queue is not None:
        try:
            asyncio.run_coroutine_threadsafe(
                ws_broadcast_queue.put(data),
                asyncio.get_event_loop(),
            )
        except Exception as e:
            log.warning(f"WS broadcast failed: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_broadcast_queue
    ws_broadcast_queue = asyncio.Queue()

    # Pre-load model if checkpoint exists
    ckpt = "checkpoints/best_model.pth"
    if Path(ckpt).exists():
        try:
            load_model(ckpt)
            log.info("Model pre-loaded.")
        except Exception as e:
            log.warning(f"Could not pre-load model: {e}")

    yield  # app runs here
    log.info("Shutting down.")


app = FastAPI(
    title="Brain Tumor Segmentation API",
    version="2.0.0",
    description="GAN-based MRI brain tumor segmentation with SSIM, PSNR, Dice metrics.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PredictResponse(BaseModel):
    id: str
    filename: str
    tumor_area_pct: float
    overlay_b64: str
    mask_b64: str
    has_ground_truth: bool
    metrics: Optional[dict] = None
    created_at: str


class TrainStartRequest(BaseModel):
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    epochs: int = 50
    batch_size: int = 8
    img_size: int = 256
    lr_gen: float = 2e-4
    lr_disc: float = 2e-4
    lambda_seg: float = 10.0
    early_stop_patience: int = 15


class TrainStatusResponse(BaseModel):
    status: str
    run_id: Optional[str]
    current_epoch: Optional[int]
    best_dice: float
    latest_metrics: dict


class ResultSummary(BaseModel):
    id: str
    filename: str
    created_at: str
    tumor_area_pct: float
    has_ground_truth: bool
    dice: Optional[float]
    ssim: Optional[float]
    psnr: Optional[float]
    iou: Optional[float]



@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_endpoint(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    checkpoint: str = Form("checkpoints/best_model.pth"),
):
    """Upload an MRI image and receive the predicted tumor mask + overlay."""
    if not Path(checkpoint).exists():
        raise HTTPException(404, f"Checkpoint not found: {checkpoint}")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(400, "Empty file.")

    try:
        result = predict(img_bytes, checkpoint_path=checkpoint, threshold=threshold)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    # Persist to DB
    db = SessionLocal()
    try:
        record = PredictionResult(
            filename=file.filename,
            tumor_pct=result["tumor_area_pct"],
            has_gt=0,
            overlay_b64=result["overlay_b64"],
            mask_b64=result["mask_b64"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        rid = record.id
        ts = record.created_at.isoformat()
    finally:
        db.close()

    return PredictResponse(
        id=rid,
        filename=file.filename,
        tumor_area_pct=result["tumor_area_pct"],
        overlay_b64=result["overlay_b64"],
        mask_b64=result["mask_b64"],
        has_ground_truth=False,
        metrics=None,
        created_at=ts,
    )


@app.post("/api/predict-with-mask", response_model=PredictResponse, tags=["Prediction"])
async def predict_with_mask(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    threshold: float = Form(0.5),
    checkpoint: str = Form("checkpoints/best_model.pth"),
):
    """Upload MRI + ground truth mask → returns full SSIM, PSNR, Dice, IoU metrics."""
    if not Path(checkpoint).exists():
        raise HTTPException(404, f"Checkpoint not found: {checkpoint}")

    img_bytes = await file.read()
    mask_bytes = await mask.read()

    try:
        result = predict(
            img_bytes,
            checkpoint_path=checkpoint,
            threshold=threshold,
            ground_truth=mask_bytes,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    metrics = result.get("metrics", {})
    db = SessionLocal()
    try:
        record = PredictionResult(
            filename=file.filename,
            tumor_pct=result["tumor_area_pct"],
            has_gt=1,
            dice=metrics.get("dice"),
            iou=metrics.get("iou"),
            ssim=metrics.get("ssim"),
            psnr=metrics.get("psnr"),
            overlay_b64=result["overlay_b64"],
            mask_b64=result["mask_b64"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        rid = record.id
        ts = record.created_at.isoformat()
    finally:
        db.close()

    return PredictResponse(
        id=rid,
        filename=file.filename,
        tumor_area_pct=result["tumor_area_pct"],
        overlay_b64=result["overlay_b64"],
        mask_b64=result["mask_b64"],
        has_ground_truth=True,
        metrics=metrics,
        created_at=ts,
    )


@app.get("/api/results/{result_id}/overlay", tags=["Prediction"])
async def get_overlay_image(result_id: str):
    """Return overlay PNG as binary."""
    db = SessionLocal()
    try:
        rec = db.get(PredictionResult, result_id)
        if not rec:
            raise HTTPException(404, "Result not found.")
        img_bytes = b64_to_bytes(rec.overlay_b64)
    finally:
        db.close()
    return StreamingResponse(iter([img_bytes]), media_type="image/png")



@app.get("/api/results", response_model=list[ResultSummary], tags=["Results"])
async def list_results(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
    has_gt: Optional[bool] = None,
):
    db = SessionLocal()
    try:
        q = db.query(PredictionResult).order_by(PredictionResult.created_at.desc())
        if has_gt is not None:
            q = q.filter(PredictionResult.has_gt == int(has_gt))
        recs = q.offset(skip).limit(limit).all()
        return [
            ResultSummary(
                id=r.id,
                filename=r.filename,
                created_at=r.created_at.isoformat(),
                tumor_area_pct=r.tumor_pct,
                has_ground_truth=bool(r.has_gt),
                dice=r.dice,
                ssim=r.ssim,
                psnr=r.psnr,
                iou=r.iou,
            )
            for r in recs
        ]
    finally:
        db.close()


@app.get("/api/results/{result_id}", tags=["Results"])
async def get_result(result_id: str):
    db = SessionLocal()
    try:
        rec = db.get(PredictionResult, result_id)
        if not rec:
            raise HTTPException(404, "Not found.")
        return {
            "id": rec.id,
            "filename": rec.filename,
            "created_at": rec.created_at.isoformat(),
            "tumor_area_pct": rec.tumor_pct,
            "has_ground_truth": bool(rec.has_gt),
            "metrics": {"dice": rec.dice, "iou": rec.iou, "ssim": rec.ssim, "psnr": rec.psnr},
            "overlay_b64": rec.overlay_b64,
            "mask_b64": rec.mask_b64,
        }
    finally:
        db.close()


@app.delete("/api/results/{result_id}", tags=["Results"])
async def delete_result(result_id: str):
    db = SessionLocal()
    try:
        rec = db.get(PredictionResult, result_id)
        if not rec:
            raise HTTPException(404, "Not found.")
        db.delete(rec)
        db.commit()
        return {"deleted": result_id}
    finally:
        db.close()

@app.post("/api/train/start", tags=["Training"])
async def start_training(req: TrainStartRequest):
    global training_state, training_stop_event

    if training_state["status"] == "running":
        raise HTTPException(409, "Training is already running.")

    cfg = TrainConfig(
        train_dir=req.train_dir,
        val_dir=req.val_dir,
        epochs=req.epochs,
        batch_size=req.batch_size,
        img_size=req.img_size,
        lr_gen=req.lr_gen,
        lr_disc=req.lr_disc,
        lambda_seg=req.lambda_seg,
        early_stop_patience=req.early_stop_patience,
    )

    # Create DB run record
    db = SessionLocal()
    try:
        run = TrainingRun(config=asdict(cfg))
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id
    finally:
        db.close()

    training_stop_event.clear()
    training_state.update(status="running", run_id=run_id, epoch=None, best_dice=0.0, latest_metrics={})

    def _train_thread():
        global training_state
        try:
            def on_progress(epoch_log):
                training_state["epoch"] = epoch_log["epoch"]
                training_state["latest_metrics"] = epoch_log
                if epoch_log["val"]["dice"] > training_state["best_dice"]:
                    training_state["best_dice"] = epoch_log["val"]["dice"]
                _broadcast({"type": "progress", "data": epoch_log})

            trainer = Trainer(cfg, progress_callback=on_progress)
            history = trainer.run(stop_event=training_stop_event)

            # Update DB
            db2 = SessionLocal()
            try:
                r = db2.get(TrainingRun, run_id)
                r.status = "completed"
                r.finished_at = datetime.utcnow()
                r.best_dice = trainer.best_dice
                r.epochs_done = len(history)
                r.history = history
                db2.commit()
            finally:
                db2.close()

            training_state["status"] = "idle"
            _broadcast({"type": "done", "best_dice": trainer.best_dice})

        except Exception as e:
            log.exception(f"Training failed: {e}")
            training_state["status"] = "idle"
            _broadcast({"type": "error", "message": str(e)})
            db3 = SessionLocal()
            try:
                r = db3.get(TrainingRun, run_id)
                if r:
                    r.status = "failed"
                    r.finished_at = datetime.utcnow()
                    db3.commit()
            finally:
                db3.close()

    t = threading.Thread(target=_train_thread, daemon=True)
    t.start()
    return {"run_id": run_id, "status": "started", "message": "Training started in background."}


@app.post("/api/train/stop", tags=["Training"])
async def stop_training():
    global training_state, training_stop_event
    if training_state["status"] != "running":
        raise HTTPException(400, "No training is running.")
    training_stop_event.set()
    training_state["status"] = "stopping"
    return {"message": "Stop signal sent."}


@app.get("/api/train/status", response_model=TrainStatusResponse, tags=["Training"])
async def get_training_status():
    return TrainStatusResponse(
        status=training_state["status"],
        run_id=training_state["run_id"],
        current_epoch=training_state["epoch"],
        best_dice=training_state["best_dice"],
        latest_metrics=training_state["latest_metrics"],
    )


@app.get("/api/history", tags=["Training"])
async def get_all_history():
    db = SessionLocal()
    try:
        runs = db.query(TrainingRun).order_by(TrainingRun.started_at.desc()).all()
        return [
            {
                "id": r.id,
                "started_at": r.started_at.isoformat(),
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "status": r.status,
                "best_dice": r.best_dice,
                "epochs_done": r.epochs_done,
                "config": r.config,
            }
            for r in runs
        ]
    finally:
        db.close()


@app.get("/api/history/{run_id}/epochs", tags=["Training"])
async def get_epoch_history(run_id: str):
    db = SessionLocal()
    try:
        run = db.get(TrainingRun, run_id)
        if not run:
            raise HTTPException(404, "Run not found.")
        return run.history or []
    finally:
        db.close()


@app.get("/api/model/info", tags=["Model"])
async def model_info(checkpoint: str = "checkpoints/best_model.pth"):
    info = {
        "checkpoint_exists": Path(checkpoint).exists(),
        "checkpoint_path": checkpoint,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "cuda_available": torch.cuda.is_available(),
    }
    if info["checkpoint_exists"]:
        try:
            ckpt = torch.load(checkpoint, map_location="cpu")
            info["epoch"] = ckpt.get("epoch")
            info["best_dice"] = ckpt.get("best_dice")
            info["config"] = ckpt.get("config", {})
        except Exception as e:
            info["load_error"] = str(e)

    from model import UNetGenerator, PatchDiscriminator
    gen = UNetGenerator()
    disc = PatchDiscriminator()
    gen_params = sum(p.numel() for p in gen.parameters()) / 1e6
    disc_params = sum(p.numel() for p in disc.parameters()) / 1e6
    info["generator_params_M"] = round(gen_params, 2)
    info["discriminator_params_M"] = round(disc_params, 2)
    return info



active_ws_connections: set[WebSocket] = set()


@app.websocket("/ws/training")
async def websocket_training(ws: WebSocket):
    await ws.accept()
    active_ws_connections.add(ws)
    try:
        # Send current state on connect
        await ws.send_json({"type": "state", "data": {
            "status": training_state["status"],
            "epoch": training_state["epoch"],
            "best_dice": training_state["best_dice"],
            "latest_metrics": training_state["latest_metrics"],
        }})

        # Fan out broadcast messages
        async def _sender():
            while True:
                msg = await ws_broadcast_queue.get()
                dead = set()
                for conn in list(active_ws_connections):
                    try:
                        await conn.send_json(msg)
                    except Exception:
                        dead.add(conn)
                active_ws_connections -= dead

        sender_task = asyncio.create_task(_sender())

        # Keep alive — echo pings
        while True:
            try:
                text = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if text == "ping":
                    await ws.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await ws.send_json({"type": "heartbeat"})
            except WebSocketDisconnect:
                break

        sender_task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        active_ws_connections.discard(ws)



@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "training": training_state["status"],
        "cuda": torch.cuda.is_available(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/", tags=["System"])
async def root():
    return {"message": "Brain Tumor Segmentation API v2.0", "docs": "/docs"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)