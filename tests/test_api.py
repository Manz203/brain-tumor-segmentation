import io
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient
from neuroseg.api import app
from neuroseg.model import build_model

def png(arr):
    buf=io.BytesIO(); Image.fromarray(arr).save(buf,format="PNG"); return buf.getvalue()

def test_api_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROSEG_RESULTS",str(tmp_path))
    monkeypatch.delenv("NEUROSEG_CHECKPOINT",raising=False)
    with TestClient(app) as client:
        assert client.get("/").status_code==200
        assert client.get("/health").json()=={"ready":False}
        assert client.get("/api/model/info").status_code==503

def test_prediction_and_metrics(tmp_path,monkeypatch):
    model=build_model("attention_unet",4)
    ck=tmp_path/"test.pth"
    # Synthetic checkpoint verifies wiring, not model quality.
    torch.save({"model":model.state_dict(),"config":{"architecture":"attention_unet","width":4,"size":32,"threshold":.5},
                "epoch":0,"val_dice":0.},ck)
    monkeypatch.setenv("NEUROSEG_CHECKPOINT",str(ck))
    monkeypatch.setenv("NEUROSEG_RESULTS",str(tmp_path))
    img=png(np.arange(1024,dtype=np.uint8).reshape(32,32))
    gt=png(np.zeros((32,32),dtype=np.uint8))
    with TestClient(app) as client:
        assert client.get("/health").json()=={"ready":True}
        assert client.get("/api/model/performance").status_code==404
        r=client.post("/api/predict",files={"file":("mri.png",img,"image/png"),"ground_truth":("mask.png",gt,"image/png")},
                      data={"source":"preprocessed"})
        assert r.status_code==200,r.text
        assert 0<=r.json()["comparison"]["dice"]<=1
        assert "mask_b64" in r.json()
        assert "error_map_b64" in r.json()
        assert r.json()["inference_ms"]>=0
        assert r.json()["comparison"]["tp"]+r.json()["comparison"]["fp"]+r.json()["comparison"]["fn"]+r.json()["comparison"]["tn"]==1024
        assert client.post("/api/predict",files={"file":("bad.png",b"bad","image/png")}).status_code==400
        assert client.get("/api/monitoring").json()["errors"]==1
        assert "neuroseg_predictions_total 2" in client.get("/metrics").text
