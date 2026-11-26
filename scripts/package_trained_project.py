"""Package verified real checkpoints and source without redistributing MRI data."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
import torch
from fastapi.testclient import TestClient

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",required=True,type=Path)
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--dest",required=True,type=Path)
    args=p.parse_args()
    results=args.results.resolve()
    source=Path(__file__).resolve().parents[1]
    dest=args.dest.resolve()
    if dest.exists(): raise FileExistsError(dest)
    summary=json.loads((results/"test_metrics.json").read_text())
    selection=json.loads((results/"selection.json").read_text())
    checkpoint=results/"best_model.pth"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==summary["checkpoint_sha256"]
    assert summary["architecture"]==selection["architecture"]
    assert summary["test_patients"]==24 and summary["test_images"]==307
    assert summary["best_epoch"]>=1
    assert 0<=summary["patient_macro"]["dice"]<=1
    table=pd.read_csv(results/"model_comparison.csv")
    assert set(table.architecture)=={"unet","attention_unet"}
    # Functional smoke test on an actual held-out MRI slice and its binary mask.
    import os
    os.environ["NEUROSEG_RESULTS"]=str(results)
    os.environ["NEUROSEG_LOAD_CANDIDATES"]="1"
    os.environ.pop("NEUROSEG_CHECKPOINT",None)
    from neuroseg.api import app
    img=next(iter(sorted((args.data/"test"/"images").glob("*.png"))))
    gt=args.data/"test"/"masks"/img.name
    with TestClient(app) as client:
        assert client.get("/health").json()=={"ready":True}
        info=client.get("/api/model/info").json()
        assert {m["architecture"] for m in info["models"]}=={"unet","attention_unet"}
        assert client.get("/api/model/performance").status_code==200
        smoke=[]
        for m in info["models"]:
            r=client.post("/api/predict",
                          files={"file":("sample.png",img.read_bytes(),"image/png"),
                                 "ground_truth":("mask.png",gt.read_bytes(),"image/png")},
                          data={"source":"preprocessed","architecture":m["architecture"]})
            assert r.status_code==200,r.text
            body=r.json()
            assert body["mask_b64"] and 0<=body["comparison"]["dice"]<=1
            smoke.append({"architecture":m["architecture"],"http_status":r.status_code,
                          "prediction_returned":True})
    dest.mkdir(parents=True)
    for folder in ("neuroseg","deploy"):
        shutil.copytree(source/folder,dest/folder,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for name in ("requirements-v2.txt","requirements-mlops.txt","Start-NeuroSeg.bat","launch_neuroseg.py",
                 "README.md","Dockerfile.v2","compose.yaml"):
        shutil.copy2(source/name,dest/name)
    target=dest/"neuroseg_results"
    target.mkdir()
    for name in ("best_model.pth","selection.json","model_comparison.csv","test_metrics.json",
                 "training_history.csv","training_curves.png","registry.json"):
        f=results/name
        if f.exists(): shutil.copy2(f,target/name)
    for row in table.sort_values("validation_dice",ascending=False).drop_duplicates("architecture").itertuples():
        folder=target/row.candidate
        folder.mkdir(parents=True)
        shutil.copy2(results/row.candidate/"best_unet.pth",folder/"best_unet.pth")
    proof={"trained_checkpoint_sha256":summary["checkpoint_sha256"],"test_patients":summary["test_patients"],
           "test_images":summary["test_images"],"smoke_tests":smoke,
           "note":"Actual held-out image used in test; MRI images and patient-level records not included in this distribution."}
    (dest/"verification.json").write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2))

if __name__=="__main__": main()
