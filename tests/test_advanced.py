import numpy as np
import pandas as pd
import torch
from neuroseg.model import build_model,loss_fn,load_checkpoint
from neuroseg.metrics import patient_summary,METRICS
from neuroseg.data import blob_hash
import hashlib

def test_attention_roundtrip(tmp_path):
    model=build_model("attention_unet",4)
    x=torch.rand(2,1,32,32)
    y=(x>.8).float()
    pred=model(x)
    assert pred.shape==y.shape
    loss_fn(pred,y).backward()
    p=tmp_path/"model.pth"
    torch.save({"model":model.state_dict(),"config":{"architecture":"attention_unet","width":4}},p)
    loaded,_=load_checkpoint(p)
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(loaded(x),model(x))

def test_patient_macro_is_not_slice_macro():
    rows=pd.DataFrame([{"patient_group":p,"tumor_type":"Glioma",**dict.fromkeys(METRICS,v)}
                        for p,v in [("a",1.),("a",1.),("a",1.),("b",0.)]])
    d=patient_summary(rows,draws=100)
    assert d["slice_macro"]["dice"]==.75
    assert d["patient_macro"]["dice"]==.5

def test_git_blob_hash(tmp_path):
    p=tmp_path/"sample";p.write_bytes(b"abc")
    assert blob_hash(p)==hashlib.sha1(b"blob 3\0abc").hexdigest()
