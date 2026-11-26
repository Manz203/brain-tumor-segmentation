import numpy as np
import torch
from neuroseg.model import UNet,loss_fn,scores
from neuroseg.data import normalize

def test_forward_backward():
    m=UNet(8)
    x=torch.rand(2,1,64,64)
    y=(x>.8).float()
    logits=m(x)
    assert logits.shape==y.shape
    loss=loss_fn(logits,y)
    assert torch.isfinite(loss)
    loss.backward()

def test_metrics():
    y=torch.tensor([[[[0.,1.],[1.,0.]]]])
    logits=(y*2-1)*20
    assert scores(logits,y)["dice"].item()==1.
    assert scores(-logits,y)["dice"].item()<1e-5

def test_normalization():
    x=normalize(np.array([[0,1,2],[3,4,5]]))
    assert x.dtype==np.uint8 and x.min()==0 and x.max()==255
