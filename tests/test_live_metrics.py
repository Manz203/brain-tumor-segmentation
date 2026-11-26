import numpy as np
from neuroseg.predictor import compare_masks

def test_live_metrics_and_colors():
    pred=np.array([[1,1,0,0]],dtype=bool)
    gt=np.array([[1,0,1,0]],dtype=bool)
    m,img=compare_masks(pred,gt)
    assert m["dice"]==.5 and m["iou"]==1/3
    assert m["precision"]==m["recall"]==m["specificity"]==.5
    assert [m[k] for k in ("tp","fp","fn","tn")]==[1,1,1,1]
    assert img.tolist()==[[[80,220,130],[255,87,105],[89,169,255],[0,0,0]]]
    perfect,_=compare_masks(gt,gt)
    assert perfect["dice"]==1 and perfect["iou"]==1
    assert perfect["dice"]!=m["dice"]

def test_empty_ground_truth():
    z=np.zeros((2,2),dtype=bool)
    m,_=compare_masks(z,z)
    assert m["empty_both"] and m["dice"]==m["iou"]==1
    assert m["precision"] is None and m["recall"] is None
