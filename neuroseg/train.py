import argparse
import json
import random
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .model import UNet,loss_fn,scores
from .data import prepare,MRIDataset

def run(args):
    if args.size<32 or args.size%16 or args.epochs<1 or args.batch<1 or args.width<1 or args.patience<1 or args.workers<0: raise ValueError("Invalid training configuration")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda" and not args.allow_cpu: raise RuntimeError("Enable a Kaggle GPU first")
    print(f"Device: {device}; GPU count: {torch.cuda.device_count()}",flush=True)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    root=Path(args.data)
    df=prepare(root,Path(args.input))
    manifest=root/"dataset_manifest.csv"
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    df.to_csv(out/"dataset_manifest.csv",index=False)
    print(df.groupby("split").agg(images=("filename","count"),patients=("patient_group","nunique")),flush=True)
    loaders={}
    for split in ("train","val","test"):
        loaders[split]=DataLoader(MRIDataset(root,df,split,args.size),batch_size=args.batch,
                                   shuffle=split=="train",num_workers=args.workers,
                                   pin_memory=device.type=="cuda")
    model=UNet(args.width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=3,factor=.5)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    cfg={"width":args.width,"size":args.size,"threshold":.5,"normalization":"foreground_percentile_1_99","manifest_sha256":digest}
    best=-1.; stale=0; history=[]; start=1
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=True)
        if ck["config"]!=cfg: raise ValueError("Resume configuration differs")
        model.load_state_dict(ck["model"],strict=True); opt.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        history=ck["history"]; best=ck["best"]; stale=ck["stale"]; start=ck["epoch"]+1
        if not (out/"best_unet.pth").exists(): raise RuntimeError("Resume requires existing best_unet.pth in output directory")
    def epoch(split):
        training=split=="train"; model.train(training); total=0; loss_sum=0.; dice_sum=0.
        for x,y,*_ in loaders[split]:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            if training: opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training),torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                logits=model(x); loss=loss_fn(logits,y)
            if training:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt); scaler.update()
            n=x.size(0); total+=n; loss_sum+=loss.item()*n
            dice_sum+=scores(logits.detach(),y)["dice"].sum().item()
        return loss_sum/total,dice_sum/total
    for e in range(start,args.epochs+1):
        tl,td=epoch("train"); vl,vd=epoch("val"); scheduler.step(vd)
        row={"epoch":e,"train_loss":tl,"train_dice":td,"val_loss":vl,"val_dice":vd,"lr":opt.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row),flush=True)
        improved=vd>best
        if improved: best=vd; stale=0
        else: stale+=1
        if improved:
            torch.save({"model":model.state_dict(),"config":cfg,"epoch":e,"val_dice":vd},out/"best_unet.pth")
        torch.save({"model":model.state_dict(),"config":cfg,"epoch":e,"best":best,"stale":stale,
                    "optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"history":history},out/"last_unet.pth")
        pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
        if stale>=args.patience: break
    hist=pd.DataFrame(history)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for split in ("train","val"):
        ax[0].plot(hist.epoch,hist[split+"_loss"],label=split)
        ax[1].plot(hist.epoch,hist[split+"_dice"],label=split)
    for a,title in zip(ax,["Loss","Dice"]): a.set_title(title); a.set_xlabel("Epoch"); a.legend()
    fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=150); plt.close(fig)
    ck=torch.load(out/"best_unet.pth",map_location=device,weights_only=True)
    model.load_state_dict(ck["model"],strict=True); model.eval()
    print("Best validation Dice:",ck["val_dice"],flush=True)
    if args.skip_test:
        print("Test set reserved for final model selection."); return
    rows=[]; examples=[]
    with torch.inference_mode():
        for x,y,names,groups,types in loaders["test"]:
            logits=model(x.to(device)); y_device=y.to(device)
            m=scores(logits,y_device)
            pred=(logits.sigmoid()>.5).cpu().numpy()
            for i,name in enumerate(names):
                row={"filename":name,"patient_group":groups[i],"tumor_type":types[i]}
                row.update({key:float(v[i].item()) for key,v in m.items()}); rows.append(row)
                examples.append((row["dice"],x[i,0].numpy(),y[i,0].numpy(),pred[i,0],name))
    result=pd.DataFrame(rows); result.to_csv(out/"per_sample_metrics.csv",index=False)
    cols=["dice","iou","precision","recall","specificity"]
    summary={"best_epoch":ck["epoch"],"best_validation_dice":ck["val_dice"],"test_images":len(result),
             "test_patients":result.patient_group.nunique(),"slice_macro":result[cols].mean().to_dict(),
             "patient_macro":result.groupby("patient_group")[cols].mean().mean().to_dict(),
             "by_tumor_type":result.groupby("tumor_type")[cols].mean().to_dict(orient="index"),
             "limitations":"Tumor-positive 2D slices only; no clinical validation."}
    (out/"test_metrics.json").write_text(json.dumps(summary,indent=2))
    examples.sort(key=lambda r:r[0])
    selected=examples[:3]+examples[-3:]
    fig,ax=plt.subplots(len(selected),4,figsize=(12,3*len(selected)),squeeze=False)
    for a,(dice,img,gt,pred,name) in zip(ax,selected):
        a[0].imshow(img,cmap="gray"); a[1].imshow(gt,cmap="gray"); a[2].imshow(pred,cmap="gray")
        a[3].imshow(img,cmap="gray"); a[3].imshow(np.ma.masked_where(~pred,pred),cmap="autumn",alpha=.45)
        for axis,title in zip(a,[name,"Ground truth",f"Prediction Dice={dice:.3f}","Overlay"]): axis.set_title(title); axis.axis("off")
    fig.tight_layout(); fig.savefig(out/"segmentation_examples.png",dpi=120); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="/tmp/neuroseg_processed")
    p.add_argument("--input",default="/kaggle/input")
    p.add_argument("--out",default="/kaggle/working/neuroseg_results")
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--size",type=int,default=256)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--patience",type=int,default=6)
    p.add_argument("--resume")
    p.add_argument("--skip-test",action="store_true")
    p.add_argument("--allow-cpu",action="store_true")
    run(p.parse_args())

if __name__=="__main__": main()
