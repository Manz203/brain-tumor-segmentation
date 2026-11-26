"""One command: compare validation-only candidates, then test the winner once."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import torch
from .data import prepare
from .evaluate import evaluate_checkpoint
from .model import load_checkpoint

def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Enable GPU in Kaggle.")
    data = Path(args.data).resolve()
    prepare(data, Path(args.input))
    tracking = None
    if args.mlflow:
        from .tracking import configure
        tracking = configure(out, args.tracking_uri)
    if (out/"selection.json").exists():
        checkpoint=out/"best_model.pth"
        if not checkpoint.is_file(): raise RuntimeError("Frozen selection checkpoint missing")
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (out/"test_metrics.json").is_file():
            summary=json.loads((out/"test_metrics.json").read_text())
            if summary.get("checkpoint_sha256")!=digest: raise RuntimeError("Evaluation checkpoint mismatch")
        else:
            summary=evaluate_checkpoint(checkpoint,data,out,batch=args.batch,workers=args.workers)
        if tracking is not None and not (out/"registry.json").exists():
            from .tracking import register_champion
            register_champion(tracking,checkpoint,out,summary)
        print("Using frozen selection; no new candidates or test-driven tuning.",flush=True)
        return summary
    candidates = [(arch,args.lr) for arch in ("unet","attention_unet")]
    if args.tune:
        candidates += [(arch,args.lr/3) for arch in ("unet","attention_unet")]
    rows = []
    for arch, lr in candidates:
        directory = out/"candidates"/f"{arch}_lr{lr:.7g}"
        directory.mkdir(parents=True, exist_ok=True)
        params = {"architecture":arch,"lr":lr,"epochs":args.epochs,"width":args.width,
                  "size":args.size,"batch":args.batch,"seed":42}
        config_path = directory/"experiment.json"
        if config_path.exists() and json.loads(config_path.read_text()) != params:
            raise ValueError("Candidate configuration changed; choose a new output directory")
        config_path.write_text(json.dumps(params,indent=2))
        cmd = [sys.executable,"-u","-m","neuroseg.train","--architecture",arch,
               "--epochs",str(args.epochs),"--batch",str(args.batch),"--lr",str(lr),
               "--width",str(args.width),"--size",str(args.size),"--workers",str(args.workers),
               "--data",str(data),"--out",str(directory),"--skip-test"]
        if args.allow_cpu: cmd += ["--allow-cpu"]
        if args.multi_gpu: cmd += ["--multi-gpu"]
        if (directory/"best_unet.pth").is_file() and (directory/"last_unet.pth").is_file():
            cmd += ["--resume",str(directory/"last_unet.pth")]
        subprocess.run(cmd,check=True)
        ck = torch.load(directory/"best_unet.pth",map_location="cpu",weights_only=True)
        row = {**params,"best_epoch":ck["epoch"],"validation_dice":ck["val_dice"],
               "manifest_sha256":ck["config"]["manifest_sha256"],"candidate":str(directory.relative_to(out))}
        if tracking is not None:
            from .tracking import log_candidate
            row["mlflow_run_id"] = log_candidate(tracking,directory,params)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["validation_dice","architecture","lr"],ascending=[False,True,True])
    if table.manifest_sha256.nunique()!=1:
        raise RuntimeError("Candidates used different datasets")
    table.to_csv(out/"model_comparison.csv",index=False)
    winner = table.iloc[0].to_dict()
    source = out/winner["candidate"]
    checkpoint = out/"best_model.pth"
    shutil.copy2(source/"best_unet.pth",checkpoint)
    for name in ("training_curves.png","training_history.csv","dataset_manifest.csv"):
        shutil.copy2(source/name,out/name)
    # Save the selection before touching test labels. If interrupted, use evaluate explicitly.
    (out/"selection.json").write_text(json.dumps(winner,indent=2))
    summary = evaluate_checkpoint(checkpoint,data,out,batch=args.batch,workers=args.workers)
    if tracking is not None:
        from .tracking import register_champion
        register_champion(tracking,checkpoint,out,summary)
    print("Selected",winner["architecture"],"using validation Dice",winner["validation_dice"],flush=True)
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="/tmp/neuroseg_processed")
    p.add_argument("--input",default="/kaggle/input")
    p.add_argument("--out",default="/kaggle/working/neuroseg_study")
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--size",type=int,default=256)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--tune",action="store_true",help="Train two additional learning-rate candidates")
    p.add_argument("--multi-gpu",action="store_true")
    p.add_argument("--allow-cpu",action="store_true")
    p.add_argument("--mlflow",action="store_true")
    p.add_argument("--tracking-uri")
    run(p.parse_args())

if __name__=="__main__": main()
