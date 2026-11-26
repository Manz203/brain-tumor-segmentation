from pathlib import Path
import subprocess
import tempfile
import zipfile
import shutil
import hashlib
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

MIRROR="https://github.com/jasonnan2/Deep-Learning-MRI-tumor.git"
FOLDERS=["brainTumorDataPublic_1-766","brainTumorDataPublic_767-1532","brainTumorDataPublic_1533-2298","brainTumorDataPublic_2299-3064"]
EXPECTED={"train":(2124,163),"val":(633,46),"test":(307,24)}
LABELS={1:"Meningioma",2:"Glioma",3:"Pituitary tumor"}

def normalize(image):
    image=np.asarray(image,dtype=np.float32)
    if not np.isfinite(image).all(): raise ValueError("Invalid intensity")
    fg=image[image>0]
    if not fg.size: raise ValueError("Empty MRI")
    lo,hi=np.percentile(fg,[1,99])
    if hi<=lo: raise ValueError("Invalid MRI intensity range")
    result=np.clip((image-lo)/(hi-lo),0,1)
    result[image==0]=0
    return np.rint(result*255).astype(np.uint8)

def validate(root):
    df=pd.read_csv(root/"dataset_manifest.csv")
    assert len(df)==3064 and df.filename.is_unique
    assert df.patient_group.nunique()==233
    assert df.groupby("patient_group").split.nunique().max()==1
    for split,(n,p) in EXPECTED.items():
        subset=df[df.split==split]
        assert len(subset)==n and subset.patient_group.nunique()==p
        expected=set(subset.filename)
        for folder in ("images","masks"):
            assert {x.name for x in (root/split/folder).glob("*.png")}==expected
    return df

def source_files(input_dir,cache):
    files=sorted(input_dir.rglob("*.mat"),key=lambda p:p.name) if input_dir.exists() else []
    files=[p for p in files if p.stem.isdigit()]
    if len(files)==3064 and len({p.name for p in files})==3064: return files
    repo=cache/"source"
    if not (repo/".git").exists():
        subprocess.run(["git","clone","--depth","1","--filter=blob:none","--sparse",MIRROR,str(repo)],check=True)
    subprocess.run(["git","-C",str(repo),"sparse-checkout","set",*FOLDERS],check=True)
    files=sorted([p for folder in FOLDERS for p in (repo/folder).glob("*.mat")],key=lambda p:p.name)
    if len(files)!=3064 or len({p.name for p in files})!=3064:
        raise RuntimeError("Source mirror incomplete; attach the original 3064 MATLAB files as a private Kaggle input.")
    return files

def prepare(root,input_dir=Path("/kaggle/input")):
    root=Path(root); input_dir=Path(input_dir)
    if (root/"dataset_manifest.csv").exists(): return validate(root)
    archives=sorted(input_dir.rglob("neuroseg_processed_v2.zip")) if input_dir.exists() else []
    if archives:
        root.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archives[0]) as z:
            for member in z.infolist():
                if not (root/member.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Unsafe archive path")
            z.extractall(root)
        return validate(root)
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neuroseg_source_") as tmp:
        files=source_files(input_dir,Path(tmp))
        rows=[]
        for path in files:
            with h5py.File(path,"r") as f:
                d=f["cjdata"]
                pid="".join(chr(int(x)) for x in np.asarray(d["PID"]).ravel() if int(x)).strip()
                rows.append({"filename":path.stem+".png","patient_id":pid,"label":int(np.asarray(d["label"]).item()),
                             "height":d["image"].shape[0],"width":d["image"].shape[1]})
        df=pd.DataFrame(rows)
        assert df.patient_id.ne("").all() and df.patient_id.nunique()==233
        assert df.groupby("patient_id").label.nunique().max()==1
        assert set(df.label)==set(LABELS)
        patients=df[["patient_id","label"]].drop_duplicates()
        train,temp=train_test_split(patients,test_size=.30,random_state=42,stratify=patients.label)
        val,test=train_test_split(temp,test_size=1/3,random_state=42,stratify=temp.label)
        mapping={pid:split for split,part in [("train",train),("val",val),("test",test)] for pid in part.patient_id}
        df["split"]=df.patient_id.map(mapping)
        assert df.split.notna().all()
        for split,(n,p) in EXPECTED.items():
            subset=df[df.split==split]
            assert len(subset)==n and subset.patient_id.nunique()==p, "Source differs from original split"
            for folder in ("images","masks"): (root/split/folder).mkdir(parents=True,exist_ok=True)
        aliases={pid:f"patient_{i:03d}" for i,pid in enumerate(sorted(df.patient_id.unique()),1)}
        df["patient_group"]=df.patient_id.map(aliases)
        df["tumor_type"]=df.label.map(LABELS)
        for i,(path,row) in enumerate(zip(files,df.itertuples(index=False)),1):
            with h5py.File(path,"r") as f:
                d=f["cjdata"]; image=np.asarray(d["image"]).T; mask=np.asarray(d["tumorMask"]).T
            assert image.shape==mask.shape and np.isin(mask,[0,1,255]).all()
            Image.fromarray(normalize(image)).save(root/row.split/"images"/row.filename)
            Image.fromarray((mask>0).astype(np.uint8)*255).save(root/row.split/"masks"/row.filename)
            if i%500==0: print("Processed",i,flush=True)
        df.drop(columns="patient_id").to_csv(root/"dataset_manifest.csv",index=False)
    return validate(root)

class MRIDataset(Dataset):
    def __init__(self,root,df,split,size=256):
        self.root=Path(root)/split
        self.rows=df[df.split==split].reset_index(drop=True)
        self.size=size
        self.augment=split=="train"
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        with Image.open(self.root/"images"/r.filename) as im: image=im.convert("L")
        with Image.open(self.root/"masks"/r.filename) as im: mask=im.convert("L")
        if self.augment:
            if torch.rand(())<.5: image=TF.hflip(image); mask=TF.hflip(mask)
            angle=float(torch.empty(1).uniform_(-10,10).item())
            image=TF.rotate(image,angle,interpolation=InterpolationMode.BILINEAR)
            mask=TF.rotate(mask,angle,interpolation=InterpolationMode.NEAREST)
        image=image.resize((self.size,self.size),Image.Resampling.BILINEAR)
        mask=mask.resize((self.size,self.size),Image.Resampling.NEAREST)
        return TF.to_tensor(image),(TF.to_tensor(mask)>.5).float(),r.filename,r.patient_group,r.tumor_type
