from pathlib import Path
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
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
MIRROR_REF="67eaa59c52041abff0e8d78da8ccc53cd4c257de"
SECOND_MIRROR="https://raw.githubusercontent.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/c37454fbe8eae6c0ac32ab01d8030229ffe1d02e/Brain-Tumor-Dataset/Brain-Tumor-Images-Mat-Files"
HASHES=json.loads(Path(__file__).with_name("source_hashes.json").read_text())
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

def blob_hash(path):
    content=path.read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()

def verified(path):
    return path.is_file() and blob_hash(path)==HASHES.get(path.name)

def source_files(input_dir,cache):
    # Prefer a complete original dataset already attached as input.
    files=sorted((p for p in input_dir.rglob("*.mat") if p.stem.isdigit()),key=lambda p:p.name) if input_dir.exists() else []
    if len(files)==3064 and {p.name for p in files}==set(HASHES):
        if not all(verified(p) for p in files):
            raise ValueError("Attached MATLAB files differ from the pinned source hashes")
        print("Using verified attached original MATLAB files.",flush=True)
        return files
    cache.mkdir(parents=True,exist_ok=True)
    repo=cache/"source"
    raw=cache/"raw"
    raw.mkdir(exist_ok=True)
    # Reuse any successful downloads, even after an interrupted attempt.
    if all(verified(raw/name) for name in HASHES):
        return sorted(raw.glob("*.mat"),key=lambda p:p.name)
    # Sparse Git checkout is normally faster than thousands of HTTP requests.
    try:
        if not (repo/".git").exists():
            subprocess.run(["git","clone","--depth","1","--filter=blob:none","--sparse",MIRROR,str(repo)],check=True,timeout=1800)
        subprocess.run(["git","-C",str(repo),"sparse-checkout","set",*FOLDERS],check=True,timeout=1800)
        files=sorted([p for folder in FOLDERS for p in (repo/folder).glob("*.mat")],key=lambda p:p.name)
        if len(files)==3064 and {p.name for p in files}==set(HASHES) and all(verified(p) for p in files):
            print("Dataset verified against pinned mirror file hashes.",flush=True)
            return files
        print("Sparse checkout incomplete or hash mismatch; switching to verified raw downloads.",flush=True)
    except (subprocess.SubprocessError,OSError) as exc:
        print(f"Sparse checkout unavailable: {exc}. Trying raw downloads.",flush=True)
    def download(name):
        dest=raw/name
        if verified(dest): return dest
        n=int(Path(name).stem)
        folder=FOLDERS[(n-1)//766]
        urls=[
            f"https://raw.githubusercontent.com/jasonnan2/Deep-Learning-MRI-tumor/{MIRROR_REF}/{folder}/{name}",
            f"{SECOND_MIRROR}/{name}"
        ]
        last=None
        for attempt in range(3):
            for url in urls:
                try:
                    request=urllib.request.Request(url,headers={"User-Agent":"NeuroSeg-research-dataset/2"})
                    with urllib.request.urlopen(request,timeout=45) as response:
                        content=response.read(16*1024*1024+1)
                    digest=hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
                    if digest!=HASHES[name]: raise ValueError("File hash mismatch")
                    tmp=dest.with_suffix(".part")
                    tmp.write_bytes(content); tmp.replace(dest)
                    return dest
                except (OSError,ValueError) as exc: last=exc
            time.sleep(2**attempt)
        raise RuntimeError(f"Unable to obtain verified {name}: {last}")
    # Probe access first rather than issuing thousands of requests on a blocked network.
    download("1.mat")
    remaining=[name for name in HASHES if not verified(raw/name)]
    print(f"Downloading {len(remaining)} remaining files from pinned mirrors.",flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures=[executor.submit(download,name) for name in remaining]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%250==0 or i==len(futures): print(f"Downloaded {i}/{len(futures)}",flush=True)
    files=sorted(raw.glob("*.mat"),key=lambda p:p.name)
    if len(files)!=3064 or not all(verified(p) for p in files):
        raise RuntimeError("Dataset verification failed.")
    return files

def prepare(root,input_dir=Path("/kaggle/input")):
    root=Path(root); input_dir=Path(input_dir)
    if (root/"dataset_manifest.csv").exists(): return validate(root)
    # A previously saved notebook may expose the processed folder directly.
    if input_dir.exists():
        for candidate in sorted(input_dir.rglob("dataset_manifest.csv")):
            if all((candidate.parent/s/"images").is_dir() for s in EXPECTED):
                validate(candidate.parent)
                shutil.copytree(candidate.parent,root,dirs_exist_ok=True)
                return validate(root)
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
    # Keep verified source files in the session cache for retries.
    files=source_files(input_dir,root.parent/"neuroseg_source_cache")
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
