"""Download small attributed research image/mask pairs, not original MAT/PID records."""
import hashlib
import io
import json
import urllib.request
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from neuroseg.data import HASHES,FOLDERS,MIRROR_REF,SECOND_MIRROR,normalize

def prepare(target):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    samples=[]
    for n in (1,1500,3005):
        name=f"{n}.mat"
        folder=FOLDERS[(n-1)//766]
        urls=[f"https://raw.githubusercontent.com/jasonnan2/Deep-Learning-MRI-tumor/{MIRROR_REF}/{folder}/{name}",f"{SECOND_MIRROR}/{name}"]
        for url in urls:
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"NeuroSeg-demo/3"}),timeout=45) as r:
                    raw=r.read(16*1024*1024+1)
                if hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()!=HASHES[name]: raise ValueError("Hash mismatch")
                with h5py.File(io.BytesIO(raw),"r") as f:
                    d=f["cjdata"]; image=np.asarray(d["image"]).T; mask=np.asarray(d["tumorMask"]).T
                if image.shape!=mask.shape or not np.isin(mask,[0,1,255]).all(): raise ValueError("Bad pair")
                sample_id=f"sample_{n}"
                Image.fromarray(normalize(image)).save(target/f"{sample_id}_image.png")
                Image.fromarray((mask>0).astype(np.uint8)*255).save(target/f"{sample_id}_mask.png")
                samples.append({"id":sample_id,"title":f"Research MRI {n} (with annotation)","source":url,"split":"not verified"})
                break
            except (OSError,ValueError,KeyError) as exc:
                print(f"Sample {n}: {exc}",flush=True)
    attribution={"creator":"Jun Cheng","dataset":"brain tumor dataset","doi":"https://doi.org/10.6084/m9.figshare.1512427","license":"https://creativecommons.org/licenses/by/4.0/","changes":"MATLAB arrays transposed, foreground-normalized to 8-bit PNG, mask binarized; no PID fields exported. Hashes verified against pinned mirrors; not independently checked against original archive.","samples":samples}
    (target/"manifest.json").write_text(json.dumps(attribution,indent=2))
    print(f"Prepared {len(samples)} matched research samples",flush=True)

if __name__=="__main__":
    prepare(Path(__file__).resolve().parents[1]/"neuroseg"/"samples")
