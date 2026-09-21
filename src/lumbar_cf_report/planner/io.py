from __future__ import annotations
from pathlib import Path
import hashlib,json,torch,numpy as np

def safe_load(p):
    try:return torch.load(p,map_location='cpu',weights_only=False)
    except TypeError:return torch.load(p,map_location='cpu')

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def dumpj(x,p):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w') as f:json.dump(x,f,indent=2,ensure_ascii=False,default=str)

def serials(x):return [str(v.item() if hasattr(v,'item') else v) for v in x]

def reindex(t,src_serials,target_serials):
    m={s:i for i,s in enumerate(serials(src_serials))}
    idx=[m[str(s)] for s in serials(target_serials)]
    return torch.as_tensor(t)[idx]

def set_seed(seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
