
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import torch

LEVELS=["L1/2","L2/3","L3/4","L4/5","L5/S1"]
TASKS=["disc","stenosis","nerve"]

def load_cfg(path: str|Path)->dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def dump_json(obj: Any, path: str|Path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=_json_default),encoding="utf-8")

def _json_default(x):
    if isinstance(x,(np.floating,np.integer)): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)

def write_jsonl(rows: Iterable[dict], path: str|Path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,default=_json_default)+"\n")

def read_jsonl(path: str|Path)->List[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def sha256_file(path: str|Path)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize(v: torch.Tensor, eps:float=1e-8)->torch.Tensor:
    return v/(v.norm(dim=-1,keepdim=True)+eps)

def cosine(a:torch.Tensor,b:torch.Tensor,eps:float=1e-8)->torch.Tensor:
    return (normalize(a,eps)*normalize(b,eps)).sum(-1)

def safe_mean(xs):
    xs=[float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else float("nan")

def tensor_state_sha(named_tensors: Iterable[Tuple[str,torch.Tensor]])->str:
    h=hashlib.sha256()
    for n,t in sorted(named_tensors,key=lambda x:x[0]):
        h.update(n.encode()); x=t.detach().cpu().contiguous()
        h.update(str(tuple(x.shape)).encode()); h.update(str(x.dtype).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()

def progress(new:float, old:float, direct:float, eps:float=1e-6)->float:
    den=float(direct)-float(old)
    if abs(den)<eps:
        return 1.0 if new>=old-eps else 0.0
    return float((new-old)/den)

def task_weight_from_gaps(gaps: Dict[str,float], eps:float=0.02)->Dict[str,float]:
    raw={t:max(0.0,float(gaps.get(t,0.0)))+eps for t in TASKS}
    s=sum(raw.values())
    return {t:raw[t]/s for t in TASKS}

def flatten_candidates(obj,prefix=""):
    out=[]
    if torch.is_tensor(obj):
        out.append((prefix,obj))
    elif isinstance(obj,dict):
        for k,v in obj.items(): out.extend(flatten_candidates(v,f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj): out.extend(flatten_candidates(v,f"{prefix}[{i}]"))
    return out
