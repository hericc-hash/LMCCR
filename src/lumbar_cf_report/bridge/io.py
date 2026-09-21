from __future__ import annotations
from pathlib import Path
import hashlib, json, os, torch

def sha256(path):
    p=Path(path); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def load_pt(path):
    return torch.load(path,map_location='cpu',weights_only=False)

def dump_json(obj,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def read_jsonl(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    return rows

def write_jsonl(rows,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

def env_path(name):
    x=os.environ.get(name,'').strip();return Path(x).resolve() if x else None
