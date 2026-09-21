from __future__ import annotations
import json,hashlib,torch
from pathlib import Path

def read_jsonl(p):
    out=[]
    with open(p,encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def write_jsonl(rows,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

def append_jsonl(row,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()

def load_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump_json(x,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def load_pt(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
