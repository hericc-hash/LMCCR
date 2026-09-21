from __future__ import annotations
import glob,os
from pathlib import Path
import numpy as np
from .io import read_jsonl
from .pool import normalize_tag


def discover_legacy_pools(cfg,out_dir=None):
    env=os.environ.get("R32S4_LEGACY_POOL","").strip();paths=[]
    if env:paths.extend(Path(x.strip()) for x in env.split(":") if x.strip())
    for pat in cfg.get("candidate_pool_globs",[]):
        paths.extend(Path(p) for p in glob.glob(pat))
    uniq=[];seen=set()
    for p in paths:
        try:r=p.resolve()
        except Exception:r=p
        if out_dir is not None:
            try:
                if Path(out_dir).resolve() in r.parents:continue
            except Exception:pass
        if r.exists() and str(r) not in seen:
            seen.add(str(r));uniq.append(r)
    # Prefer richer/newer files when multiple caches contain the same serial.
    uniq.sort(key=lambda p:(p.stat().st_mtime if p.exists() else 0),reverse=True)
    return uniq


def load_legacy_index(paths):
    idx={};source={}
    for p in paths:
        try:rows=read_jsonl(p)
        except Exception:continue
        for r in rows:
            s=str(r.get("serial",""))
            if not s or s in idx:continue
            idx[s]=r;source[s]=str(p)
    return idx,source


def load_legacy_rows_by_serial(paths):
    out={}
    for p in paths:
        try:rows=read_jsonl(p)
        except Exception:continue
        for r in rows:
            s=str(r.get("serial",""))
            if not s:continue
            out.setdefault(s,[]).append((r,str(p)))
    return out


def compatible(row,legacy,cfg):
    if not legacy:return False,"missing"
    if cfg.get("require_core_exact",True) and list(map(int,row["core_binary"]))!=list(map(int,legacy.get("core_binary",[]))):return False,"core_mismatch"
    if list(map(int,row["slot_valid"]))!=list(map(int,legacy.get("slot_valid",[]))):return False,"valid_mismatch"
    a=np.asarray(row.get("planner_probs",[]),float);b=np.asarray(legacy.get("planner_probs",[]),float)
    if a.shape!=b.shape or (a.size and float(np.max(np.abs(a-b)))>float(cfg.get("planner_prob_tolerance",1e-6))):return False,"planner_prob_mismatch"
    return True,"ok"


def candidate_map(legacy,allowed_tags=None):
    allowed=set(allowed_tags or []) if allowed_tags else None;out={}
    for c in legacy.get("candidates",[]):
        tag=normalize_tag(c.get("tag",""));text=str(c.get("text","")).strip()
        if not tag or not text or (allowed is not None and tag not in allowed):continue
        if tag not in out:out[tag]=dict(c,tag=tag,text=text)
    return out


def merged_candidate_map(row,legacy_rows,cfg):
    allowed=cfg.get("allowed_tags");merged={};sources={};compatible_rows=0;incompatible=[]
    for legacy,src in legacy_rows:
        ok,why=compatible(row,legacy,cfg)
        if not ok:
            incompatible.append({"source":src,"reason":why});continue
        compatible_rows+=1
        cmap=candidate_map(legacy,allowed)
        for tag,c in cmap.items():
            if tag not in merged:
                merged[tag]=c;sources[tag]=src
    return merged,sources,{"compatible_rows":compatible_rows,"incompatible":incompatible}
