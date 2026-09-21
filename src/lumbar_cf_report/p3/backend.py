
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from .common import sha256_file, flatten_candidates

ALIASES={
 "serials":["source_serial","serials","serial","ids"],
 "split":["split","splits","cohort"],
 "X":["X","segment_base","x"],
 "E":["E","task_features","e"],
 "C":["C","coordinate_state","c"],
 "y":["y","labels","label"],
 "quality":["quality","task_quality","q"],
 "valid":["valid","task_valid","label_valid"]
}

class FeatureBank:
    def __init__(self,d:dict):
        self.raw=d
        for name,keys in ALIASES.items():
            val=None; used=None
            for k in keys:
                if k in d: val=d[k]; used=k; break
            if val is None:
                if name=="quality":
                    val=torch.ones_like(torch.as_tensor(self._get("y")).float())
                elif name=="valid":
                    val=torch.ones_like(torch.as_tensor(self._get("y")).float())
                else: raise KeyError(f"Feature bank missing {name}; aliases={keys}; keys={list(d.keys())}")
            setattr(self,name,val)
        self.serials=[int(x) for x in list(self.serials)]
        self.split=[str(x) for x in list(self.split)]
        for k in ["X","E","C","y","quality","valid"]:
            setattr(self,k,torch.as_tensor(getattr(self,k)).float())
    def _get(self,name):
        for k in ALIASES[name]:
            if k in self.raw: return self.raw[k]
        raise KeyError(name)
    def indices(self,split_name:str)->List[int]:
        return [i for i,s in enumerate(self.split) if s==split_name]

def load_bank(path:str|Path)->FeatureBank:
    return FeatureBank(torch.load(path,map_location="cpu",weights_only=False))

def dynamic_model_class(model_py:str|Path):
    p=Path(model_py)
    spec=importlib.util.spec_from_file_location("stage120_frozen_snapshot",p)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {p}")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if not hasattr(mod,"Stage120JointModel"): raise RuntimeError("Stage120JointModel missing")
    return mod.Stage120JointModel

def load_model(cfg:dict,device:torch.device,checkpoint_override:str|None=None):
    cls=dynamic_model_class(cfg["paths"]["frozen_model_py"])
    model=cls(**cfg["model"]).to(device)
    cp=checkpoint_override or cfg["paths"]["frozen_checkpoint"]
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    sd=payload.get("model_state",payload.get("model_state_dict",payload.get("state_dict",payload)))
    miss,unexp=model.load_state_dict(sd,strict=False)
    if miss or unexp:
        raise RuntimeError(f"Non-strict mismatch loading frozen model: missing={miss[:20]} unexpected={unexp[:20]}")
    return model,payload

def extract_endpoint(cf:dict|torch.Tensor)->torch.Tensor:
    if torch.is_tensor(cf): x=cf
    else:
        pref=["reconstructed_task_features","task_features","ehat","evidence"]
        x=None
        for k in pref:
            if k in cf and torch.is_tensor(cf[k]): x=cf[k]; break
        if x is None:
            cand=[(n,t) for n,t in flatten_candidates(cf) if t.ndim>=3 and t.shape[-2:]==(3,128)]
            if len(cand)!=1: raise RuntimeError(f"Cannot uniquely resolve CF endpoint: {[(n,tuple(t.shape)) for n,t in cand]}")
            x=cand[0][1]
    if x.ndim==4 and x.shape[1]==1: x=x[:,0]
    if x.ndim!=3 or tuple(x.shape[-2:])!=(3,128):
        raise RuntimeError(f"Unexpected endpoint shape {tuple(x.shape)}")
    return x

def extract_encoded_factual(fa:dict)->torch.Tensor:
    for k in ["reconstructed_task_features","task_features","ehat","evidence"]:
        if k in fa and torch.is_tensor(fa[k]):
            x=fa[k]
            if x.ndim==4 and x.shape[1]==1: x=x[:,0]
            if x.ndim==3 and tuple(x.shape[-2:])==(3,128): return x
    cand=[(n,t) for n,t in flatten_candidates(fa) if t.ndim==3 and tuple(t.shape[-2:])==(3,128)]
    if len(cand)!=1: raise RuntimeError(f"Cannot resolve encoded factual endpoint {[(n,tuple(t.shape)) for n,t in cand]}")
    return cand[0][1]

def state_logits(model,evidence:torch.Tensor,c:torch.Tensor)->torch.Tensor:
    # evidence: [B,L,3,128] or [B,3,128]; c: [B,L,47] or [B,47]
    if evidence.ndim==3: evidence=evidence[:,None]
    if c.ndim==2: c=c[:,None]
    raw=model._state_from_evidence(evidence,c)
    if torch.is_tensor(raw):
        cand=[("tensor",raw)]
    else:
        cand=flatten_candidates(raw)
    # Prefer explicit logits with exact [B,L,3].
    target=(evidence.shape[0],evidence.shape[1],3)
    exact=[(n,t) for n,t in cand if tuple(t.shape)==target]
    logit=[(n,t) for n,t in exact if "logit" in n.lower()]
    chosen=logit[0][1] if len(logit)==1 else (exact[0][1] if len(exact)==1 else None)
    if chosen is None:
        # Handle list of 3 [B,L] task tensors.
        parts=[(n,t) for n,t in cand if tuple(t.shape)==target[:2]]
        if len(parts)==3:
            chosen=torch.stack([t for _,t in parts],-1)
    if chosen is None:
        raise RuntimeError(f"Could not resolve state logits. candidates={[(n,tuple(t.shape)) for n,t in cand]}")
    return chosen
