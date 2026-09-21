from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import torch
from torch.utils.data import Dataset
from .constants import LEVELS, TASKS
from .io import torch_load

# v2 deliberately does not require legacy A/R. Existing v1 banks remain compatible.
REQUIRED_KEYS = ["case_ids", "split", "X", "C", "y", "E"]

@dataclass
class BankInfo:
    n_cases:int; n_levels:int; n_tasks:int; x_dim:int; c_dim:int; e_dim:int
    legacy_a_dim:Optional[int]; legacy_r_dim:Optional[int]; splits:Dict[str,int]

class FeatureBank:
    """CCMRT source-evidence bank.

    Required:
      X [N,5,Dx] frozen the lumbar visual evidence branch segment base
      E [N,5,3,De] frozen task evidence
      C [N,5,Dc] Development-normalized patient-specific coordinates
      y [N,5,3] labels
    Optional:
      quality [N,5,3], A/R legacy v3 tensors (teacher diagnostics only), source_serial
    """
    def __init__(self,path:str):
        self.path=path; d=torch_load(path)
        if not isinstance(d,dict): raise TypeError(f"Feature bank must be dict, got {type(d)}")
        miss=[k for k in REQUIRED_KEYS if k not in d]
        if miss: raise KeyError(f"Feature bank missing required keys: {miss}")
        self.raw=d; self.case_ids=[str(x) for x in d["case_ids"]]; self.split=[str(x) for x in d["split"]]
        self.source_serial=d.get("source_serial",list(range(len(self.case_ids))))
        self.X=torch.as_tensor(d["X"],dtype=torch.float32); self.C=torch.as_tensor(d["C"],dtype=torch.float32)
        self.E=torch.as_tensor(d["E"],dtype=torch.float32); self.y=torch.as_tensor(d["y"],dtype=torch.float32)
        self.A=torch.as_tensor(d["A"],dtype=torch.float32) if d.get("A") is not None else None
        self.R=torch.as_tensor(d["R"],dtype=torch.float32) if d.get("R") is not None else None
        q=d.get("quality")
        if q is None: self.quality=torch.zeros((len(self.case_ids),5,3),dtype=torch.float32)
        else:
            q=torch.as_tensor(q,dtype=torch.float32)
            if q.ndim==2: q=q.unsqueeze(-1).expand(-1,-1,3)
            self.quality=q
        self.metadata=d.get("metadata",{}); self._validate()
    def _validate(self):
        n=len(self.case_ids)
        assert len(self.split)==n
        assert self.X.shape[:2]==(n,5),self.X.shape; assert self.C.shape[:2]==(n,5),self.C.shape
        assert self.E.shape[:3]==(n,5,3),self.E.shape; assert self.y.shape==(n,5,3),self.y.shape
        assert self.quality.shape==(n,5,3),self.quality.shape
        if self.A is not None: assert self.A.shape[:2]==(n,5),self.A.shape
        if self.R is not None: assert self.R.shape[:3]==(n,5,3),self.R.shape
        for name,t in [("X",self.X),("C",self.C),("E",self.E),("y",self.y),("quality",self.quality)]:
            if not torch.isfinite(t).all(): raise ValueError(f"Non-finite in {name}")
        if len(set(self.case_ids))!=n: raise ValueError("case_ids must be unique")
    @property
    def info(self):
        sc={}
        for s in self.split: sc[s]=sc.get(s,0)+1
        return BankInfo(len(self.case_ids),5,3,self.X.shape[-1],self.C.shape[-1],self.E.shape[-1],
                        None if self.A is None else self.A.shape[-1],None if self.R is None else self.R.shape[-1],sc)
    def indices(self,names:Sequence[str])->List[int]:
        names=set(names); return [i for i,s in enumerate(self.split) if s in names]
    def get_level(self,idx:int,level:int):
        out={"x":self.X[idx,level],"c":self.C[idx,level],"e":self.E[idx,level],"y":self.y[idx,level],"quality":self.quality[idx,level]}
        if self.A is not None: out["legacy_a"]=self.A[idx,level]
        if self.R is not None: out["legacy_r"]=self.R[idx,level]
        return out

class FactualLevelDataset(Dataset):
    def __init__(self,bank,split_names): self.bank=bank; self.rows=[(i,l) for i in bank.indices(split_names) for l in range(5)]
    def __len__(self): return len(self.rows)
    def __getitem__(self,j):
        i,l=self.rows[j]; d=self.bank.get_level(i,l); d.update({"case_index":i,"level_index":l,"case_id":self.bank.case_ids[i]}); return d


class FactualCaseDataset(Dataset):
    def __init__(self,bank:FeatureBank,split_names:Sequence[str]):
        self.bank=bank; self.rows=bank.indices(split_names)
    def __len__(self): return len(self.rows)
    def __getitem__(self,j):
        i=self.rows[j]
        out={"x":self.bank.X[i],"c":self.bank.C[i],"e":self.bank.E[i],"y":self.bank.y[i],"quality":self.bank.quality[i],
             "case_index":i,"case_id":self.bank.case_ids[i]}
        if self.bank.A is not None: out["legacy_a"]=self.bank.A[i]
        if self.bank.R is not None: out["legacy_r"]=self.bank.R[i]
        return out

