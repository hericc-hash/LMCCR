from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict,List,Sequence,Tuple
import torch
from torch.utils.data import Dataset
from .data import FeatureBank

@dataclass
class PairRow:
    recipient:int; donor:int; level:int; task:int; coordinate_distance:float; quality_distance:float

def _dist(a,b): return float(torch.linalg.vector_norm(a-b).item())

def build_cross_bank_pair_rows(recipient_bank:FeatureBank,donor_bank:FeatureBank,recipient_splits:Sequence[str],donor_splits:Sequence[str],
                               quality_weight=0.25,topk=8,deterministic_nearest=False,max_pairs_per_slot=1,seed=20260823):
    rng=random.Random(seed); rec=recipient_bank.indices(recipient_splits); don=donor_bank.indices(donor_splits); rows=[]; index:Dict[Tuple[int,int,int],List[int]]={}
    for d in don:
        for l in range(5):
            for t in range(3): index.setdefault((l,t,int(donor_bank.y[d,l,t]>=0.5)),[]).append(d)
    for r in rec:
        for l in range(5):
            for t in range(3):
                yr=int(recipient_bank.y[r,l,t]>=0.5); cand=[d for d in index.get((l,t,1-yr),[]) if donor_bank.case_ids[d]!=recipient_bank.case_ids[r]]
                if not cand: continue
                scored=[]; cr=recipient_bank.C[r,l]; qr=recipient_bank.quality[r,l,t]
                for d in cand:
                    cd=_dist(cr,donor_bank.C[d,l]); qd=abs(float(qr)-float(donor_bank.quality[d,l,t])); scored.append((cd+quality_weight*qd,cd,qd,d))
                scored.sort(key=lambda z:z[0]); pool=scored[:max(1,min(topk,len(scored)))]
                chosen=pool[:max_pairs_per_slot] if deterministic_nearest else rng.sample(pool,k=min(max_pairs_per_slot,len(pool)))
                rows += [PairRow(r,d,l,t,cd,qd) for _,cd,qd,d in chosen]
    return rows

def build_pair_rows(bank,*args,**kwargs): return build_cross_bank_pair_rows(bank,bank,*args,**kwargs)

class CounterfactualPairDataset(Dataset):
    def __init__(self,bank:FeatureBank,rows:List[PairRow],alphas=(0.25,0.5,0.75,1.0),seed=20260823):
        self.bank=bank; self.rows=rows; self.alphas=tuple(float(a) for a in alphas); self.seed=seed
    def __len__(self): return len(self.rows)
    def __getitem__(self,j):
        p=self.rows[j]; rng=random.Random(self.seed+j*104729); alpha=rng.choice(self.alphas)
        a=self.bank.get_level(p.recipient,p.level); b=self.bank.get_level(p.donor,p.level)
        out={"xa":a["x"],"ea":a["e"],"ca":a["c"],"ya":a["y"],"qa":a["quality"],
             "xb":b["x"],"eb":b["e"],"cb":b["c"],"yb":b["y"],"qb":b["quality"],
             "recipient_index":p.recipient,"donor_index":p.donor,"level_index":p.level,"task_index":p.task,
             "alpha":torch.tensor(alpha,dtype=torch.float32),"coordinate_distance":torch.tensor(p.coordinate_distance),"quality_distance":torch.tensor(p.quality_distance),
             "recipient_id":self.bank.case_ids[p.recipient],"donor_id":self.bank.case_ids[p.donor]}
        if "legacy_a" in a: out["legacy_aa"]=a["legacy_a"]; out["legacy_ab"]=b["legacy_a"]
        if "legacy_r" in a: out["legacy_ra"]=a["legacy_r"]; out["legacy_rb"]=b["legacy_r"]
        return out

