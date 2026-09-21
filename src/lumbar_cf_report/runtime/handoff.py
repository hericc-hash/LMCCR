from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Any
import torch
from .common import torch_load, normalized_serial

class CounterfactualHandoff:
    """Validated APREB/CCMRT evidence handoff for any declared cohort."""

    def __init__(self, path: str | Path, expected_split: str | None = None, expected_cases: int | None = None, alpha: float = 1.0):
        self.path=Path(path); obj=torch_load(self.path,'cpu')
        if expected_split is not None and obj.get('split') != expected_split:
            raise RuntimeError(f"expected split={expected_split}, got {obj.get('split')}")
        if abs(float(obj.get('alpha',float('nan')))-float(alpha))>1e-8: raise RuntimeError(f"handoff alpha must be {alpha}, got {obj.get('alpha')}")
        recs=obj.get('records',[])
        if not recs: raise RuntimeError('handoff contains no records')
        self.records: Dict[Tuple[int,int,int],Dict[str,Any]]={}; self.factual={}; counts={}
        for r0 in recs:
            r=dict(r0); rid=normalized_serial(r['recipient_id']); did=normalized_serial(r['donor_id']); li=int(r['level_index']); ti=int(r['task_index'])
            r['recipient_id']=rid; r['donor_id']=did; key=(rid,li,ti)
            if key in self.records: raise RuntimeError(f'duplicate handoff key={key}')
            fe=torch.as_tensor(r['factual_E5x3']).float().cpu(); ce=torch.as_tensor(r['counterfactual_E5x3']).float().cpu()
            if fe.shape!=(5,3,128) or ce.shape!=(5,3,128): raise RuntimeError(f'E shape mismatch at {key}: {tuple(fe.shape)} {tuple(ce.shape)}')
            if rid in self.factual and not torch.allclose(self.factual[rid],fe,atol=1e-6,rtol=1e-6): raise RuntimeError(f'inconsistent factual E for serial={rid}')
            self.factual.setdefault(rid,fe); self.records[key]=r; counts[rid]=counts.get(rid,0)+1
        if expected_cases is not None and len(counts)!=int(expected_cases): raise RuntimeError(f'expected {expected_cases} recipients, got {len(counts)}')
        if any(v!=15 for v in counts.values()): raise RuntimeError('every recipient must contain 5 levels x 3 task records')
        self.recipient_serials=sorted(counts)
    def record(self,rid,li,ti): return self.records[(int(rid),int(li),int(ti))]
    def factual_e(self,rid): return self.factual[int(rid)]
    def counterfactual_e(self,rid,li,ti): return torch.as_tensor(self.record(rid,li,ti)['counterfactual_E5x3']).float().cpu()

