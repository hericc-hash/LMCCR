from __future__ import annotations
import torch
from .io import safe_load,reindex,serials

def build_split(mediation_path,bank_path):
    m=safe_load(mediation_path);b=safe_load(bank_path)
    sk=next(k for k in ('serials','source_serial','case_ids') if k in b)
    E=b.get('E',b.get('task_features'));X=b.get('X',b.get('segment_base'));C=b.get('C',b.get('coordinate_state'))
    s=m['serials']
    rawE=reindex(E,b[sk],s).float()
    ret={'serials':serials(s),'raw_E':rawE,'global_source':torch.as_tensor(m['global_source']).float(),'coordinate_state':torch.as_tensor(m['coordinate_state']).float(),'task_quality':torch.as_tensor(m['task_quality']).float(),'task_valid':torch.as_tensor(m['task_valid']).float(),'labels':torch.as_tensor(m['labels']).float(),'label_valid':torch.as_tensor(m['label_valid']).float()}
    for k in ('slot_labels','slot_valid','structure_flags'):
        if k in m:ret[k]=torch.as_tensor(m[k]).float()
    return ret

def disease_valid(d):return d['label_valid']*d['task_valid']
