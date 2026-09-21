from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict,List,Tuple
import torch
from ..clinical.schema import LEVELS,TASKS,MAIN_SLOT_NAMES,disease_slot_index
from .common import normalized_serial

@dataclass(frozen=True)
class DonorCandidate:
    index:int; coordinate_distance:float; quality_distance:float; score:float

def coordinate_distance(a,b):
    a=torch.as_tensor(a).float(); b=torch.as_tensor(b).float()
    if b.ndim==1:b=b.unsqueeze(0)
    return torch.sqrt(((b-a.unsqueeze(0))**2).mean(-1).clamp_min(0))

def build_candidate_cache(recipient_dataset,donor_dataset,topk=5,quality_weight=.25):
    """Build opposite-label, coordinate-matched donor candidates."""
    cache={}
    for ri in range(len(recipient_dataset)):
        for li in range(5):
            for ti in range(3):
                if float(recipient_dataset.label_valid[ri,li,ti])<=.5: continue
                target=float(recipient_dataset.labels[ri,li,ti]); mask=(donor_dataset.label_valid[:,li,ti]>.5)&(donor_dataset.labels[:,li,ti]!=target); idx=torch.where(mask)[0]
                if idx.numel()==0: continue
                cd=coordinate_distance(recipient_dataset.coordinate_state[ri,li],donor_dataset.coordinate_state[idx,li]); qd=(donor_dataset.task_quality[idx,li,ti]-recipient_dataset.task_quality[ri,li,ti]).abs(); score=cd+float(quality_weight)*qd
                order=torch.argsort(score)[:min(int(topk),int(idx.numel()))]
                cache[(ri,li,ti)]=[DonorCandidate(int(idx[o]),float(cd[o]),float(qd[o]),float(score[o])) for o in order.tolist()]
    return cache

def build_report_selection_manifest(va,tr,handoff,eligible_names,seed=20260815,slots_per_case=3,topk=5,quality_weight=.25):
    allowed={(li,ti) for li,l in enumerate(LEVELS) for ti,t in enumerate(TASKS) if f'{t}:{l}' in set(eligible_names)}
    if len(allowed)!=12: raise RuntimeError(f'frozen eligible slot count must be 12, got {len(allowed)}')
    cache=build_candidate_cache(va,tr,topk,quality_weight); rng=random.Random(int(seed)); rows=[]; available_counts=[]; donor_match=0
    for ri in range(len(va)):
        rid=normalized_serial(va.serials[ri]); keys=[k for k in cache if k[0]==ri and (k[1],k[2]) in allowed]; available_counts.append(len(keys))
        selected=[]
        for ti_sel in range(3):
            kk=[k for k in keys if k[2]==ti_sel]; rng.shuffle(kk)
            if kk:selected.append(kk[0])
        if len(selected)<max(1,int(slots_per_case)):
            remain=[k for k in keys if k not in selected]; rng.shuffle(remain); selected.extend(remain[:max(0,int(slots_per_case)-len(selected))])
        selected=selected[:max(1,int(slots_per_case))]
        for _,li,ti in selected:
            hr=handoff.record(rid,li,ti); candidate=cache[(ri,li,ti)][0]; candidate_donor=normalized_serial(tr.serials[candidate.index]); transported_donor=int(hr['donor_id']); donor_match+=int(candidate_donor==transported_donor)
            slot=disease_slot_index(li,ti)
            rows.append({
                'recipient_serial':rid,'level_index':int(li),'task_index':int(ti),'slot_index':int(slot),'slot_name':MAIN_SLOT_NAMES[slot],
                'recipient_label':float(hr['recipient_target_label']),'transport_donor_serial':transported_donor,'transport_donor_label':float(hr['donor_target_label']),
                'transport_coordinate_distance':float(hr.get('coordinate_distance',float('nan'))),
                'matched_top1_donor_serial':candidate_donor,'matched_coordinate_distance':float(candidate.coordinate_distance),
            })
    if len(va)!=100 or len(rows)!=300: raise RuntimeError(f'frozen report selection must produce 300 rows from Internal100; cases={len(va)} rows={len(rows)}')
    if any(x!=12 for x in available_counts): raise RuntimeError(f'every Internal recipient must expose all 12 frozen eligible slots; distribution={sorted(set(available_counts))}')
    per_task={i:sum(int(r['task_index']==i) for r in rows) for i in range(3)}
    per_rec={r:sum(int(x['recipient_serial']==r) for x in rows) for r in {int(x['recipient_serial']) for x in rows}}
    if per_task!={0:100,1:100,2:100} or any(v!=3 for v in per_rec.values()): raise RuntimeError(f'balanced_task contract failed per_task={per_task}')
    return rows, {'selected_pairs':300,'available_slots_per_recipient':12,'per_task':per_task,'top1_donor_match_count':donor_match,'top1_donor_match_fraction':donor_match/300.0}

