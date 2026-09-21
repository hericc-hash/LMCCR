from __future__ import annotations
import json
from .stage23_exact import replay_serials,binary_from_probs
from .pool import prepare_row

def _stream_subset(path,serials):
    want={str(x) for x in serials};out={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);s=str(r.get('serial'))
            if s in want:out[s]=r
    miss=want-set(out)
    if miss:raise RuntimeError(f'Subset rows missing from {path}: {sorted(miss)[:8]}')
    return out

def base_internal_subset(stage3a_inputs,v5_path,serials):
    want=[str(x) for x in serials];src=_stream_subset(stage3a_inputs,want);v5=_stream_subset(v5_path,want);out=[]
    for s in want:
        q=dict(src[s]);z=v5[s];ref=q.get('reference') or q.get('reference_raw') or z.get('reference_raw') or z.get('reference') or ''
        if not ref:raise RuntimeError('Internal reference missing '+s)
        q['reference_raw']=ref
        if 'slot_labels' not in q:q['slot_labels']=z['slot_labels']
        if 'slot_valid' not in q:q['slot_valid']=z['slot_valid']
        out.append(q)
    return out

def prepare_internal_subset(rows,serials,cfg23,ck23,data23,lordosis_threshold,parser_mod,scaffold_cfg,device='cpu'):
    by={str(r['serial']):r for r in rows};want=[str(x) for x in serials];probs=replay_serials(cfg23,ck23,data23,'internal',want,device=device);cores,ths=binary_from_probs(probs,ck23['thresholds'],lordosis_threshold);out=[]
    for s,p,core in zip(want,probs.tolist(),cores):
        q=dict(by[s]);old=[int(float(x)>.5) for x in q.get('core_binary',core)]
        if len(old)==16 and old!=core:raise RuntimeError(f'Exact Stage2.3 replay core mismatch serial={q["serial"]}')
        q['planner_probs']=list(map(float,p));q['planner_thresholds']=list(map(float,ths));q['core_binary']=list(map(int,core));q=prepare_row(q,parser_mod,scaffold_cfg);out.append(q)
    return out
