from __future__ import annotations
from .io import read_jsonl

def _bin(xs):return [int(float(x)>.5) for x in xs]
def build_train388(path):
    base={}
    for r in read_jsonl(path):
        if r.get('sample_type')!='oracle_core_sft':continue
        s=str(r['serial']);q=dict(r);q['slot_labels']=_bin(q['core_binary'])
        if 'slot_valid' not in q or len(q['slot_valid'])<16:raise RuntimeError('train388 row missing Clinical16 slot_valid '+s)
        if not isinstance(q.get('direct_draft'),str):raise RuntimeError('train388 row missing direct_draft '+s)
        ref=q.get('reference_raw',q.get('reference',''))
        if not isinstance(ref,str) or not ref:raise RuntimeError('train388 row missing reference '+s)
        q['reference_raw']=ref;base[s]=q
    if len(base)!=388:raise RuntimeError(f'Expected 388 oracle_core_sft rows, got {len(base)}')
    return [base[k] for k in sorted(base,key=lambda x:(0,int(x)) if x.isdigit() else (1,x))]
def merge_plans(rows,serials,probs,cores,thresholds):
    pmap={str(s):(list(map(float,p)),list(map(int,c))) for s,p,c in zip(serials,probs,cores)};out=[]
    for r in rows:
        s=str(r['serial'])
        if s not in pmap:raise RuntimeError('Planner replay missing serial '+s)
        q=dict(r);q['planner_probs'],q['core_binary']=pmap[s];q['planner_thresholds']=list(map(float,thresholds));out.append(q)
    return out
