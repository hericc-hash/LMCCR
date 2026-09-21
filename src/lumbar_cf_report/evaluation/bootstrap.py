from __future__ import annotations
from collections import defaultdict
import numpy as np

def cluster_boot_mean(rows,patient_key,value_fn,B=5000,seed=20260829):
    by=defaultdict(list)
    for r in rows:
        v=value_fn(r)
        if v is None:continue
        try:v=float(v)
        except:continue
        if not np.isfinite(v):continue
        by[int(r[patient_key])].append(v)
    pats=sorted(by)
    if not pats:return {'estimate':None,'ci95':[None,None],'n_patients':0,'bootstrap_replicates':B}
    est=float(np.mean([x for p in pats for x in by[p]]));rng=np.random.default_rng(seed);vals=[]
    for _ in range(B):
        samp=rng.choice(pats,size=len(pats),replace=True);a=[]
        for p in samp:a.extend(by[int(p)])
        vals.append(float(np.mean(a)))
    lo,hi=np.quantile(vals,[.025,.975]);return {'estimate':est,'ci95':[float(lo),float(hi)],'n_patients':len(pats),'bootstrap_replicates':B}

def paired_cluster_boot(diff_rows,patient_key,value_key,B=5000,seed=20260829):
    return cluster_boot_mean(diff_rows,patient_key,lambda r:r.get(value_key),B,seed)

