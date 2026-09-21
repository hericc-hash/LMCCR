from __future__ import annotations
from pathlib import Path
import json,os
from .io import safe_load,sha256_file

def find_phasea(cfg):
    env=os.environ.get('PHASEA_DIR','').strip()
    dirs=[]
    if env:dirs=[Path(env)]
    else:
        for root in cfg.get('phaseA_search_roots',[]):
            p=Path(root)
            if p.exists():
                dirs.extend([q.parent for q in p.rglob(cfg['phaseA_candidate_name'])])
    good=[]
    for d in dirs:
        cp=d/cfg['phaseA_candidate_name'];dp=d/cfg['phaseA_internal_decision_name']
        if not cp.exists() or not dp.exists():continue
        try:dec=json.load(open(dp))
        except Exception:continue
        if dec.get('verdict')!='STAGE2_3_FACTUAL_FULL_PASS' or not dec.get('cf_phase_unlock',False):continue
        ck=safe_load(cp)
        if ck.get('selection_mode') not in ('CONTEXT_RESIDUAL','ANCHOR_ONLY'):continue
        good.append((cp.stat().st_mtime,d,ck,dec))
    if not good:raise FileNotFoundError('No Stage2.3-A FULL_PASS result found. Set PHASEA_DIR explicitly.')
    good.sort(reverse=True,key=lambda z:z[0]);_,d,ck,dec=good[0]
    return {'dir':str(d),'candidate':str(d/cfg['phaseA_candidate_name']),'candidate_sha256':sha256_file(d/cfg['phaseA_candidate_name']),'checkpoint':ck,'internal_decision':dec}
