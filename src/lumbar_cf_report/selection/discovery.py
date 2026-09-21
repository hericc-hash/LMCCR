from __future__ import annotations
from pathlib import Path
import glob,os

def latest_complete(prefixes,required,env_key=None,label='artifact'):
    e=os.environ.get(env_key or '','').strip() if env_key else ''
    if e:
        p=Path(e).resolve();miss=[x for x in required if not (p/x).exists()]
        if miss:raise RuntimeError(f'Explicit {env_key} incomplete for {label}: {miss}')
        return p,{'mode':'explicit','env_key':env_key}
    xs=[]
    for pat in prefixes:xs += [Path(x) for x in glob.glob(pat)]
    xs=sorted(set(xs),key=lambda p:p.stat().st_mtime if p.exists() else -1,reverse=True)
    ok=[p for p in xs if all((p/x).exists() for x in required)]
    if not ok:raise FileNotFoundError(f'No complete {label} found for prefixes={prefixes}')
    return ok[0],{'mode':'auto_latest_complete','n_candidates':len(xs),'n_complete':len(ok)}
