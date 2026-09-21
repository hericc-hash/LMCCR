from __future__ import annotations
import importlib,importlib.util
from pathlib import Path
import numpy as np

def load_parser(name):return importlib.import_module(name)
def parser_source(name):
    s=importlib.util.find_spec(name)
    if s is None or not s.origin:raise ImportError(name)
    return Path(s.origin).resolve()
def parse(mod,text):
    f=getattr(mod,'parse_focused_semantics',None)
    if not callable(f):raise RuntimeError('Frozen parser missing parse_focused_semantics')
    return f(text)
def normalize(x):
    """Normalize the exact frozen factual_generation_v1 parser output to Clinical16.

    Production semantic-dict contract (fixed order):
      lordosis, disc:L1/2..L5/S1, stenosis:L1/2..L5/S1, nerve:L1/2..L5/S1.

    The parser uses ``None`` for an unmentioned/undetected semantic slot (notably
    lordosis), so ``None`` must be interpreted as 0 rather than dropped.  Never
    infer Clinical16 by collecting only numeric dict values: that silently turns
    the production 16-slot dict into 15 values whenever lordosis is None.
    """
    semantic_keys=[
        'lordosis',
        'disc:L1/2','disc:L2/3','disc:L3/4','disc:L4/5','disc:L5/S1',
        'stenosis:L1/2','stenosis:L2/3','stenosis:L3/4','stenosis:L4/5','stenosis:L5/S1',
        'nerve:L1/2','nerve:L2/3','nerve:L3/4','nerve:L4/5','nerve:L5/S1',
    ]
    if isinstance(x,dict):
        # Exact production semantic dict: use semantic keys, never insertion order.
        if all(k in x for k in semantic_keys):
            vals=[]
            for k in semantic_keys:
                v=x[k]
                if v is None:
                    vals.append(0.0)
                elif isinstance(v,(bool,int,float,np.integer,np.floating)):
                    vals.append(float(v))
                else:
                    raise RuntimeError(f'Non-scalar frozen parser slot {k}={v!r}')
            return np.asarray(vals,dtype=float)
        # Explicit vector containers remain supported for regression utilities.
        for k in ['slots','predictions','slot_predictions','main','core_binary','binary']:
            if k in x:
                a=np.asarray(x[k],float).reshape(-1)
                if len(a)>=16:return a[:16]
    if isinstance(x,(list,tuple,np.ndarray)):
        a=np.asarray(x,float).reshape(-1)
        if len(a)>=16:return a[:16]
    raise RuntimeError(f'Cannot normalize parser output type={type(x)} value={str(x)[:400]}')

def report_metrics(mod,texts,labels,valid):
    pf=getattr(mod,'parse_focused_semantics',None);mf=getattr(mod,'compute_report_slot_metrics',None)
    if not callable(pf) or not callable(mf):raise RuntimeError('Frozen exact report metric API missing')
    parsed=[pf(x) for x in texts];m=mf(parsed,np.asarray(labels,float),np.asarray(valid,float))
    return parsed,m
def character_metrics(mod,preds,refs):
    f=getattr(mod,'character_metrics',None)
    if not callable(f):raise RuntimeError('Frozen exact character_metrics missing; ROUGE/BLEU comparability required')
    return f(list(preds),list(refs))
