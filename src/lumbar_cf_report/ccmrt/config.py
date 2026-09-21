from __future__ import annotations
import copy

def _parse_value(v):
    s=str(v).strip()
    if s.lower() in {"true","false"}: return s.lower()=="true"
    if s.lower() in {"none","null"}: return None
    try:
        if "." in s or "e" in s.lower(): return float(s)
        return int(s)
    except Exception:
        return s

def parse_overrides(items):
    out={}
    for item in items:
        if "=" not in item: raise ValueError(f"override must be key=value: {item}")
        key,val=item.split("=",1); cur=out
        parts=key.split(".")
        for p in parts[:-1]: cur=cur.setdefault(p,{})
        cur[parts[-1]]=_parse_value(val)
    return out

def deep_update(base, override):
    out=copy.deepcopy(base)
    for k,v in override.items():
        if isinstance(v,dict) and isinstance(out.get(k),dict): out[k]=deep_update(out[k],v)
        else: out[k]=v
    return out

