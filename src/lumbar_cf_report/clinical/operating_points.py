from __future__ import annotations
from pathlib import Path
import json
import numpy as np

LEVELS = ["L1/2", "L2/3", "L3/4", "L4/5", "L5/S1"]
TASKS = ["disc", "stenosis", "nerve"]
MAIN_SLOT_NAMES = ["lordosis"] + [f"{t}:{l}" for t in TASKS for l in LEVELS]


def load_thresholds(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    th=obj.get('thresholds',obj)
    required=['lordosis','disc','stenosis','nerve','structure']
    miss=[k for k in required if k not in th]
    if miss: raise KeyError(f'threshold config missing {miss}')
    out={k:float(th[k]) for k in required}
    for k,v in out.items():
        if not 0.0 < v < 1.0: raise ValueError(f'invalid threshold {k}={v}')
    return out,obj


def threshold_for_slot_name(name, th):
    if name=='lordosis': return float(th['lordosis'])
    if ':' not in name: raise KeyError(name)
    task=name.split(':',1)[0]
    if task not in TASKS: raise KeyError(name)
    return float(th[task])


def threshold_for_slot_index(slot_index, th):
    return threshold_for_slot_name(MAIN_SLOT_NAMES[int(slot_index)],th)


def binarize_main(probabilities, th):
    p=np.asarray(probabilities,float)
    if p.shape[-1] != len(MAIN_SLOT_NAMES):
        raise ValueError(f'expected last dim 16, got {p.shape}')
    ts=np.asarray([threshold_for_slot_name(n,th) for n in MAIN_SLOT_NAMES],float)
    return (p>=ts).astype(int)

