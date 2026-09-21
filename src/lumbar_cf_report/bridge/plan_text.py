from __future__ import annotations
from .schema import LEVELS,TASKS,MAIN_SLOT_NAMES,STRUCTURE_TERMS

def _yn(v):return '阳性' if v else '阴性'
def serialize_plan(main_probs,struct_probs,disease_thresholds,lordosis_threshold=.5,structure_threshold=.5):
    th=[lordosis_threshold]+[float(disease_thresholds[t]) for t in range(3) for _ in range(5)]
    lines=['<临床计划>']
    lines.append(f'腰椎曲度异常: {_yn(main_probs[0]>=th[0])} (p={main_probs[0]:.3f})')
    k=1
    cn={'disc':'椎间盘异常','stenosis':'椎管狭窄','nerve':'神经根受压'}
    for ti,t in enumerate(TASKS):
        for li,lv in enumerate(LEVELS):
            p=float(main_probs[k]);pos=p>=th[k]
            lines.append(f'{lv} {cn[t]}: {_yn(pos)} (p={p:.3f})');k+=1
    if struct_probs is not None:
        lines.append('辅助结构状态:')
        for n,p in zip(STRUCTURE_TERMS,struct_probs):lines.append(f'- {n}: {_yn(float(p)>=structure_threshold)} (p={float(p):.3f})')
    lines.append('</临床计划>')
    return '\n'.join(lines)
