from __future__ import annotations
LEVELS=['L1/2','L2/3','L3/4','L4/5','L5/S1']
def _bin(xs): return [int(float(x)>.5) for x in xs]
def core_contract(core,valid):
    c=_bin(core);v=_bin(valid)
    if len(c)!=16 or len(v)!=16: raise ValueError('Clinical16 required')
    def st(i): return 'INVALID' if not v[i] else ('POS' if c[i] else 'NEG')
    lines=[f'LORDOSIS={st(0)}']
    k=1
    for task in ['DISC','STENOSIS','NERVE']:
        for lv in LEVELS:
            lines.append(f'{task} {lv}={st(k)}');k+=1
    return '\n'.join(lines)
