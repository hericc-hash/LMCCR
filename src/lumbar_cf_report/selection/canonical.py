from __future__ import annotations
from typing import Dict, List, Optional
import re

LEVELS=['L1/2','L2/3','L3/4','L4/5','L5/S1']
ROOTS={'L1/2':'L2','L2/3':'L3','L3/4':'L4','L4/5':'L5','L5/S1':'S1'}
SLOT_NAMES=['LORDOSIS']+[f'DISC_{x}' for x in LEVELS]+[f'STENOSIS_{x}' for x in LEVELS]+[f'NERVE_{x}' for x in LEVELS]

# Curated fallbacks. Runtime calibration still validates every phrase against the
# exact frozen parser before using it.
CANDIDATES={
    'LORDOSIS':['腰椎生理曲度变直。','腰椎生理曲度稍变直。','腰椎曲度变直。'],
}
for lv in LEVELS:
    CANDIDATES[f'DISC_{lv}']=[f'{lv}椎间盘膨出。',f'{lv}椎间盘突出。',f'{lv}椎间盘异常。']
    CANDIDATES[f'STENOSIS_{lv}']=[f'{lv}水平椎管狭窄。',f'{lv}椎管狭窄。']
    root=ROOTS[lv]
    CANDIDATES[f'NERVE_{lv}']=[f'{lv}水平相应神经根受压。',f'{lv}相应神经根受压。',f'{lv}神经根受压。',f'{root}神经根受压。',f'左侧{root}神经根受压。',f'双侧{root}神经根受压。']

# IMPORTANT: negative Core facts are represented by omission, not by natural-
# language negation. The frozen parser has its own negation rules, so using
# phrases such as “未见明确核心阳性异常” as a canonical negative target is
# unnecessarily brittle. The neutral text must contain no disease/lordosis terms.
NEUTRAL_CANDIDATES=[
    '所见：\n结论：',
    '所见：。\n结论：。',
    '所见：腰椎MRI平扫。\n结论：腰椎MRI平扫。',
    '所见：检查完成。\n结论：请结合临床。',
]

def _vec(x):
    import numpy as np
    return (np.asarray(x,float).reshape(-1)[:16] > .5).astype(int).tolist()

def _wrap_phrase(phrase:str)->str:
    phrase=str(phrase).strip().strip('，,；;。 ')
    return f'所见：{phrase}。\n结论：{phrase}。'

def _clauses(text:str):
    t=str(text or '').replace('\r','\n')
    t=t.replace('所见：','').replace('结论：','')
    parts=re.split(r'[。；;\n]+',t)
    out=[]
    for z in parts:
        z=z.strip(' ，,。；;\t')
        if 2 <= len(z) <= 120:
            out.append(z)
    return out

def _dedupe(xs):
    seen=set();out=[]
    for x in xs:
        s=str(x).strip()
        if not s or s in seen:continue
        seen.add(s);out.append(s)
    return out

def calibrate_lexicalizer(parser_mod, parse_fn, normalize_fn, rows:Optional[List[dict]]=None) -> Dict[str,str]:
    """Calibrate a parser-exact positive-only canonical lexicalizer.

    Strategy:
      1) calibrate a neutral report with no clinical slot terms;
      2) for each positive slot, first try real Development388 clauses mined from
         reference reports, then curated fallbacks;
      3) accept a phrase only if the exact frozen parser returns a one-hot vector.

    The resulting canonical report never verbalizes Core negatives. Negatives are
    encoded by absence of positive claims, which avoids parser-specific negation
    semantics and is sufficient for the frozen Clinical16 evaluator.
    """
    rows=rows or []
    zeros=[0]*16
    diag={'neutral_probes':[],'slot_probes':{}}

    neutral_pool=list(NEUTRAL_CANDIDATES)
    # Add real reports that the frozen parser itself already interprets as all-zero.
    for r in rows:
        ref=r.get('reference_raw')
        if not ref:continue
        try:
            got=_vec(normalize_fn(parse_fn(parser_mod,ref)))
        except Exception:
            continue
        if got==zeros:
            neutral_pool.append(str(ref))
    neutral=None
    for text in _dedupe(neutral_pool):
        try:
            got=_vec(normalize_fn(parse_fn(parser_mod,text)))
            diag['neutral_probes'].append({'text':text[:240],'vector':got})
            if got==zeros:
                neutral=text;break
        except Exception as e:
            diag['neutral_probes'].append({'text':text[:240],'error':repr(e)})
    if neutral is None:
        raise RuntimeError('Could not calibrate all-zero canonical report against frozen parser; probes='+str(diag['neutral_probes'][:8]))

    lex={'__neutral__':neutral,'__strategy__':'POSITIVE_ONLY_NEGATIVE_BY_OMISSION'}
    for idx,name in enumerate(SLOT_NAMES):
        want=[0]*16;want[idx]=1
        pool=[]
        # Mine actual clauses from rows where this Oracle slot is positive. Real
        # wording is preferred because it is closest to the frozen parser corpus.
        for r in rows:
            core=r.get('core_binary') or []
            valid=r.get('slot_valid') or []
            if len(core)<16 or len(valid)<16:continue
            if float(valid[idx])<=.5 or float(core[idx])<=.5:continue
            ref=r.get('reference_raw','')
            try:
                whole=_vec(normalize_fn(parse_fn(parser_mod,ref))) if ref else None
                if whole==want:pool.append(str(ref))
            except Exception:
                pass
            pool.extend(_wrap_phrase(cl) for cl in _clauses(ref))
        pool.extend(_wrap_phrase(x) for x in CANDIDATES[name])
        ok=None;probes=[]
        for text in _dedupe(pool)[:2000]:
            try:
                got=_vec(normalize_fn(parse_fn(parser_mod,text)))
                if len(probes)<20:probes.append({'text':text[:240],'vector':got})
            except Exception as e:
                if len(probes)<20:probes.append({'text':text[:240],'error':repr(e)})
                continue
            if got==want:
                # Store only the factual payload, not duplicate headers, so the
                # canonical composer remains concise.
                payload=text
                if text.startswith('所见：') and '\n结论：' in text:
                    payload=text[len('所见：'):].split('\n结论：',1)[0].strip()
                ok=payload;break
        diag['slot_probes'][name]=probes
        if ok is None:
            raise RuntimeError(f'No parser-exact lexicalization candidate for slot {idx} {name}; probes={probes[:8]}')
        lex[name]=ok
    lex['__diagnostics__']=diag
    return lex

def canonical_report(core_binary, valid, lex:Dict[str,str]) -> str:
    core=[int(float(x)>.5) for x in core_binary]
    val=[int(float(x)>.5) for x in valid]
    if len(core)!=16 or len(val)!=16:raise ValueError('Clinical16 required')
    pos=[SLOT_NAMES[i] for i,(c,v) in enumerate(zip(core,val)) if c and v]
    if not pos:return lex['__neutral__']
    findings=[];conclusion=[]
    # Positive-only canonicalization: never emit explicit NEG claims.
    if core[0] and val[0]:findings.append(lex['LORDOSIS'])
    for name in pos:
        if name=='LORDOSIS':continue
        findings.append(lex[name]);conclusion.append(lex[name])
    if not findings:findings.append('腰椎MRI平扫。')
    if not conclusion:
        conclusion.append(lex['LORDOSIS'] if core[0] and val[0] else '腰椎MRI平扫。')
    return '所见：'+''.join(findings)+'\n结论：'+''.join(conclusion)

def validate_rows(rows, lex, parser_mod, parse_fn, normalize_fn):
    bad=[]
    for r in rows:
        text=canonical_report(r['core_binary'],r['slot_valid'],lex)
        got=_vec(normalize_fn(parse_fn(parser_mod,text)))
        valid=[int(float(v)>.5) for v in r['slot_valid']]
        want=[int(float(c)>.5) for c in r['core_binary']]
        mismatch=[i for i,(g,w,v) in enumerate(zip(got,want,valid)) if v and g!=w]
        if mismatch:bad.append({'serial':r.get('serial'),'mismatch':mismatch,'text':text,'got':got,'want':want})
    return bad
