from __future__ import annotations
import re,difflib
from typing import List,Tuple,Dict
from .canonical import SLOT_NAMES,canonical_report

def _bin(xs): return [int(float(x)>.5) for x in xs]
def _vec(parser_mod,parse_fn,normalize_fn,text): return _bin(normalize_fn(parse_fn(parser_mod,text)))[:16]
def _match(got,want,valid): return all((not v) or int(g)==int(w) for g,w,v in zip(got,want,valid))
def _extras(got,want,valid): return {i for i,(g,w,v) in enumerate(zip(got,want,valid)) if v and g and not w}
def _missing(got,want,valid): return {i for i,(g,w,v) in enumerate(zip(got,want,valid)) if v and w and not g}

def _split_sections(text:str):
    t=str(text or '').replace('\r','\n').strip()
    if '所见：' in t:
        t=t.split('所见：',1)[1]
    if '\n结论：' in t:
        f,c=t.split('\n结论：',1)
    elif '结论：' in t:
        f,c=t.split('结论：',1)
    else:
        lines=[x.strip() for x in t.split('\n') if x.strip()]
        f=' '.join(lines);c=''
    return f.strip(),c.strip()

def _sentence_units(section:str):
    out=[]
    for z in re.split(r'[。；;\n]+',str(section)):
        z=z.strip(' ，,。；;\t')
        if z:out.append(z)
    return out

def _render(findings:List[str],conclusion:List[str]):
    f='。'.join(x.strip(' 。；;') for x in findings if x.strip())
    c='。'.join(x.strip(' 。；;') for x in conclusion if x.strip())
    if f:f+='。'
    if c:c+='。'
    return '所见：'+f+'\n结论：'+c

def _ops(units):
    # units: [(section,text)]
    for i,(sec,text) in enumerate(units):
        yield ('drop',i,None)
        parts=[x.strip() for x in re.split(r'[，,]',text) if x.strip()]
        if len(parts)>=2:
            for j in range(len(parts)):
                repl='，'.join(parts[:j]+parts[j+1:]).strip()
                if repl:yield ('subdrop',i,repl)

def _apply(units,op):
    kind,i,repl=op;z=list(units)
    if kind=='drop':z.pop(i)
    else:z[i]=(z[i][0],repl)
    return z

def _render_units(units):
    f=[t for s,t in units if s=='findings'];c=[t for s,t in units if s=='conclusion'];return _render(f,c)

def _append_slot(units,idx,lex):
    phrase=str(lex[SLOT_NAMES[idx]]).strip().strip('。')
    # Append to both sections: the lexicalizer was calibrated in a two-section report.
    units=list(units)+[('findings',phrase),('conclusion',phrase)]
    return units

def sanitize_reference(row,lex,parser_mod,parse_fn,normalize_fn,max_steps=32):
    raw=str(row.get('reference_raw') or '').strip();want=_bin(row['core_binary']);valid=_bin(row['slot_valid'])
    if not raw:raise RuntimeError('reference_raw missing '+str(row.get('serial')))
    got=_vec(parser_mod,parse_fn,normalize_fn,raw)
    if _match(got,want,valid):
        return raw,{'stage':'REFERENCE_ALREADY_CORE_EXACT','reference_core_exact':True,'removed_operations':0,'appended_slots':[],'similarity_to_reference':1.0}
    f,c=_split_sections(raw);units=[('findings',x) for x in _sentence_units(f)]+[('conclusion',x) for x in _sentence_units(c)]
    current=list(units);removed=[]
    for _ in range(max_steps):
        text=_render_units(current);gv=_vec(parser_mod,parse_fn,normalize_fn,text);ex=_extras(gv,want,valid)
        if not ex:break
        best=None
        for op in _ops(current):
            cand=_apply(current,op);ct=_render_units(cand);cv=_vec(parser_mod,parse_fn,normalize_fn,ct);ce=_extras(cv,want,valid)
            reduced=len(ex-ce)
            if reduced<=0:continue
            cur_supported={i for i,(g,w,v) in enumerate(zip(gv,want,valid)) if v and w and g}
            new_supported={i for i,(g,w,v) in enumerate(zip(cv,want,valid)) if v and w and g}
            lost=len(cur_supported-new_supported)
            removed_chars=max(0,len(text)-len(ct))
            score=1000*reduced-120*lost-0.05*removed_chars
            if best is None or score>best[0]:best=(score,op,cand,cv)
        if best is None:break
        _,op,current,_=best;removed.append(op)
    text=_render_units(current);gv=_vec(parser_mod,parse_fn,normalize_fn,text)
    # Explicitly restore any missing positive Core slots with parser-calibrated lexicalizations.
    appended=[]
    for idx in sorted(_missing(gv,want,valid)):
        current=_append_slot(current,idx,lex);appended.append(idx)
        gv=_vec(parser_mod,parse_fn,normalize_fn,_render_units(current))
    text=_render_units(current);gv=_vec(parser_mod,parse_fn,normalize_fn,text)
    if _match(gv,want,valid):
        return text,{'stage':'PARSER_GUIDED_SURGERY','reference_core_exact':False,'removed_operations':len(removed),'appended_slots':[SLOT_NAMES[i] for i in appended],'similarity_to_reference':difflib.SequenceMatcher(None,raw,text).ratio()}
    # Safe natural shell: retain only units that are Clinical16-zero in isolation.
    safe=[]
    for sec,u in units:
        probe=_render([u] if sec=='findings' else [],[u] if sec=='conclusion' else [])
        try:uv=_vec(parser_mod,parse_fn,normalize_fn,probe)
        except Exception:continue
        if not any(uv):safe.append((sec,u))
    current=safe
    for idx,(w,v) in enumerate(zip(want,valid)):
        if w and v:current=_append_slot(current,idx,lex)
    # If zero-style units interact with parser, greedily remove them until exact.
    for _ in range(max_steps):
        text=_render_units(current);gv=_vec(parser_mod,parse_fn,normalize_fn,text)
        if _match(gv,want,valid):
            return text,{'stage':'SAFE_STYLE_SHELL_PLUS_CORE','reference_core_exact':False,'removed_operations':len(units)-len(safe),'appended_slots':[SLOT_NAMES[i] for i,(w,v) in enumerate(zip(want,valid)) if w and v],'similarity_to_reference':difflib.SequenceMatcher(None,raw,text).ratio()}
        ex=_extras(gv,want,valid)
        if not ex:break
        removable=[i for i,(sec,u) in enumerate(current) if (sec,u) in safe]
        best=None
        for i in removable:
            cand=current[:i]+current[i+1:];cv=_vec(parser_mod,parse_fn,normalize_fn,_render_units(cand));red=len(ex-_extras(cv,want,valid))
            if red>0 and (best is None or red>best[0]):best=(red,cand)
        if best is None:break
        current=best[1]
    can=canonical_report(row['core_binary'],row['slot_valid'],lex)
    cv=_vec(parser_mod,parse_fn,normalize_fn,can)
    if not _match(cv,want,valid):raise RuntimeError('canonical fallback not parser-exact')
    return can,{'stage':'CANONICAL_FALLBACK','reference_core_exact':False,'removed_operations':len(units),'appended_slots':[SLOT_NAMES[i] for i,(w,v) in enumerate(zip(want,valid)) if w and v],'similarity_to_reference':difflib.SequenceMatcher(None,raw,can).ratio()}
