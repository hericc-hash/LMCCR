from __future__ import annotations
import re,difflib

# Generic placeholder deliberately carries no task, level, polarity, or disease identity.
DEFAULT_PLACEHOLDER='<CORE>'
_DELIM_RE=re.compile(r'([，,。；;\n])')
_CORE_PATTERNS=[
    r'(?:L[1-5]/(?:[1-5]|S1)[^，,。；;\n]{0,10})?椎间盘[^，,。；;\n]{0,8}(?:膨出|突出|脱出|疝出|异常)',
    r'(?:L[1-5]/(?:[1-5]|S1)[^，,。；;\n]{0,10})?(?:水平)?椎管[^，,。；;\n]{0,8}(?:轻度|中度|重度|明显|不同程度|稍)?狭窄',
    r'(?:双侧|左侧|右侧)?(?:L[1-5]|S1)?神经根[^，,。；;\n]{0,8}(?:受压|受累|受侵)',
    r'(?:L[1-5]/(?:[1-5]|S1)[^，,。；;\n]{0,10})?(?:相应)?神经根[^，,。；;\n]{0,8}(?:受压|受累|受侵)',
    r'马尾神经[^，,。；;\n]{0,8}(?:受压|聚集|受压聚集)',
    r'腰椎(?:生理)?曲度[^，,。；;\n]{0,8}(?:变直|稍变直)',
]
_CORE_RE=re.compile('|'.join('(?:'+p+')' for p in _CORE_PATTERNS),re.I)
_CUE_RE=re.compile(r'椎间盘|椎管|神经根|马尾神经|曲度|狭窄|膨出|突出|脱出|受压|L[1-5]/(?:[1-5]|S1)|L[1-5]|S1',re.I)

def _binvec(text,mod,parse_fn,norm):
    import numpy as np
    return (np.asarray(norm(parse_fn(mod,text)),float).reshape(-1)[:16]>.5).astype(int).tolist()

def _probe_clause(clause,mod,parse_fn,norm):
    z=str(clause).strip(' ，,。；;\t')
    if not z:return [0]*16
    return _binvec('所见：'+z+'。\n结论：'+z+'。',mod,parse_fn,norm)

def _clean_placeholders(text,placeholder):
    p=re.escape(placeholder)
    t=re.sub(r'(?:'+p+r'\s*){2,}',placeholder,str(text))
    t=re.sub(r'\s+','',t)
    return t

def _lexical_redact(clause,placeholder):
    return _clean_placeholders(_CORE_RE.sub(placeholder,str(clause)),placeholder)

def _candidate_spans_near_cues(text,min_len,max_len):
    n=len(text);out=set()
    cues=[]
    for m in _CUE_RE.finditer(text):cues.extend([m.start(),m.end()])
    # Whole clause remains a last resort for parser-positive mixed fragments.
    if n:out.add((0,n))
    for c in cues:
        lo=max(0,c-max_len);hi=min(n,c+max_len)
        starts=range(lo,min(c+1,n))
        for a in starts:
            minb=max(a+min_len,c if c>a else a+min_len)
            maxb=min(n,a+max_len,hi)
            for b in range(minb,maxb+1):
                if a<b:out.add((a,b))
    return sorted(out,key=lambda x:(x[1]-x[0],x[0]))

def _parser_guided_redact(clause,mod,parse_fn,norm,placeholder,min_len,max_len,max_iter=8):
    text=str(clause);redacted=[]
    for _ in range(max_iter):
        got=_probe_clause(text,mod,parse_fn,norm)
        if not any(got):break
        best=None
        for a,b in _candidate_spans_near_cues(text,min_len,max_len):
            frag=text[a:b]
            if placeholder in frag and frag.strip()==placeholder:continue
            cand=_clean_placeholders(text[:a]+placeholder+text[b:],placeholder)
            if cand==text:continue
            try:vg=_probe_clause(cand,mod,parse_fn,norm)
            except Exception:continue
            reduction=sum(got)-sum(vg)
            if reduction<=0:continue
            # Prefer more parser positives removed, then shorter replacements, then preservation.
            score=(-reduction,b-a,a)
            if best is None or score<best[0]:best=(score,cand,frag,vg)
        if best is None:break
        _,text,frag,_=best;redacted.append(frag)
    return text,redacted

def _split_sections(text):
    t=str(text or '').replace('\r','\n').strip()
    if '所见：' in t:t=t.split('所见：',1)[1]
    if '\n结论：' in t:f,c=t.split('\n结论：',1)
    elif '结论：' in t:f,c=t.split('结论：',1)
    else:f,c=t,''
    return f,c

def _render_sections(f,c):return '所见：'+str(f).strip()+'\n结论：'+str(c).strip()

def _redact_section(section,mod,parse_fn,norm,placeholder,min_len,max_len):
    pieces=_DELIM_RE.split(str(section));out=[];redacted=[];fallback_parts=0
    for p in pieces:
        if p in {'，',',','。','；',';','\n'}:out.append(p);continue
        if not p:continue
        z=_lexical_redact(p,placeholder)
        try:got=_probe_clause(z,mod,parse_fn,norm)
        except Exception:got=[1]*16
        if any(got):
            z,rr=_parser_guided_redact(z,mod,parse_fn,norm,placeholder,min_len,max_len);redacted.extend(rr)
        # Hard lexical guard after parser-guided surgery.
        z=_lexical_redact(z,placeholder)
        try:got=_probe_clause(z,mod,parse_fn,norm)
        except Exception:got=[1]*16
        if any(got):
            # Preserve punctuation position but do not leak a Clinical16-positive phrase.
            z=placeholder;fallback_parts+=1
        if z.strip():out.append(z)
    return _clean_placeholders(''.join(out),placeholder),redacted,fallback_parts

def _zero_only_fallback(direct,mod,parse_fn,norm,placeholder):
    f,c=_split_sections(direct);outs=[]
    for sec in (f,c):
        kept=[]
        for unit in re.split(r'([，,。；;\n])',sec):
            if unit in {'，',',','。','；',';','\n'}:
                if kept and kept[-1] not in {'，',',','。','；',';','\n'}:kept.append(unit)
                continue
            z=_lexical_redact(unit,placeholder).strip()
            if not z:continue
            try:v=_probe_clause(z,mod,parse_fn,norm)
            except Exception:v=[1]*16
            if not any(v):kept.append(z)
            else:kept.append(placeholder)
        outs.append(''.join(kept))
    return _render_sections(*outs)

def build_scaffold(direct,mod,parse_fn,norm,cfg):
    placeholder=str(cfg.get('placeholder',DEFAULT_PLACEHOLDER));min_len=int(cfg.get('parser_guided_min_span_chars',2));max_len=int(cfg.get('parser_guided_max_span_chars',36));max_chars=int(cfg.get('maximum_chars',1000))
    direct=str(direct or '').strip();f,c=_split_sections(direct)
    rf,rr1,fb1=_redact_section(f,mod,parse_fn,norm,placeholder,min_len,max_len);rc,rr2,fb2=_redact_section(c,mod,parse_fn,norm,placeholder,min_len,max_len)
    scaffold=_render_sections(rf,rc)
    if len(scaffold)>max_chars:scaffold=scaffold[:max_chars].rsplit('。',1)[0]+'。\n结论：'
    got=_binvec(scaffold,mod,parse_fn,norm);hard_fallback=False
    if any(got):
        scaffold=_zero_only_fallback(direct,mod,parse_fn,norm,placeholder);hard_fallback=True;got=_binvec(scaffold,mod,parse_fn,norm)
    if any(got):raise RuntimeError('Core-redacted scaffold failed Clinical16-zero hard contract')
    payload=re.sub(r'所见：|结论：|[\s，,。；;]','',scaffold)
    safe_payload=payload.replace(placeholder,'')
    direct_payload=re.sub(r'所见：|结论：|[\s，,。；;]','',direct)
    preserved=difflib.SequenceMatcher(None,direct_payload,safe_payload).ratio() if direct_payload or safe_payload else 1.0
    return scaffold,{
        'parser_zero':True,'chars':len(scaffold),'payload_chars':len(payload),'safe_payload_chars':len(safe_payload),
        'placeholder_count':scaffold.count(placeholder),'parser_guided_redactions':len(rr1)+len(rr2),
        'local_fallback_parts':fb1+fb2,'hard_fallback':hard_fallback,'preserved_safe_fraction':preserved,
        'empty':len(safe_payload)<2,
    }


def _redact_exact_lexicalizer_phrases(text, lex, placeholder):
    """Replace parser-calibrated Core lexicalizations before target-side projection.

    This is used only for target-scaffold similarity. The training target itself is
    unchanged. Longest phrases are replaced first to avoid partial overlaps.
    """
    t=str(text or '')
    vals=[]
    for k,v in (lex or {}).items():
        if str(k).startswith('__'):continue
        z=str(v or '').strip().strip('，,；;。 ')
        if z:vals.append(z)
    for z in sorted(set(vals),key=len,reverse=True):
        t=t.replace(z,placeholder)
    return _clean_placeholders(t,placeholder)

def build_similarity_scaffold(text, lex, mod, parse_fn, norm, cfg):
    """Best-effort parser-zero scaffold for *similarity only*.

    Source/Direct scaffolds still use ``build_scaffold`` and retain the strict
    Clinical16-zero hard contract. This helper is deliberately non-fatal because
    the target-side scaffold is only an auxiliary safe-copy projection, not a model
    input and not a factual safety boundary.

    Strategy:
      1) remove exact parser-calibrated Core lexicalizations;
      2) try the normal strict scaffold builder;
      3) if a rare target wording still leaks, project each local unit to either a
         parser-zero safe fragment or <CORE>;
      4) if cross-unit parser interactions remain, return a neutral placeholder
         scaffold rather than aborting the full388 build.
    """
    placeholder=str(cfg.get('placeholder',DEFAULT_PLACEHOLDER))
    prepared=_redact_exact_lexicalizer_phrases(text,lex,placeholder)
    try:
        sc,meta=build_scaffold(prepared,mod,parse_fn,norm,cfg)
        meta=dict(meta);meta['similarity_projection_mode']='STRICT_AFTER_EXACT_LEX_REDACTION'
        return sc,meta
    except RuntimeError:
        pass

    f,c=_split_sections(prepared)
    projected=[]
    for sec in (f,c):
        out=[]
        for unit in _DELIM_RE.split(str(sec)):
            if unit in {'，',',','。','；',';','\n'}:
                out.append(unit);continue
            if not unit:continue
            z=_lexical_redact(unit,placeholder).strip()
            if not z:continue
            try:v=_probe_clause(z,mod,parse_fn,norm)
            except Exception:v=[1]*16
            out.append(z if not any(v) else placeholder)
        projected.append(_clean_placeholders(''.join(out),placeholder))
    sc=_render_sections(projected[0],projected[1])
    try:got=_binvec(sc,mod,parse_fn,norm)
    except Exception:got=[1]*16
    mode='LOCAL_ZERO_PROJECTION'
    if any(got):
        # Auxiliary-only last resort. Keep a deterministic parser-zero shape so a
        # rare target expression can reduce its own copy score but never abort
        # training or relax the source scaffold safety contract.
        sc='所见：'+placeholder+'。\n结论：'+placeholder+'。'
        try:got=_binvec(sc,mod,parse_fn,norm)
        except Exception:got=[1]*16
        mode='NEUTRAL_PLACEHOLDER_PROJECTION'
    if any(got):
        # <CORE> should be parser-zero by contract, but retain an empty neutral
        # fallback for maximum robustness of this similarity-only branch.
        sc='所见：\n结论：'
        got=_binvec(sc,mod,parse_fn,norm)
        mode='EMPTY_NEUTRAL_PROJECTION'
    payload=re.sub(r'所见：|结论：|[\s，,。；;]','',sc)
    safe_payload=payload.replace(placeholder,'')
    return sc,{
        'parser_zero':not any(got),'chars':len(sc),'payload_chars':len(payload),
        'safe_payload_chars':len(safe_payload),'placeholder_count':sc.count(placeholder),
        'parser_guided_redactions':0,'local_fallback_parts':0,'hard_fallback':mode!='LOCAL_ZERO_PROJECTION',
        'preserved_safe_fraction':0.0,'empty':len(safe_payload)<2,
        'similarity_projection_mode':mode,
    }
