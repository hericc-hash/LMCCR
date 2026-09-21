from __future__ import annotations
import re,difflib
from .canonical import SLOT_NAMES,ROOTS

CONNECTORS='及并伴和与、'
PUNCT='，,。；;\n'

def _bin(xs):return [int(float(x)>.5) for x in xs]
def _vec(text,mod,parse_fn,norm):
    import numpy as np
    return (np.asarray(norm(parse_fn(mod,text)),float).reshape(-1)[:16]>.5).astype(int).tolist()
def _unsupported(g,w,v):return [i for i,(a,b,c) in enumerate(zip(g,w,v)) if c and a and not b]
def _missing(g,w,v):return [i for i,(a,b,c) in enumerate(zip(g,w,v)) if c and b and not a]
def _exact(g,w,v):return all((not c) or a==b for a,b,c in zip(g,w,v))

def _clean(t):
    t=str(t);t=re.sub(r'[ \t]+','',t);t=re.sub(r'，{2,}','，',t);t=re.sub(r'；{2,}','；',t);t=re.sub(r'。{2,}','。',t)
    t=re.sub(r'([，；。\n])(?:及|并|伴|和|与|、)+',r'\1',t);t=re.sub(r'(?:及|并|伴|和|与|、)+([，；。\n])',r'\1',t)
    t=t.replace('，。','。').replace('；。','。').replace('，；','；').replace('；，','；').replace('所见：，','所见：').replace('结论：，','结论：')
    return t.strip()

def _slot_meta(idx):
    name=SLOT_NAMES[idx]
    if idx==0:return {'task':'LORDOSIS','level':None,'root':None,'name':name}
    task,level=name.split('_',1);return {'task':task,'level':level,'root':ROOTS.get(level),'name':name}

def _content_spans(text):
    spans=[]
    for m in re.finditer(r'[^，,。；;\n]+',str(text)):
        s,e=m.span();chunk=m.group(0)
        for h in ('所见：','结论：'):
            if chunk.startswith(h):s+=len(h);chunk=chunk[len(h):]
        if chunk.strip():
            lead=len(chunk)-len(chunk.lstrip());trail=len(chunk)-len(chunk.rstrip());ss=s+lead;ee=e-trail
            if ee>ss:spans.append((ss,ee,text[ss:ee]))
    return spans

def _probe_part(part,mod,parse_fn,norm):return _vec('所见：'+str(part).strip()+'。\n结论：'+str(part).strip()+'。',mod,parse_fn,norm)

def _grammar_penalty(z):
    z=str(z).strip(' ，,。；;\n');p=0
    if not z:return 0
    if re.search(r'^(?:及|并|伴|和|与|、)|(?:及|并|伴|和|与|、)$',z):p+=4
    if re.search(r'(?:L[1-5](?:/(?:[1-5]|S1))?|S1|双侧|左侧|右侧|相应|水平|椎管|神经根|椎间盘)$',z):p+=4
    if re.search(r'(?:椎管|神经根|椎间盘)(?:，|。|；|$)',z) and not any(x in z for x in ('狭窄','受压','膨出','突出','脱出','异常')):p+=3
    return p

def _regex_candidate_spans(part,meta):
    task,lv,root=meta['task'],meta['level'],meta['root'];out=set()
    if task=='LORDOSIS':patterns=[r'腰椎(?:生理)?曲度[^，,。；;\n]{0,10}(?:变直|稍变直)']
    elif task=='DISC':patterns=[rf'{re.escape(lv)}[^，,。；;\n]{{0,16}}(?:椎间盘)?[^，,。；;\n]{{0,8}}(?:膨出|突出|脱出|疝出|异常)',rf'{re.escape(lv)}']
    elif task=='STENOSIS':patterns=[rf'{re.escape(lv)}[^，,。；;\n]{{0,18}}(?:椎管)?[^，,。；;\n]{{0,8}}狭窄',rf'{re.escape(lv)}']
    else:
        patterns=[rf'{re.escape(lv)}[^，,。；;\n]{{0,18}}(?:神经根|马尾)[^，,。；;\n]{{0,10}}(?:受压|受累|聚集)',rf'(?:双侧|左侧|右侧)?{re.escape(root)}[^，,。；;\n]{{0,8}}神经根[^，,。；;\n]{{0,8}}(?:受压|受累)',rf'{re.escape(root)}']
    for pat in patterns:
        for m in re.finditer(pat,part,re.I):
            out.add(m.span())
            a,b=m.span()
            if a>0 and part[a-1] in CONNECTORS+'，,':out.add((a-1,b))
            if b<len(part) and part[b:b+1] in CONNECTORS:out.add((a,b+1))
    return out

def _generic_near_target(part,meta,min_len,max_len,context):
    out=set();seeds=[];lv=meta['level'];root=meta['root'];task=meta['task']
    if lv:
        for m in re.finditer(re.escape(lv),part,re.I):seeds.extend([m.start(),m.end()])
    if root and task=='NERVE':
        for m in re.finditer(re.escape(root),part,re.I):seeds.extend([m.start(),m.end()])
    cues={'DISC':['椎间盘','膨出','突出','脱出'],'STENOSIS':['椎管','狭窄'],'NERVE':['神经根','马尾','受压'],'LORDOSIS':['曲度','变直']}[task]
    for cue in cues:
        for m in re.finditer(cue,part):seeds.extend([m.start(),m.end()])
    n=len(part)
    for c in seeds:
        lo=max(0,c-context);hi=min(n,c+context)
        for a in range(lo,min(c+1,n)):
            for L in range(min_len,min(max_len,n-a)+1):
                b=a+L
                if b>hi:break
                if a<c<b or c in (a,b):out.add((a,b))
    return out

def _best_targeted_delete(text,got,want,valid,target_idx,mod,parse_fn,norm,cfg):
    if not got[target_idx] or want[target_idx] or not valid[target_idx]:return None
    present_wanted=[i for i,(g,w,v) in enumerate(zip(got,want,valid)) if v and g and w]
    meta=_slot_meta(target_idx);cur_u=_unsupported(got,want,valid);best=None
    min_len=int(cfg.get('generic_min_span_chars',2));max_len=int(cfg.get('generic_max_span_chars',34));context=int(cfg.get('target_context_chars',24))
    for ps,pe,part in _content_spans(text):
        try:pv=_probe_part(part,mod,parse_fn,norm)
        except Exception:continue
        if not pv[target_idx]:continue
        spans=_regex_candidate_spans(part,meta)|_generic_near_target(part,meta,min_len,max_len,context)|{(0,len(part))}
        for a,b in sorted(spans,key=lambda x:(x[1]-x[0],x[0])):
            if not (0<=a<b<=len(part)):continue
            frag=part[a:b]
            cand=_clean(text[:ps+a]+text[ps+b:])
            if cand==text:continue
            try:
                vg=_vec(cand,mod,parse_fn,norm)
                local_after=_probe_part(part[:a]+part[b:],mod,parse_fn,norm)
            except Exception:continue
            # A target fact may be repeated in Findings and Conclusion. Accept a deletion
            # when it removes the target from THIS local clause even if another occurrence
            # keeps the global slot positive; the outer loop will patch the next occurrence.
            if local_after[target_idx]:continue
            lost=sum(1 for i in present_wanted if not vg[i]);u_after=_unsupported(vg,want,valid);residue=_clean(part[:a]+part[b:]);gp=_grammar_penalty(residue)
            regex_hit=(a,b) in _regex_candidate_spans(part,meta)
            score=(lost,1 if vg[target_idx] else 0,len(u_after),gp,0 if regex_hit else 1,b-a,a)
            rec={'score':score,'text':cand,'got':vg,'target_idx':target_idx,'target_name':meta['name'],'deleted':frag,'span':[ps+a,ps+b],'grammar_penalty':gp,'unsupported_after':u_after,'regex_hit':regex_hit}
            if best is None or score<best['score']:best=rec
    return best

def _insert_conclusion(text,phrase):
    phrase=str(phrase).strip().strip('，,；;。 ');t=str(text).rstrip()
    if '结论：' in t:
        h,c=t.rsplit('结论：',1);c=c.strip().rstrip('。；;，, ');c=(c+'；' if c else '')+phrase;return _clean(h+'结论：'+c+'。')
    return _clean(t+'\n结论：'+phrase+'。')

def _insert_findings(text,phrase):
    phrase=str(phrase).strip().strip('，,；;。 ');t=str(text)
    if '所见：' in t:
        h,rest=t.split('所见：',1)
        if '\n结论：' in rest:f,c=rest.split('\n结论：',1);f=f.strip().rstrip('。；;，, ');f=(f+'；' if f else '')+phrase;return _clean(h+'所见：'+f+'。\n结论：'+c)
    return _clean('所见：'+phrase+'。\n'+t)

def _best_missing_insert(text,got,want,valid,target_idx,phrase,mod,parse_fn,norm):
    if got[target_idx] or not want[target_idx] or not valid[target_idx]:return None
    cur_u=_unsupported(got,want,valid);cands=[]
    for mode,cand in [('conclusion',_insert_conclusion(text,phrase)),('findings',_insert_findings(text,phrase))]:
        try:vg=_vec(cand,mod,parse_fn,norm)
        except Exception:continue
        if not vg[target_idx]:continue
        new_u=_unsupported(vg,want,valid);added_bad=len(set(new_u)-set(cur_u));sim=difflib.SequenceMatcher(None,text,cand).ratio();score=(added_bad,len(new_u),1-sim,0 if mode=='conclusion' else 1)
        cands.append({'score':score,'text':cand,'got':vg,'mode':mode,'target_idx':target_idx,'target_name':SLOT_NAMES[target_idx],'phrase':phrase})
    return min(cands,key=lambda x:x['score']) if cands else None

def precision_finalize(raw,core,valid,lex,mod,parse_fn,norm,cfg=None):
    cfg=cfg or {};want=_bin(core);val=_bin(valid);raw=str(raw or '').strip();rawgot=_vec(raw,mod,parse_fn,norm);text=raw;got=rawgot
    before_u=_unsupported(got,want,val);before_m=_missing(got,want,val);deletions=[];insertions=[];attempted=0;success=0
    for _ in range(int(cfg.get('maximum_iterations',16))):
        us=_unsupported(got,want,val)
        if not us:break
        progressed=False
        # Harder slots first: nerve, stenosis, disc, lordosis; within task preserve order.
        order=sorted(us,key=lambda i:({11:0,12:0,13:0,14:0,15:0,6:1,7:1,8:1,9:1,10:1,1:2,2:2,3:2,4:2,5:2,0:3}.get(i,4),i))
        for idx in order:
            attempted+=1;rec=_best_targeted_delete(text,got,want,val,idx,mod,parse_fn,norm,cfg)
            if rec is None:continue
            text=rec['text'];got=rec['got'];deletions.append(rec);success+=1;progressed=True;break
        if not progressed:break
    # Missing positives: targeted minimal one-phrase insertion, one slot at a time.
    for idx in list(_missing(got,want,val)):
        phrase=str(lex[SLOT_NAMES[idx]]).strip().strip('，,；;。 ')
        if not phrase:continue
        attempted+=1;rec=_best_missing_insert(text,got,want,val,idx,phrase,mod,parse_fn,norm)
        if rec is None:continue
        text=rec['text'];got=rec['got'];insertions.append(rec);success+=1
    after_u=_unsupported(got,want,val);after_m=_missing(got,want,val);sim=difflib.SequenceMatcher(None,raw,text).ratio() if raw or text else 1.0
    return text,{
        'changed':text!=raw,'fallback':False,'raw_exact':_exact(rawgot,want,val),'final_exact':_exact(got,want,val),
        'unsupported_before':len(before_u),'unsupported_after':len(after_u),'missing_before':len(before_m),'missing_after':len(after_m),
        'deleted_spans':len(deletions),'deleted_chars':sum(len(x['deleted']) for x in deletions),
        'inserted_phrases':len(insertions),'inserted_chars':sum(len(x['phrase']) for x in insertions),
        'unresolved_slots':len(set(after_u+after_m)),'unresolved_slot_names':[SLOT_NAMES[i] for i in sorted(set(after_u+after_m))],
        'targeted_patch_attempts':attempted,'targeted_patch_successes':success,'targeted_patch_success_rate':success/attempted if attempted else 1.0,
        'similarity_raw_final':sim,'edit_char_fraction':1.0-sim,
        'deletions':[{k:v for k,v in x.items() if k not in ('text','got','score')} for x in deletions],
        'insertions':[{k:v for k,v in x.items() if k not in ('text','got','score')} for x in insertions],
    }
