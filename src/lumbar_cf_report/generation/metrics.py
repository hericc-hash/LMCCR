from __future__ import annotations
import math,re
from collections import Counter
import numpy as np
from .eval_bridge import load_parser,report_metrics,character_metrics

def _binary_metrics(pred,gt,valid):
    p=np.asarray(pred,int);y=np.asarray(gt,int);v=np.asarray(valid,int)>0
    tp=int(((p==1)&(y==1)&v).sum());fp=int(((p==1)&(y==0)&v).sum());fn=int(((p==0)&(y==1)&v).sum());pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.;f1=2*pr*re/(pr+re) if pr+re else 0.
    return {'micro_f1':f1,'precision':pr,'recall':re,'tp':tp,'fp':fp,'fn':fn}
def _char_bleu4_one(pred,ref):
    h=list(str(pred));r=list(str(ref));
    if not h or not r:return 0.0
    ps=[]
    for n in range(1,5):
        hg=Counter(tuple(h[i:i+n]) for i in range(max(0,len(h)-n+1)));rg=Counter(tuple(r[i:i+n]) for i in range(max(0,len(r)-n+1)));den=sum(hg.values());match=sum(min(c,rg[g]) for g,c in hg.items());ps.append((match+1.0)/(den+1.0))
    bp=1.0 if len(h)>=len(r) else math.exp(1.0-len(r)/max(1,len(h)));return bp*math.exp(sum(.25*math.log(max(1e-12,p)) for p in ps))
def char_bleu4(preds,refs):return float(np.mean([_char_bleu4_one(p,r) for p,r in zip(preds,refs)])) if preds else 0.0
def evaluate(rows,parser_module,text_key='generated'):
    mod=load_parser(parser_module);texts=[str(r[text_key]) for r in rows];gt=np.asarray([r['slot_labels'] for r in rows],int);valid=np.asarray([r['slot_valid'] for r in rows],int);core=np.asarray([r['core_binary'] for r in rows],int);parsed,gtm=report_metrics(mod,texts,gt,valid);_,cm=report_metrics(mod,texts,core,valid);refs=[str(r.get('reference_raw') or '') for r in rows]
    lang=character_metrics(mod,texts,refs);lang=dict(lang) if isinstance(lang,dict) else {'character_metrics_raw':lang};lang['char_bleu4_standardized']=char_bleu4(texts,refs)
    return {'n':len(rows),'report_vs_gt':gtm,'report_vs_core':cm,'core_vs_gt':_binary_metrics(core,gt,valid),'language':lang}
def language_value(lang,kind):
    if kind=='bleu':
        for k in ['bleu4','bleu_4','bleu-4','BLEU-4','char_bleu4_standardized']:
            if k in lang and isinstance(lang[k],(int,float,np.generic)):return float(lang[k])
    want=['rougel','rougelf1','charrougelf1'] if kind=='rouge' else []
    found=[]
    def rec(x):
        if isinstance(x,dict):
            for k,v in x.items():
                key=str(k).lower().replace('_','').replace('-','')
                if key in want and isinstance(v,(int,float,np.generic)):found.append(float(v))
                rec(v)
    rec(lang);return found[0] if found else None
def _scaffold_parts(text):
    t=str(text or '').replace('所见：','').replace('结论：','').replace('<CORE>','')
    return [x.strip() for x in re.split(r'[，,。；;\n]+',t) if len(x.strip())>=3]

def process_summary(rows):
    if not rows:return {}
    fm=[r['finalize_meta'] for r in rows];ret=[]
    for r in rows:
        ps=_scaffold_parts(r.get('linguistic_scaffold',''));final=str(r.get('generated',''));ret.append(sum(1 for p in ps if p in final)/len(ps) if ps else 1.0)
    attempts=sum(x.get('targeted_patch_attempts',0) for x in fm);success=sum(x.get('targeted_patch_successes',0) for x in fm)
    return {
      'surgical_change_rate':float(np.mean([bool(x['changed']) for x in fm])),
      'fallback_rate':float(np.mean([bool(x.get('fallback',False)) for x in fm])),
      'final_exact_rate':float(np.mean([bool(x.get('final_exact',False)) for x in fm])),
      'mean_deleted_spans':float(np.mean([x.get('deleted_spans',0) for x in fm])),
      'mean_deleted_chars':float(np.mean([x.get('deleted_chars',0) for x in fm])),
      'mean_inserted_phrases':float(np.mean([x.get('inserted_phrases',0) for x in fm])),
      'mean_inserted_chars':float(np.mean([x.get('inserted_chars',0) for x in fm])),
      'mean_unresolved_slots':float(np.mean([x.get('unresolved_slots',0) for x in fm])),
      'mean_similarity_raw_final':float(np.mean([x.get('similarity_raw_final',1.0) for x in fm])),
      'mean_edit_char_fraction':float(np.mean([x.get('edit_char_fraction',0.0) for x in fm])),
      'targeted_patch_success_rate':float(success/attempts) if attempts else 1.0,
      'mean_scaffold_exact_chunk_retention':float(np.mean(ret))}
