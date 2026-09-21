from __future__ import annotations
import math
from collections import Counter
import numpy as np
from .eval_bridge import report_metrics,character_metrics,parse,normalize

def bleu_one(pred,ref):
    h=list(str(pred));r=list(str(ref))
    if not h or not r:return 0.0
    ps=[]
    for n in range(1,5):
        hg=Counter(tuple(h[i:i+n]) for i in range(max(0,len(h)-n+1)));rg=Counter(tuple(r[i:i+n]) for i in range(max(0,len(r)-n+1)));den=sum(hg.values());m=sum(min(c,rg[g]) for g,c in hg.items());ps.append((m+1.0)/(den+1.0))
    bp=1.0 if len(h)>=len(r) else math.exp(1.0-len(r)/max(1,len(h)))
    return bp*math.exp(sum(.25*math.log(max(1e-12,p)) for p in ps))
def language_value(x,kind):
    if kind=='bleu':
        for k in ['bleu4','bleu_4','bleu-4','BLEU-4','char_bleu4_standardized']:
            if isinstance(x,dict) and k in x and isinstance(x[k],(int,float,np.generic)):return float(x[k])
    want={'rougel','rougelf1','charrougelf1'} if kind=='rouge' else set();found=[]
    def rec(v):
        if isinstance(v,dict):
            for k,z in v.items():
                kk=str(k).lower().replace('_','').replace('-','')
                if kk in want and isinstance(z,(int,float,np.generic)):found.append(float(z))
                rec(z)
    rec(x);return found[0] if found else None
def rouge_one(mod,pred,ref):return language_value(character_metrics(mod,[str(pred)],[str(ref)]),'rouge') or 0.0
def parsed_vec(mod,text):return (np.asarray(normalize(parse(mod,text)),float).reshape(-1)[:16]>.5).astype(int)
def case_counts(pred,gt,valid):
    p=np.asarray(pred,int);y=np.asarray(gt,int);v=np.asarray(valid,int)>0
    tp=int(((p==1)&(y==1)&v).sum());fp=int(((p==1)&(y==0)&v).sum());fn=int(((p==0)&(y==1)&v).sum());tn=int(((p==0)&(y==0)&v).sum());return tp,fp,fn,tn
def case_f1(pred,gt,valid):
    tp,fp,fn,_=case_counts(pred,gt,valid);d=2*tp+fp+fn
    return 1.0 if d==0 else 2*tp/d
def case_acc(pred,gt,valid):
    p=np.asarray(pred,int);y=np.asarray(gt,int);v=np.asarray(valid,int)>0;return float((p[v]==y[v]).mean()) if v.any() else 1.0
def binary_f1(pred,gt,valid):
    tp=fp=gp=0
    for p,y,v in zip(pred,gt,valid):
        if not v:continue
        tp+=int(p and y);fp+=int(p and not y);gp+=int(y)
    d=tp+fp+gp;return 2*tp/d if d else 0.0
def eval_texts(mod,texts,refs,gt,valid,core):
    _,gm=report_metrics(mod,texts,np.asarray(gt,float),np.asarray(valid,float));_,cm=report_metrics(mod,texts,np.asarray(core,float),np.asarray(valid,float));ch=character_metrics(mod,texts,refs);r=language_value(ch,'rouge');b=float(np.mean([bleu_one(a,z) for a,z in zip(texts,refs)]))
    return {'clinical_f1':float(gm['micro_f1']),'fidelity_f1':float(cm['micro_f1']),'rouge_l':float(r),'bleu4':b}
