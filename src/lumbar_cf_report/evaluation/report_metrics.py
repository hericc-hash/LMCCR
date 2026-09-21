#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import defaultdict
from typing import Dict,Mapping,Optional,Sequence
import re
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,average_precision_score
from .schema import LEVELS,TASKS,MAIN_SLOT_NAMES,STRUCTURE_TERMS

LEVEL_PATTERNS={
    "L1/2":[r"L1\s*[/－\-]\s*2",r"腰1\s*[/－\-]\s*2",r"腰1\s*[/－\-]\s*腰2"],
    "L2/3":[r"L2\s*[/－\-]\s*3",r"腰2\s*[/－\-]\s*3",r"腰2\s*[/－\-]\s*腰3"],
    "L3/4":[r"L3\s*[/－\-]\s*4",r"腰3\s*[/－\-]\s*4",r"腰3\s*[/－\-]\s*腰4"],
    "L4/5":[r"L4\s*[/－\-]\s*5",r"腰4\s*[/－\-]\s*5",r"腰4\s*[/－\-]\s*腰5"],
    "L5/S1":[r"L5\s*[/－\-]\s*S1",r"腰5\s*[/－\-]\s*骶1",r"腰5\s*[/－\-]\s*S1"],
}
ROOT_TO_LEVEL={"L2":"L1/2","L3":"L2/3","L4":"L3/4","L5":"L4/5","S1":"L5/S1"}
TASK_TERMS={
    "disc":["椎间盘膨出","椎间盘突出","椎间盘向","椎间盘后突","脱出","游离","椎间盘异常","椎间盘信号减低"],
    "stenosis":["椎管狭窄","椎管变窄","侧隐窝狭窄","侧隐窝变窄","椎间孔狭窄","椎间孔变窄","神经根管狭窄"],
    "nerve":["神经根受压","神经受压","马尾受压","马尾神经受压","马尾聚集","马尾神经聚集","终丝受压"],
}
NEGATION_TERMS=["未见","无明显","无","未提示","未发现","不伴","未受压"]

def _norm(x):return re.sub(r"\s+","","" if x is None else str(x).replace("；","。").replace(";","。"))
def _neg_before(text,start,window=8):return any(t in text[max(0,start-window):start] for t in NEGATION_TERMS)
def _levels(clause):return [l for l,ps in LEVEL_PATTERNS.items() if any(re.search(p,clause,flags=re.I) for p in ps)]
def _root_levels(clause):
    out=[]
    for root,level in ROOT_TO_LEVEL.items():
        if re.search(rf"(?:左侧|右侧|双侧|两侧)?{re.escape(root)}(?:及|、|，|和|与|双侧|左侧|右侧)?(?:神经根)",clause,flags=re.I):out.append(level)
    return out

def parse_focused_semantics(text)->Dict[str,Optional[int]]:
    t=_norm(text);r={n:0 for n in MAIN_SLOT_NAMES}
    if any(x in t for x in ["曲度变直","曲度稍直","生理曲度变直","生理曲度稍直","生理弯曲变直"]):r["lordosis"]=1
    elif any(x in t for x in ["曲度存在","生理曲度存在","曲度正常","生理弯曲存在"]):r["lordosis"]=0
    else:r["lordosis"]=None
    for clause in [x for x in re.split(r"[。；;\n]",t) if x]:
        levels=_levels(clause)
        for task,terms in TASK_TERMS.items():
            pos=False
            for term in terms:
                for m in re.finditer(re.escape(term),clause):
                    if not _neg_before(clause,m.start()):pos=True;break
                if pos:break
            if not pos:continue
            tl=list(levels) if levels else (_root_levels(clause) if task=="nerve" else [])
            for level in tl:r[f"{task}:{level}"]=1
    return r

def structure_polarity(text,term):
    t=_norm(text);ms=list(re.finditer(re.escape(term),t))
    if not ms:return -1
    pos=any(not _neg_before(t,m.start(),10) for m in ms);neg=any(_neg_before(t,m.start(),10) for m in ms)
    return 1 if pos else (0 if neg else -1)

def slot_metrics_from_probabilities(probs,labels,valid,min_pos=5,min_neg=5):
    p=np.asarray(probs,float);y=np.asarray(labels,int);v=np.asarray(valid)>0.5;rows=[];aucs=[];aps=[]
    for j,n in enumerate(MAIN_SLOT_NAMES):
        m=v[:,j];yt=y[m,j];pt=p[m,j];pos=int((yt==1).sum());neg=int((yt==0).sum());eligible=pos>=min_pos and neg>=min_neg;auc=None;ap=None
        if pos>0 and neg>0:auc=float(roc_auc_score(yt,pt));ap=float(average_precision_score(yt,pt))
        if eligible and auc is not None:aucs.append(auc);aps.append(ap)
        rows.append({"slot":n,"n":int(m.sum()),"positive":pos,"negative":neg,"auc":auc,"auprc":ap,"eligible":eligible})
    return {"eligible_macro_auc":None if not aucs else float(np.mean(aucs)),"eligible_macro_auprc":None if not aps else float(np.mean(aps)),"eligible_slots":len(aucs),"per_slot":rows}

def compute_report_slot_metrics(parsed,labels,valid):
    labels=np.asarray(labels).astype(int);valid=np.asarray(valid).astype(bool);pred=np.full(labels.shape,-1,dtype=int)
    for i,row in enumerate(parsed):
        for j,n in enumerate(MAIN_SLOT_NAMES):
            if row.get(n) is not None:pred[i,j]=int(row[n])
    obs=valid&(pred>=0);yt=labels[obs];yp=pred[obs]
    fp=int(((pred==1)&(labels==0)&valid).sum());fn=int(((pred==0)&(labels==1)&valid).sum());pos=int(((labels==1)&valid).sum());neg=int(((labels==0)&valid).sum())
    per={}
    for ti,t in enumerate(TASKS):
        idx=[1+ti*5+i for i in range(5)];m=obs[:,idx];a=labels[:,idx][m];b=pred[:,idx][m];per[t]={"f1":float(f1_score(a,b,zero_division=0)) if a.size else 0.,"precision":float(precision_score(a,b,zero_division=0)) if a.size else 0.,"recall":float(recall_score(a,b,zero_division=0)) if a.size else 0.,"observed_slots":int(m.sum())}
    return {"parse_rate":float(obs.sum()/max(valid.sum(),1)),"micro_f1":float(f1_score(yt,yp,zero_division=0)) if yt.size else 0.,"precision":float(precision_score(yt,yp,zero_division=0)) if yt.size else 0.,"recall":float(recall_score(yt,yp,zero_division=0)) if yt.size else 0.,"hallucination_rate":float(fp/max(neg,1)),"omission_rate":float(fn/max(pos,1)),"per_task":per,"predictions":pred}

def structure_semantic_metrics(predictions,references):
    per={};rm=pm=match=hall=pc=pt=0
    for term in STRUCTURE_TERMS:
        r=np.asarray([structure_polarity(x,term) for x in references]);p=np.asarray([structure_polarity(x,term) for x in predictions]);r0=r>=0;p0=p>=0;m=r0&p0;h=(~r0)&p0;rm+=int(r0.sum());pm+=int(p0.sum());match+=int(m.sum());hall+=int(h.sum());pc+=int((r[m]==p[m]).sum());pt+=int(m.sum());per[term]={"reference_mentions":int(r0.sum()),"generated_mentions":int(p0.sum()),"mention_recall":float(m.sum()/max(r0.sum(),1)),"hallucination_rate":float(h.sum()/max(p0.sum(),1)),"polarity_accuracy":float((r[m]==p[m]).mean()) if m.any() else None}
    return {"mention_recall":float(match/max(rm,1)),"mention_precision":float(match/max(pm,1)),"structure_hallucination_rate":float(hall/max(pm,1)),"polarity_accuracy":float(pc/max(pt,1)),"per_term":per}

def character_metrics(predictions,references):
    def lcs(a,b):
        if len(a)<len(b):a,b=b,a
        prev=[0]*(len(b)+1)
        for ca in a:
            cur=[0]
            for j,cb in enumerate(b,1):cur.append(prev[j-1]+1 if ca==cb else max(cur[-1],prev[j]))
            prev=cur
        return prev[-1]
    rr=[];ff=[]
    for p,r in zip(predictions,references):
        p=_norm(p);r=_norm(r);z=lcs(p,r);pr=z/max(len(p),1);re0=z/max(len(r),1);rr.append(2*pr*re0/max(pr+re0,1e-12));pc=defaultdict(int);rc=defaultdict(int)
        for c in p:pc[c]+=1
        for c in r:rc[c]+=1
        ov=sum(min(pc[k],rc[k]) for k in set(pc)|set(rc));a=ov/max(len(p),1);b=ov/max(len(r),1);ff.append(2*a*b/max(a+b,1e-12))
    return {"char_rouge_l_f1":float(np.mean(rr)) if rr else 0.,"char_multiset_f1":float(np.mean(ff)) if ff else 0.}
