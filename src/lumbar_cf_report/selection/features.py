from __future__ import annotations
import difflib
from collections import Counter
import numpy as np

TASK_SLICES={'lordosis':slice(0,1),'disc':slice(1,6),'stenosis':slice(6,11),'nerve':slice(11,16)}
TAGS=['greedy','greedy_final','sample0','sample1','sample2','sample3']


def _safe_scaffold(s):
    return str(s).replace('<CORE>','').replace('所见：','').replace('结论：','')

def _ngrams(s,n):
    x=list(str(s));return Counter(tuple(x[i:i+n]) for i in range(max(0,len(x)-n+1)))

def _overlap(a,b,n):
    A=_ngrams(a,n);B=_ngrams(b,n);m=sum(min(c,B[g]) for g,c in A.items())
    pa=m/max(1,sum(A.values()));rb=m/max(1,sum(B.values()));return pa,rb

def _rep_ratio(text):
    x=list(str(text));g=[tuple(x[i:i+2]) for i in range(max(0,len(x)-1))]
    return len(set(g))/max(1,len(g))

def _core_f1(p,c,v):
    p=np.asarray(p,int);c=np.asarray(c,int);v=np.asarray(v,int)>0
    tp=int(((p==1)&(c==1)&v).sum());fp=int(((p==1)&(c==0)&v).sum());gp=int(((c==1)&v).sum());d=tp+fp+gp
    return 2*tp/d if d else 0.0

def extract(row,cand):
    text=str(cand['text']);pv=np.asarray(cand['parser_vec'],int);core=np.asarray(row['core_binary'],int);valid=np.asarray(row['slot_valid'],int)>0
    probs=np.asarray(row['planner_probs'],float);ths=np.asarray(row['planner_thresholds'],float);direct=np.asarray(row.get('direct_parser_vec',[0]*16),int)
    denom=np.where(probs>=ths,np.maximum(1e-6,1-ths),np.maximum(1e-6,ths));conf=np.clip(np.abs(probs-ths)/denom,0,1)
    mis=(pv!=core)&valid;safe=_safe_scaffold(row['linguistic_scaffold']);direct_text=str(row.get('direct_draft',''))
    sim=difflib.SequenceMatcher(None,text,safe).ratio() if text or safe else 1.0
    direct_sim=difflib.SequenceMatcher(None,text,direct_text).ratio() if text or direct_text else 1.0
    feat=[];names=[]
    def add(n,x):names.append(n);feat.append(float(x))
    add('bias',1);add('text_len',min(len(text),1000)/300);add('safe_len',min(len(safe),1000)/300)
    add('len_ratio_to_safe',min(len(text)/max(1,len(safe)),4)/4);add('seqsim_safe',sim)
    add('direct_text_len',min(len(direct_text),1000)/300);add('len_ratio_to_direct',min(len(text)/max(1,len(direct_text)),4)/4);add('seqsim_direct',direct_sim)
    add('repeat_unique_bigram_ratio',_rep_ratio(text));add('has_findings',int('所见：' in text));add('has_conclusion',int('结论：' in text));add('newline_count',min(text.count('\n'),4)/4)
    add('punct_density',sum(text.count(x) for x in '，。；,;')/max(1,len(text)));add('core_fidelity_f1',_core_f1(pv,core,valid));add('mismatch_frac',mis.sum()/max(1,valid.sum()))
    add('mismatch_conf_sum',float(conf[mis].sum())/max(1,valid.sum()));add('mismatch_conf_mean',float(conf[mis].mean()) if mis.any() else 0)
    add('high_conf_mismatch_frac',((mis)&(conf>=.5)).sum()/max(1,valid.sum()));add('low_conf_mismatch_frac',((mis)&(conf<.25)).sum()/max(1,valid.sum()))
    add('candidate_pos_frac',((pv==1)&valid).sum()/max(1,valid.sum()));add('planner_pos_frac',((core==1)&valid).sum()/max(1,valid.sum()))
    add('candidate_direct_agree_frac',((pv==direct)&valid).sum()/max(1,valid.sum()));add('planner_direct_disagree_frac',((core!=direct)&valid).sum()/max(1,valid.sum()))
    for n in range(1,5):
        p,r=_overlap(text,safe,n);add(f'safe_ngram{n}_precision',p);add(f'safe_ngram{n}_recall',r)
        dp,dr=_overlap(text,direct_text,n);add(f'direct_ngram{n}_precision',dp);add(f'direct_ngram{n}_recall',dr)
    for task,sl in TASK_SLICES.items():
        vv=valid[sl];mm=mis[sl];cc=conf[sl];pp=pv[sl];co=core[sl];dd=direct[sl];d=max(1,vv.sum())
        add(f'{task}_mismatch_frac',mm.sum()/d);add(f'{task}_mismatch_conf_sum',float(cc[mm].sum())/d)
        add(f'{task}_candidate_pos_frac',((pp==1)&vv).sum()/d);add(f'{task}_planner_pos_frac',((co==1)&vv).sum()/d);add(f'{task}_direct_agree_frac',((pp==dd)&vv).sum()/d)
    for i in range(16):
        add(f's{i}_candidate',pv[i]);add(f's{i}_planner',core[i]);add(f's{i}_prob',probs[i]);add(f's{i}_conf',conf[i])
        add(f's{i}_mismatch',int(valid[i] and pv[i]!=core[i]));add(f's{i}_direct',direct[i]);add(f's{i}_cand_eq_direct',int(valid[i] and pv[i]==direct[i]))
    am=row.get('adaptive_meta',{}) or {}
    add('adaptive_difficulty_score',am.get('score',0));add('adaptive_expand_flag',int(bool(row.get('expanded',False))))
    add('adaptive_best_core_mismatch',am.get('best_core_mismatch',0));add('adaptive_best_high_conf_mismatch',am.get('best_high_conf_mismatch',0))
    add('adaptive_sn_high_conf_mismatch',am.get('best_stenosis_nerve_high_conf_mismatch',0));add('adaptive_raw_final_parser_disagreement',am.get('raw_final_parser_disagreement',0))
    add('adaptive_planner_direct_high_conf_disagreement',am.get('planner_direct_high_conf_disagreement',0));add('adaptive_text_pathology',am.get('text_pathology',0))
    tag=str(cand.get('tag',''))
    for t in TAGS:add('tag_'+t,int(tag==t))
    return np.asarray(feat,dtype=np.float32),names

# Language-centric fields for within-case relative context. These are all deployment-safe.
_RELATIVE_BASE_NAMES=[
    'text_len','len_ratio_to_safe','seqsim_safe','len_ratio_to_direct','seqsim_direct','repeat_unique_bigram_ratio','punct_density',
    'safe_ngram1_precision','safe_ngram1_recall','safe_ngram2_precision','safe_ngram2_recall','safe_ngram3_precision','safe_ngram3_recall','safe_ngram4_precision','safe_ngram4_recall',
    'direct_ngram1_precision','direct_ngram1_recall','direct_ngram2_precision','direct_ngram2_recall','direct_ngram3_precision','direct_ngram3_recall','direct_ngram4_precision','direct_ngram4_recall',
    'core_fidelity_f1','mismatch_frac'
]

def _novel_bigram_fraction(a,b):
    A=set(_ngrams(a,2));B=set(_ngrams(b,2))
    return len(A-B)/max(1,len(A))

def extract_case(row):
    """Return case-contextual candidate feature matrix and names.

    It starts with the v2 deployment-safe feature vector and appends only
    candidate-vs-peer / candidate-vs-greedy/final relative signals. No reference
    or GT-derived quantity is read here.
    """
    base=[];names=None
    for cand in row['candidates']:
        x,n=extract(row,cand)
        if names is None:names=n
        elif n!=names:raise RuntimeError('Base feature schema drift inside case')
        base.append(x)
    X=np.asarray(base,np.float32)
    idx={n:i for i,n in enumerate(names)}
    extra=[];extra_names=[]
    texts=[str(c['text']) for c in row['candidates']]
    tags=[str(c.get('tag','')) for c in row['candidates']]
    greedy=texts[tags.index('greedy')] if 'greedy' in tags else texts[0]
    final=texts[tags.index('greedy_final')] if 'greedy_final' in tags else greedy
    peer_sims=[]
    for i,t in enumerate(texts):
        ss=[difflib.SequenceMatcher(None,t,u).ratio() for j,u in enumerate(texts) if j!=i]
        peer_sims.append((float(np.mean(ss)) if ss else 1.0,float(np.min(ss)) if ss else 1.0))
    peer=np.zeros((len(texts),6),np.float32)
    for i,t in enumerate(texts):
        peer[i]=[
            peer_sims[i][0],peer_sims[i][1],
            difflib.SequenceMatcher(None,t,greedy).ratio(),
            difflib.SequenceMatcher(None,t,final).ratio(),
            np.clip((len(t)-len(greedy))/max(20,len(greedy)),-2,2)/2,
            _novel_bigram_fraction(t,greedy)
        ]
    peer_names=['peer_seqsim_mean','peer_seqsim_min','seqsim_to_greedy','seqsim_to_greedy_final','length_delta_to_greedy','bigram_novelty_vs_greedy']
    extra.append(peer);extra_names.extend(peer_names)
    rel_blocks=[];rel_names=[]
    for n in _RELATIVE_BASE_NAMES:
        j=idx[n];v=X[:,j].astype(np.float32);mean=float(v.mean());mx=float(v.max())
        rank=np.asarray([(float((v<x).sum())+.5*float((v==x).sum()-1))/max(1,len(v)-1) for x in v],dtype=np.float32)
        rel_blocks.append(np.stack([v-mean,v-mx,rank],axis=1))
        rel_names.extend([f'rel_{n}_minus_mean',f'rel_{n}_minus_max',f'rel_{n}_rank'])
    if rel_blocks:
        extra.append(np.concatenate(rel_blocks,axis=1));extra_names.extend(rel_names)
    E=np.concatenate(extra,axis=1) if extra else np.zeros((len(texts),0),np.float32)
    return np.concatenate([X,E],axis=1).astype(np.float32),names+extra_names


def language_proxy_from_case_features(X,names):
    """Deterministic deployment-safe lexical retention proxy used only as a regularizer."""
    idx={n:i for i,n in enumerate(names)}
    def col(n):return X[:,idx[n]] if n in idx else np.zeros(len(X),np.float32)
    direct_ng=np.mean(np.stack([col(f'direct_ngram{k}_precision') for k in [2,3,4]]+[col(f'direct_ngram{k}_recall') for k in [2,3,4]],axis=1),axis=1)
    safe_ng=np.mean(np.stack([col(f'safe_ngram{k}_precision') for k in [2,3,4]]+[col(f'safe_ngram{k}_recall') for k in [2,3,4]],axis=1),axis=1)
    lenclose=1.-np.minimum(np.abs(col('len_ratio_to_direct')-.25)/.75,1.)
    raw=.30*col('seqsim_direct')+.12*col('seqsim_safe')+.28*direct_ng+.12*safe_ng+.10*col('repeat_unique_bigram_ratio')+.08*lenclose
    if len(raw)<=1:return np.asarray(raw,np.float32)
    lo,hi=float(raw.min()),float(raw.max())
    return ((raw-lo)/(hi-lo+1e-8)).astype(np.float32)
