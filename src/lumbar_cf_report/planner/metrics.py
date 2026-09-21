from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,brier_score_loss

def best_f1_threshold(y,p):
    y=np.asarray(y).astype(int);p=np.asarray(p,float)
    if len(np.unique(y))<2:return .5
    cand=np.unique(np.concatenate(([0.,.5,1.],p)))
    best=( -1,.5)
    for th in cand:
        z=(p>=th).astype(int);f=f1_score(y,z,zero_division=0)
        if f>best[0]+1e-12:best=(f,float(th))
    return best[1]

def binary_metrics(y,p,th=.5):
    y=np.asarray(y).astype(int);p=np.asarray(p,float);z=(p>=th).astype(int)
    auc=float('nan') if len(np.unique(y))<2 else float(roc_auc_score(y,p))
    ap=float('nan') if y.sum()==0 else float(average_precision_score(y,p))
    return {'n':int(len(y)),'positives':int(y.sum()),'prevalence':float(y.mean()) if len(y) else float('nan'),'threshold':float(th),'auroc':auc,'auprc':ap,'f1':float(f1_score(y,z,zero_division=0)),'precision':float(precision_score(y,z,zero_division=0)),'recall':float(recall_score(y,z,zero_division=0)),'brier':float(brier_score_loss(y,p)) if len(y) else float('nan')}

def task_metrics(probs,labels,valid,thresholds):
    out={}
    for ti,t in enumerate(('disc','stenosis','nerve')):
        m=np.asarray(valid[:,:,ti])>.5;y=np.asarray(labels[:,:,ti])[m];p=np.asarray(probs[:,:,ti])[m]
        out[t]=binary_metrics(y,p,float(thresholds[ti]))
    return out

def mean_task_metric(m,key,tasks=('disc','stenosis','nerve')):
    vals=[m[t][key] for t in tasks if np.isfinite(m[t][key])]
    return float(np.mean(vals)) if vals else float('nan')
