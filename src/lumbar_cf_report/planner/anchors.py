from __future__ import annotations
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from .metrics import best_f1_threshold,task_metrics

def fit_task(E,y,v,cfg):
    m=v.reshape(-1)>.5;X=E.reshape(-1,E.shape[-1])[m];yy=y.reshape(-1)[m].astype(int)
    lr=LogisticRegression(C=float(cfg['C']),class_weight=cfg['class_weight'],solver=cfg['solver'],max_iter=int(cfg['max_iter']))
    lr.fit(X,yy);return lr

def logits_all(models,E):
    arr=[]
    for ti,m in enumerate(models):arr.append(m.decision_function(E[:,:,ti,:].reshape(-1,E.shape[-1])).reshape(E.shape[0],E.shape[1]))
    return np.stack(arr,axis=2)

def params(models):
    w=np.stack([m.coef_[0] for m in models]);b=np.array([m.intercept_[0] for m in models]);return torch.tensor(w).float(),torch.tensor(b).float()
