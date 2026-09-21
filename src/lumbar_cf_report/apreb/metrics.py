#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from .schema import LEVELS,TASKS,SLOT_NAMES


def safe_auc(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if len(np.unique(y))<2:return None
    return float(roc_auc_score(y,p))

def safe_ap(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if len(np.unique(y))<2:return None
    return float(average_precision_score(y,p))

def per_slot_metrics(labels,probs,valid,min_positive=5,min_negative=5):
    y=torch.as_tensor(labels).cpu().numpy();p=torch.as_tensor(probs).cpu().numpy();v=torch.as_tensor(valid).cpu().numpy()>0.5
    rows=[];aucs=[];aps=[]
    for ti,task in enumerate(TASKS):
        for li,level in enumerate(LEVELS):
            m=v[:,li,ti];yy=y[m,li,ti];pp=p[m,li,ti];pos=int((yy>0.5).sum());neg=int((yy<=0.5).sum());eligible=pos>=min_positive and neg>=min_negative
            auc=safe_auc(yy,pp);ap=safe_ap(yy,pp)
            row={"slot":f"{task}:{level}","task":task,"level":level,"n":int(m.sum()),"positive":pos,"negative":neg,"auc":auc,"auprc":ap,"eligible":eligible};rows.append(row)
            if eligible and auc is not None:aucs.append(auc)
            if eligible and ap is not None:aps.append(ap)
    return rows,{"eligible_macro_auc":float(np.mean(aucs)) if aucs else None,"eligible_macro_auprc":float(np.mean(aps)) if aps else None,"eligible_slots":len(aucs)}

def cosine_mean(a,b):
    a=torch.as_tensor(a).float();b=torch.as_tensor(b).float()
    return float(torch.nn.functional.cosine_similarity(a,b,dim=-1).mean().item())

