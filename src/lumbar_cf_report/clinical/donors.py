#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,List,Tuple
import torch

@dataclass(frozen=True)
class DonorCandidate:
    index:int; coordinate_distance:float; quality_distance:float; score:float

def coordinate_distance(a,b):
    a=torch.as_tensor(a).float(); b=torch.as_tensor(b).float();
    if b.ndim==1:b=b.unsqueeze(0)
    return torch.sqrt(((b-a.unsqueeze(0))**2).mean(-1).clamp_min(0))

def build_external_coordinate_candidate_cache(recipient_dataset,donor_dataset,topk:int=5,quality_weight:float=0.25):
    cache:Dict[Tuple[int,int,int],List[DonorCandidate]]={}
    for ri in range(len(recipient_dataset)):
        for li in range(5):
            for ti in range(3):
                if float(recipient_dataset.label_valid[ri,li,ti])<=.5: continue
                target=float(recipient_dataset.labels[ri,li,ti]); mask=(donor_dataset.label_valid[:,li,ti]>.5)&(donor_dataset.labels[:,li,ti]!=target); idx=torch.where(mask)[0]
                if idx.numel()==0:continue
                cd=coordinate_distance(recipient_dataset.coordinate_state[ri,li],donor_dataset.coordinate_state[idx,li]); qd=(donor_dataset.task_quality[idx,li,ti]-recipient_dataset.task_quality[ri,li,ti]).abs(); score=cd+float(quality_weight)*qd
                order=torch.argsort(score)[:min(int(topk),int(idx.numel()))]; rows=[]
                for oi in order.tolist():
                    di=int(idx[oi]); rows.append(DonorCandidate(di,float(cd[oi]),float(qd[oi]),float(score[oi])))
                cache[(ri,li,ti)]=rows
    return cache

def build_train_coordinate_candidate_cache(dataset,topk:int=5,quality_weight:float=0.25):
    cache={}
    for ri in range(len(dataset)):
        for li in range(5):
            for ti in range(3):
                if float(dataset.label_valid[ri,li,ti])<=.5:continue
                target=float(dataset.labels[ri,li,ti]); mask=(dataset.label_valid[:,li,ti]>.5)&(dataset.labels[:,li,ti]!=target); mask[ri]=False; idx=torch.where(mask)[0]
                if idx.numel()==0:continue
                cd=coordinate_distance(dataset.coordinate_state[ri,li],dataset.coordinate_state[idx,li]); qd=(dataset.task_quality[idx,li,ti]-dataset.task_quality[ri,li,ti]).abs(); score=cd+float(quality_weight)*qd; order=torch.argsort(score)[:min(int(topk),int(idx.numel()))]
                cache[(ri,li,ti)]=[DonorCandidate(int(idx[o]),float(cd[o]),float(qd[o]),float(score[o])) for o in order.tolist()]
    return cache

