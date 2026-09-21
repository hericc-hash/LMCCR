from __future__ import annotations
import numpy as np, torch
import torch.nn.functional as F
from .metrics import task_metrics,mean_task_metric

def resolve_device(device):
    req=torch.device(device)
    if req.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError(f'CUDA requested but unavailable: {device}')
        if req.index is None: req=torch.device('cuda',torch.cuda.current_device())
    return req

def ensure_model_device(model,device):
    dev=resolve_device(device); model.to(dev)
    bad=[n for n,p in model.named_parameters() if p.device!=dev]
    badb=[n for n,b in model.named_buffers() if b.device!=dev]
    if bad or badb: raise RuntimeError(f'device contract failed: expected={dev}, bad_params={bad[:8]}, bad_buffers={badb[:8]}')
    return dev

def masked_bce(logits,y,v,pos_weight=None):
    loss=F.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=pos_weight)
    return (loss*v).sum()/v.sum().clamp_min(1.)

def pos_weight(y,v):
    pos=(y*v).sum((0,1));neg=((1-y)*v).sum((0,1));return (neg/pos.clamp_min(1.)).clamp(.5,6.)

def forward_idx(model,d,idx,device,E_override=None):
    ii=torch.as_tensor(idx,dtype=torch.long)
    E=d['raw_E'][ii] if E_override is None else E_override
    return model(d['global_source'][ii].to(device),E.to(device),d['coordinate_state'][ii].to(device),d['task_quality'][ii].to(device),d['task_valid'][ii].to(device),False)

def evaluate(model,d,idx,device,thresholds):
    device=ensure_model_device(model,device);model.eval();ps=[]
    with torch.inference_mode():
        for st in range(0,len(idx),128):
            jj=np.asarray(idx)[st:st+128];ps.append(forward_idx(model,d,jj,device)['disease_probabilities'].cpu())
    p=torch.cat(ps).numpy();ii=np.asarray(idx);v=(d['label_valid'][ii]*d['task_valid'][ii]).numpy();y=d['labels'][ii].numpy()
    return task_metrics(p,y,v,thresholds),p

def preservation_pass(cand,base,g):
    for t in ('disc','stenosis','nerve'):
        if cand[t]['auprc'] < base[t]['auprc']-g['max_auprc_drop_per_task']:return False
        if cand[t]['auroc'] < base[t]['auroc']-g['max_auc_drop_per_task']:return False
        if cand[t]['f1'] < base[t]['f1']-g['max_f1_drop_per_task']:return False
    return mean_task_metric(cand,'auprc',('stenosis','nerve')) >= mean_task_metric(base,'auprc',('stenosis','nerve'))-g['max_hard_task_mean_auprc_drop']
