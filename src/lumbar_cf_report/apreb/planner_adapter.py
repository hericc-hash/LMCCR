#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn


def safe_torch_load(path:str):
    try:return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:return torch.load(path,map_location="cpu")


class SourceProjector(nn.Module):
    def __init__(self,input_dim,hidden_size,intermediate_dim=1024,dropout=0.1):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,intermediate_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(intermediate_dim,hidden_size),nn.LayerNorm(hidden_size))
    def forward(self,x):return self.net(x.float())


class ZeroInitBoundedMLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,max_scale):
        super().__init__();self.max_scale=float(max_scale);self.scale_logit=nn.Parameter(torch.tensor(-2.0));self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,output_dim))
    def forward(self,x):
        scale=torch.sigmoid(self.scale_logit)*self.max_scale
        return scale*torch.tanh(self.net(x.float())),scale


class FrozenClinicalPlannerAdapter(nn.Module):
    """Self-contained inference-compatible replica of the the fixed clinical planner fixed planner core.

    The old bottleneck is intentionally omitted: its reconstructed task feature equals
    its input by construction. APREB therefore feeds original or A/R-reconstructed
    task evidence directly into the same frozen semantic projectors and slot head.
    """
    def __init__(self,global_dim,task_dim,coordinate_dim,hidden_size=4096,projector_hidden_dim=1024,dropout=0.1):
        super().__init__();self.global_dim=int(global_dim);self.task_dim=int(task_dim);self.coordinate_dim=int(coordinate_dim);self.hidden_size=int(hidden_size)
        self.global_projector=SourceProjector(global_dim,hidden_size,projector_hidden_dim,dropout)
        self.disease_projector=SourceProjector(task_dim,hidden_size,projector_hidden_dim,dropout)
        self.slot_embeddings=nn.Parameter(torch.zeros(1,16,hidden_size))
        self.coordinate_conditioner=ZeroInitBoundedMLP(coordinate_dim,hidden_size,256,0.10)
        self.quality_conditioner=ZeroInitBoundedMLP(2,hidden_size,128,0.08)
        self.slot_head=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,1))
        self.slot_bias=nn.Parameter(torch.zeros(16))
    def forward(self,global_source,task_features,coordinate_state,task_quality,task_valid):
        x=task_features.float();b=x.shape[0]
        global_tok=self.global_projector(global_source)[:,None]
        disease_tm=x.permute(0,2,1,3).reshape(b,15,self.task_dim)
        disease_tok=self.disease_projector(disease_tm)
        tokens=torch.cat([global_tok,disease_tok],1)+self.slot_embeddings
        coord_tm=coordinate_state[:,None].expand(-1,3,-1,-1).reshape(b,15,self.coordinate_dim)
        coord_delta,_=self.coordinate_conditioner(coord_tm)
        global_coord_delta,_=self.coordinate_conditioner(coordinate_state.mean(1,keepdim=True))
        q_tm=task_quality.permute(0,2,1).reshape(b,15,1);v_tm=task_valid.permute(0,2,1).reshape(b,15,1)
        q_delta,_=self.quality_conditioner(torch.cat([q_tm,v_tm],-1));gq=torch.ones(b,1,2,device=x.device,dtype=x.dtype);gqd,_=self.quality_conditioner(gq)
        tokens=tokens+torch.cat([global_coord_delta,coord_delta],1)+torch.cat([gqd,q_delta],1)
        logits=self.slot_head(tokens).squeeze(-1)+self.slot_bias[None]
        return {"slot_logits":logits,"slot_probabilities":torch.sigmoid(logits),"planner_tokens":tokens}


def load_frozen_planner(path:str):
    obj=safe_torch_load(path);cfg=obj.get("config",{});state=obj.get("planner_state_dict",obj)
    model=FrozenClinicalPlannerAdapter(int(cfg["global_dim"]),int(cfg["task_dim"]),int(cfg["coordinate_dim"]),int(cfg.get("hidden_size",4096)),int(cfg.get("projector_hidden_dim",1024)),float(cfg.get("dropout",0.1)))
    allowed=("global_projector.","disease_projector.","slot_embeddings","coordinate_conditioner.","quality_conditioner.","slot_head.","slot_bias")
    filtered={k:v for k,v in state.items() if any(k==p or k.startswith(p) for p in allowed)}
    missing,unexpected=model.load_state_dict(filtered,strict=False)
    required=["global_projector.net.0.weight","disease_projector.net.0.weight","slot_embeddings","slot_head.1.weight","slot_head.4.weight","slot_bias"]
    absent=[k for k in required if k not in filtered]
    if absent: raise RuntimeError(f"planner checkpoint missing required keys: {absent}")
    # Missing keys are not expected because only old bottleneck keys were filtered out.
    if missing: raise RuntimeError(f"planner adapter missing loaded keys: {missing[:20]}")
    for p in model.parameters():p.requires_grad=False
    model.eval();return model,obj

