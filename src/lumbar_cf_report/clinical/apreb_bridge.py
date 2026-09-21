#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import hashlib
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hidden=256,dropout=0.1,final_norm=False):
        super().__init__()
        layers=[nn.LayerNorm(int(in_dim)),nn.Linear(int(in_dim),int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),int(out_dim))]
        if final_norm: layers.append(nn.LayerNorm(int(out_dim)))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x.float())

class FrozenAPREBBridge(nn.Module):
    """Self-contained replica of the finalized APREB A/R tensor contract.
    It loads APREB weights but imports no older-version Python modules.
    During clinical-planner training every parameter is frozen, preserving intervention behavior by construction.
    """
    def __init__(self,segment_dim:int,task_dim:int,coordinate_dim:int,anatomy_dim:int=96,residual_dim:int=48,
                 task_embedding_dim:int=24,coordinate_latent_dim:int=48,hidden_dim:int=256,dropout:float=0.1):
        super().__init__(); self.segment_dim=int(segment_dim); self.task_dim=int(task_dim); self.coordinate_dim=int(coordinate_dim)
        self.anatomy_dim=int(anatomy_dim); self.residual_dim=int(residual_dim); self.task_embedding_dim=int(task_embedding_dim)
        self.coordinate_encoder=MLP(coordinate_dim,coordinate_latent_dim,hidden_dim,dropout,True)
        self.anatomy_encoder=MLP(segment_dim+coordinate_latent_dim,anatomy_dim,hidden_dim,dropout,True)
        self.task_embedding=nn.Embedding(3,task_embedding_dim)
        self.anatomy_evidence_decoder=MLP(anatomy_dim+coordinate_latent_dim+task_embedding_dim,task_dim,hidden_dim,dropout,False)
        self.residual_encoder=MLP(task_dim*2+anatomy_dim+coordinate_latent_dim+task_embedding_dim,residual_dim,hidden_dim,dropout,True)
        self.residual_evidence_decoder=MLP(residual_dim+task_embedding_dim,task_dim,hidden_dim,dropout,False)
        self.pathology_head=MLP(residual_dim+task_embedding_dim,1,max(64,hidden_dim//2),dropout,False)
        self.segment_decoder=MLP(anatomy_dim,segment_dim,hidden_dim,dropout,False)
        self.coordinate_decoder=MLP(anatomy_dim,coordinate_dim,hidden_dim,dropout,False)
    def encode_anatomy(self,segment_base,coordinate_state):
        s=torch.as_tensor(segment_base).float(); c=torch.as_tensor(coordinate_state).float(); ce=self.coordinate_encoder(c); a=self.anatomy_encoder(torch.cat([s,ce],-1)); return a,ce
    def _task_grid(self,b,device):
        idx=torch.arange(3,device=device).view(1,1,3).expand(b,5,3); return idx,self.task_embedding(idx)
    def anatomy_component(self,anatomy,coord_latent):
        b=anatomy.shape[0]; idx,te=self._task_grid(b,anatomy.device); aa=anatomy[:,:,None,:].expand(-1,-1,3,-1); cc=coord_latent[:,:,None,:].expand(-1,-1,3,-1)
        base=self.anatomy_evidence_decoder(torch.cat([aa,cc,te],-1)); return base,idx,te
    def encode_residual(self,task_features,anatomy,coord_latent,base,task_emb):
        x=torch.as_tensor(task_features).float(); aa=anatomy[:,:,None,:].expand(-1,-1,3,-1); cc=coord_latent[:,:,None,:].expand(-1,-1,3,-1); delta=x-base
        return self.residual_encoder(torch.cat([x,delta,aa,cc,task_emb],-1))
    def decode_single_slot(self,anatomy_level,residual_slot,coordinate_level,task_index:int):
        ce=self.coordinate_encoder(coordinate_level); tidx=torch.full((anatomy_level.shape[0],),int(task_index),device=anatomy_level.device,dtype=torch.long); te=self.task_embedding(tidx)
        base=self.anatomy_evidence_decoder(torch.cat([anatomy_level,ce,te],-1)); res=self.residual_evidence_decoder(torch.cat([residual_slot,te],-1)); return base+res
    def forward(self,segment_base,task_features,coordinate_state):
        a,ce=self.encode_anatomy(segment_base,coordinate_state); base,idx,te=self.anatomy_component(a,ce); r=self.encode_residual(task_features,a,ce,base,te)
        rc=self.residual_evidence_decoder(torch.cat([r,te],-1)); xhat=base+rc; logits=self.pathology_head(torch.cat([r,te],-1)).squeeze(-1)
        return {"anatomy":a,"residual":r,"coordinate_latent":ce,"anatomy_component":base,"residual_component":rc,"reconstructed_task_features":xhat,
                "pathology_logits":logits,"pathology_probabilities":torch.sigmoid(logits),"reconstructed_segment_base":self.segment_decoder(a),"reconstructed_coordinate":self.coordinate_decoder(a)}

def safe_torch_load(path:str):
    try:return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:return torch.load(path,map_location="cpu")

def load_frozen_visual_bottleneck(path:str):
    obj=safe_torch_load(path)
    if "config" not in obj or "state_dict" not in obj: raise RuntimeError(f"invalid APREB checkpoint: {path}")
    c=obj["config"]
    model=FrozenAPREBBridge(c["segment_dim"],c["task_dim"],c["coordinate_dim"],c["anatomy_dim"],c["residual_dim"],c["task_embedding_dim"],c["coordinate_latent_dim"],c["hidden_dim"],c["dropout"])
    model.load_state_dict(obj["state_dict"],strict=True); model.eval()
    for p in model.parameters(): p.requires_grad=False
    return model,obj

def state_dict_sha256(model:nn.Module)->str:
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode("utf-8")); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

