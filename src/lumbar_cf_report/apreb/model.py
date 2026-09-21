#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hidden=256,dropout=0.1,final_norm=False):
        super().__init__()
        layers=[nn.LayerNorm(int(in_dim)),nn.Linear(int(in_dim),int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),int(out_dim))]
        if final_norm: layers.append(nn.LayerNorm(int(out_dim)))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x.float())


class AnatomyPathologyResidualBottleneck(nn.Module):
    """Shared segment anatomy A_l + task-conditioned pathology residual R_l,t.

    A_l is produced once per patient/level from the pre-context segment base token and
    patient-specific coordinate C_l. R_l,t is produced in a shared residual encoder
    conditioned by task identity. Reconstruction is additive in evidence space:

        Xhat_l,t = B_l,t(A_l, C_l, t) + Delta_l,t(R_l,t, t)

    APREB initializes from a compatible factual-anchored residual checkpoint, freezes the anatomy/background
    pathway during counterfactual training, and adapts only the residual pathway over a multi-strength trajectory.
    """
    def __init__(self,segment_dim:int,task_dim:int,coordinate_dim:int,
                 anatomy_dim:int=96,residual_dim:int=48,task_embedding_dim:int=24,
                 coordinate_latent_dim:int=48,hidden_dim:int=256,dropout:float=0.1):
        super().__init__()
        self.segment_dim=int(segment_dim); self.task_dim=int(task_dim); self.coordinate_dim=int(coordinate_dim)
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
        s=torch.as_tensor(segment_base).float(); c=torch.as_tensor(coordinate_state).float()
        ce=self.coordinate_encoder(c)
        a=self.anatomy_encoder(torch.cat([s,ce],-1))
        return a,ce

    def _task_grid(self,b,device):
        idx=torch.arange(3,device=device).view(1,1,3).expand(b,5,3)
        return idx,self.task_embedding(idx)

    def anatomy_component(self,anatomy,coord_latent):
        b=anatomy.shape[0]; idx,te=self._task_grid(b,anatomy.device)
        aa=anatomy[:,:,None,:].expand(-1,-1,3,-1)
        cc=coord_latent[:,:,None,:].expand(-1,-1,3,-1)
        base=self.anatomy_evidence_decoder(torch.cat([aa,cc,te],-1))
        return base,idx,te

    def encode_residual(self,task_features,anatomy,coord_latent,base,task_emb):
        x=torch.as_tensor(task_features).float()
        aa=anatomy[:,:,None,:].expand(-1,-1,3,-1)
        cc=coord_latent[:,:,None,:].expand(-1,-1,3,-1)
        delta=x-base
        r=self.residual_encoder(torch.cat([x,delta,aa,cc,task_emb],-1))
        return r

    def decode_from_latents(self,anatomy,residual,coordinate_state):
        ce=self.coordinate_encoder(coordinate_state)
        base,idx,te=self.anatomy_component(anatomy,ce)
        residual_component=self.residual_evidence_decoder(torch.cat([residual,te],-1))
        return base+residual_component,base,residual_component

    def decode_single_slot(self,anatomy_level,residual_slot,coordinate_level,task_index:int):
        # anatomy_level [B,A], residual_slot [B,R], coordinate_level [B,C]
        ce=self.coordinate_encoder(coordinate_level)
        tidx=torch.full((anatomy_level.shape[0],),int(task_index),device=anatomy_level.device,dtype=torch.long)
        te=self.task_embedding(tidx)
        base=self.anatomy_evidence_decoder(torch.cat([anatomy_level,ce,te],-1))
        res=self.residual_evidence_decoder(torch.cat([residual_slot,te],-1))
        return base+res

    def forward(self,segment_base,task_features,coordinate_state) -> Dict[str,torch.Tensor]:
        a,ce=self.encode_anatomy(segment_base,coordinate_state)
        base,idx,te=self.anatomy_component(a,ce)
        r=self.encode_residual(task_features,a,ce,base,te)
        residual_component=self.residual_evidence_decoder(torch.cat([r,te],-1))
        xhat=base+residual_component
        logits=self.pathology_head(torch.cat([r,te],-1)).squeeze(-1)
        seg_hat=self.segment_decoder(a)
        coord_hat=self.coordinate_decoder(a)
        return {
            "anatomy":a,"residual":r,"coordinate_latent":ce,
            "anatomy_component":base,"residual_component":residual_component,
            "reconstructed_task_features":xhat,"pathology_logits":logits,
            "pathology_probabilities":torch.sigmoid(logits),
            "reconstructed_segment_base":seg_hat,"reconstructed_coordinate":coord_hat,
        }

