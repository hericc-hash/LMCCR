#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .schema import MAIN_SLOT_NAMES,STRUCTURE_TERMS,PLAN_NODE_NAMES,build_structure_relation_mask

class SourceProjector(nn.Module):
    def __init__(self,input_dim:int,hidden_size:int,intermediate_dim:int=1024,dropout:float=.1):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,intermediate_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(intermediate_dim,hidden_size),nn.LayerNorm(hidden_size))
    def forward(self,x):return self.net(x.float())

class BoundedMLP(nn.Module):
    def __init__(self,input_dim:int,output_dim:int,hidden_dim:int,max_scale:float):
        super().__init__(); self.max_scale=float(max_scale); self.scale_logit=nn.Parameter(torch.tensor(-2.0)); self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,output_dim))
    def forward(self,x):
        scale=torch.sigmoid(self.scale_logit)*self.max_scale; return scale*torch.tanh(self.net(x.float())),scale

class ExplicitClinicalPlanner(nn.Module):
    """Explicit mediator P.

    Visual evidence is consumed only here. The report realizer is not a submodule and receives
    only the returned plan-only tokens/states. Disease nodes are slot-local; related-structure
    nodes are downstream functions of main-plan probabilities through a fixed allowed-relation mask.
    """
    def __init__(self,global_dim:int,task_dim:int,coordinate_dim:int,hidden_size:int=4096,projector_hidden_dim:int=1024,
                 plan_state_hidden:int=512,dropout:float=.1,coordinate_max_scale:float=.10,quality_max_scale:float=.08):
        super().__init__(); self.global_dim=int(global_dim); self.task_dim=int(task_dim); self.coordinate_dim=int(coordinate_dim); self.hidden_size=int(hidden_size)
        self.global_projector=SourceProjector(global_dim,hidden_size,projector_hidden_dim,dropout)
        self.disease_projector=SourceProjector(task_dim,hidden_size,projector_hidden_dim,dropout)
        self.evidence_slot_embeddings=nn.Parameter(torch.randn(1,len(MAIN_SLOT_NAMES),hidden_size)*.01)
        self.coordinate_conditioner=BoundedMLP(coordinate_dim,hidden_size,256,coordinate_max_scale)
        self.quality_conditioner=BoundedMLP(2,hidden_size,128,quality_max_scale)
        self.slot_head=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,1))
        self.slot_bias=nn.Parameter(torch.zeros(len(MAIN_SLOT_NAMES)))

        # Plan-only representation. No raw visual token can enter this pathway.
        self.plan_node_embeddings=nn.Parameter(torch.randn(1,len(PLAN_NODE_NAMES),hidden_size)*.01)
        self.main_state_encoder=nn.Sequential(nn.LayerNorm(5),nn.Linear(5,plan_state_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(plan_state_hidden,hidden_size),nn.LayerNorm(hidden_size))
        self.structure_state_encoder=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,plan_state_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(plan_state_hidden,hidden_size),nn.LayerNorm(hidden_size))

        mask=build_structure_relation_mask(); self.register_buffer("structure_relation_mask",mask,persistent=True)
        self.structure_heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(len(MAIN_SLOT_NAMES)),nn.Linear(len(MAIN_SLOT_NAMES),64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1)) for _ in STRUCTURE_TERMS])

    def initialize_from_pretrained_planner_checkpoint(self,obj:Dict[str,object],strict:bool=False):
        state=obj.get("planner_state_dict",obj); own=self.state_dict(); copied=[]; skipped=[]
        rename={"slot_embeddings":"evidence_slot_embeddings"}
        allowed=("global_projector.","disease_projector.","coordinate_conditioner.","quality_conditioner.","slot_head.","slot_bias")
        for k,v in state.items():
            target=rename.get(k,k)
            if not (k=="slot_embeddings" or any(k==p or k.startswith(p) for p in allowed)): continue
            if target in own and tuple(own[target].shape)==tuple(v.shape): own[target].copy_(v.to(dtype=own[target].dtype)); copied.append(target)
            else: skipped.append((k,target))
        # plan node identity starts from evidence slot identity for the 16 main slots when shapes match.
        if "evidence_slot_embeddings" in copied:
            with torch.no_grad(): self.plan_node_embeddings[:,:len(MAIN_SLOT_NAMES)].copy_(self.evidence_slot_embeddings)
        required=["evidence_slot_embeddings","slot_head.1.weight","slot_head.4.weight","slot_bias","global_projector.net.0.weight","disease_projector.net.0.weight"]
        missing=[k for k in required if k not in copied]
        if strict and missing: raise RuntimeError(f"pretrained planner initialization incomplete: {missing}")
        return {"copied":copied,"skipped":skipped,"missing_required":missing}

    def _main_evidence_tokens(self,global_source,task_features,coordinate_state,task_quality,task_valid):
        x=task_features.float(); b=x.shape[0]
        gt=self.global_projector(global_source)[:,None]
        disease_tm=x.permute(0,2,1,3).reshape(b,15,self.task_dim); dt=self.disease_projector(disease_tm)
        tokens=torch.cat([gt,dt],1)+self.evidence_slot_embeddings
        coord_tm=coordinate_state[:,None].expand(-1,3,-1,-1).reshape(b,15,self.coordinate_dim); cd,cs=self.coordinate_conditioner(coord_tm); gcd,_=self.coordinate_conditioner(coordinate_state.mean(1,keepdim=True))
        qtm=task_quality.permute(0,2,1).reshape(b,15,1); vtm=task_valid.permute(0,2,1).reshape(b,15,1); qd,qs=self.quality_conditioner(torch.cat([qtm,vtm],-1)); gq=torch.ones(b,1,2,device=x.device,dtype=x.dtype); gqd,_=self.quality_conditioner(gq)
        tokens=tokens+torch.cat([gcd,cd],1)+torch.cat([gqd,qd],1)
        quality=torch.cat([torch.ones(b,1,device=x.device),qtm.squeeze(-1)],1); valid=torch.cat([torch.ones(b,1,device=x.device),vtm.squeeze(-1)],1)
        return tokens,quality,valid,cs,qs

    def structure_from_main(self,main_probabilities:torch.Tensor):
        probs=main_probabilities.float(); logits=[]
        for si,head in enumerate(self.structure_heads):
            masked=probs*self.structure_relation_mask[si][None].to(probs)
            logits.append(head(masked).squeeze(-1))
        logits=torch.stack(logits,1); return logits,torch.sigmoid(logits)

    def build_plan_tokens_from_states(self,main_logits,main_probabilities,main_quality,main_valid,structure_logits=None,structure_probabilities=None):
        p=main_probabilities.float(); lg=main_logits.float(); conf=(p-.5).abs()*2
        main_state=torch.stack([p,torch.tanh(lg),conf,main_quality.float(),main_valid.float()],-1)
        mt=self.main_state_encoder(main_state)+self.plan_node_embeddings[:,:len(MAIN_SLOT_NAMES)]
        if structure_logits is None or structure_probabilities is None:
            structure_logits,structure_probabilities=self.structure_from_main(main_probabilities)
        sp=structure_probabilities.float(); sl=structure_logits.float(); sconf=(sp-.5).abs()*2; related_mass=[]
        mask=self.structure_relation_mask.to(sp)
        for si in range(len(STRUCTURE_TERMS)):
            denom=mask[si].sum().clamp_min(1); related_mass.append((p*mask[si][None]).sum(-1)/denom)
        related_mass=torch.stack(related_mass,1)
        sstate=torch.stack([sp,torch.tanh(sl),sconf,related_mass],-1)
        st=self.structure_state_encoder(sstate)+self.plan_node_embeddings[:,len(MAIN_SLOT_NAMES):]
        return torch.cat([mt,st],1),main_state,sstate

    def forward(self,global_source,task_features,coordinate_state,task_quality,task_valid):
        evidence_tokens,q,v,cs,qs=self._main_evidence_tokens(global_source,task_features,coordinate_state,task_quality,task_valid)
        logits=self.slot_head(evidence_tokens).squeeze(-1)+self.slot_bias[None]; probs=torch.sigmoid(logits)
        slogits,sprobs=self.structure_from_main(probs)
        plan_tokens,mstate,sstate=self.build_plan_tokens_from_states(logits,probs,q,v,slogits,sprobs)
        return {"main_logits":logits,"main_probabilities":probs,"main_quality":q,"main_valid":v,"structure_logits":slogits,"structure_probabilities":sprobs,
                "plan_tokens":plan_tokens,"main_state":mstate,"structure_state":sstate,"evidence_tokens":evidence_tokens,"coordinate_condition_scale":cs,"quality_condition_scale":qs}

    def intervene_main_plan(self,main_logits,main_probabilities,main_quality,main_valid,slot_index:int,target_probability:torch.Tensor):
        """Direct do(P_j) for mediator audit. Only one main plan node is changed; structure nodes are recomputed downstream."""
        p=main_probabilities.clone(); lg=main_logits.clone(); j=int(slot_index); tp=torch.as_tensor(target_probability,device=p.device,dtype=p.dtype).view(-1).clamp(1e-4,1-1e-4)
        if tp.numel()==1 and p.shape[0]>1: tp=tp.expand(p.shape[0])
        p[:,j]=tp; lg[:,j]=torch.logit(tp)
        sl,sp=self.structure_from_main(p); tokens,ms,ss=self.build_plan_tokens_from_states(lg,p,main_quality,main_valid,sl,sp)
        return {"main_logits":lg,"main_probabilities":p,"main_quality":main_quality,"main_valid":main_valid,"structure_logits":sl,"structure_probabilities":sp,"plan_tokens":tokens,"main_state":ms,"structure_state":ss}

