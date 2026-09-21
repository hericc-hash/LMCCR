from __future__ import annotations
import math,torch
from torch import nn
import torch.nn.functional as F

class Stage23UnifiedClinicalPlanner(nn.Module):
    def __init__(self,cfg,anchor_w=None,anchor_b=None):
        super().__init__();m=cfg['model'];D=m['task_dim'];H=m['slot_hidden'];G=m['global_hidden'];C=m['coordinate_hidden'];E=m['embedding_dim']
        self.max_res=float(m['max_context_logit_residual']);self.max_gate=float(m['max_cross_slot_gate']);self.produce_plan_tokens=bool(m.get('produce_plan_tokens',False));self.plan_token_dim=int(m.get('plan_token_dim',4096))
        self.anchor_w=nn.Parameter(torch.zeros(3,D),requires_grad=False);self.anchor_b=nn.Parameter(torch.zeros(3),requires_grad=False)
        if anchor_w is not None:self.anchor_w.data.copy_(anchor_w)
        if anchor_b is not None:self.anchor_b.data.copy_(anchor_b)
        self.global_proj=nn.Sequential(nn.Linear(m['global_dim'],G),nn.GELU(),nn.LayerNorm(G))
        self.coord_proj=nn.Sequential(nn.Linear(m['coordinate_dim'],C),nn.GELU(),nn.LayerNorm(C))
        self.level_emb=nn.Embedding(5,E);self.task_emb=nn.Embedding(3,E)
        inp=D+G+C+2*E+2
        self.slot_mlp=nn.Sequential(nn.Linear(inp,H),nn.GELU(),nn.Dropout(m['dropout']),nn.Linear(H,H),nn.GELU(),nn.LayerNorm(H))
        self.cross_proj=nn.Sequential(nn.Linear(H,H),nn.GELU(),nn.Linear(H,H))
        frac=max(min(float(m.get('initial_cross_slot_gate',.02))/max(self.max_gate,1e-8),.999),.001)
        self.cross_gate_raw=nn.Parameter(torch.full((3,),math.log(frac/(1-frac))))
        self.residual_heads=nn.ModuleList([nn.Linear(H,1) for _ in range(3)])
        for h in self.residual_heads:nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
        self.lordosis_head=nn.Sequential(nn.Linear(G,64),nn.GELU(),nn.Linear(64,1))
        self.structure_head=nn.Sequential(nn.Linear(G+H,96),nn.GELU(),nn.Dropout(m['dropout']),nn.Linear(96,8))
        self.plan_state_proj=nn.Linear(H+2,int(m['plan_state_dim']))
        self.aux_state_proj=nn.Linear(G+H,int(m['plan_state_dim']))
        self.plan_token_proj=nn.Linear(int(m['plan_state_dim']),self.plan_token_dim)
    def anchor_logits(self,E):return torch.einsum('bltd,td->blt',E,self.anchor_w)+self.anchor_b[None,None,:]
    def forward(self,global_source,task_features,coordinate_state,task_quality,task_valid,return_plan_tokens=None):
        B,L,T,D=task_features.shape;g=self.global_proj(global_source);c=self.coord_proj(coordinate_state)
        le=self.level_emb(torch.arange(L,device=task_features.device))[None,:,None,:].expand(B,L,T,-1);te=self.task_emb(torch.arange(T,device=task_features.device))[None,None,:,:].expand(B,L,T,-1)
        gg=g[:,None,None,:].expand(B,L,T,-1);cc=c[:,:,None,:].expand(B,L,T,-1);q=task_quality[...,None];v=task_valid[...,None]
        h=self.slot_mlp(torch.cat([task_features,gg,cc,le,te,q,v],-1))*v
        denom=v.sum((1,2),keepdim=True).clamp_min(1.);pooled=(h.sum((1,2),keepdim=True)/denom)
        gate=self.max_gate*torch.sigmoid(self.cross_gate_raw)[None,None,:,None]
        h2=h+gate*self.cross_proj(pooled).expand_as(h)
        r=torch.cat([self.residual_heads[t](h2[:,:,t]) for t in range(T)],-1)
        r=self.max_res*torch.tanh(r)*task_valid
        a=self.anchor_logits(task_features);dlog=a+r;dprob=torch.sigmoid(dlog)
        lord=self.lordosis_head(g).squeeze(-1);main=torch.cat([lord[:,None],dlog.permute(0,2,1).reshape(B,15)],1)
        ph=(h2*task_valid[...,None]).sum((1,2))/task_valid.sum((1,2)).unsqueeze(-1).clamp_min(1.)
        struct=self.structure_head(torch.cat([g,ph],-1))
        slot_states=self.plan_state_proj(torch.cat([h2,dlog[...,None],dprob[...,None]],-1)).permute(0,2,1,3).reshape(B,15,-1)
        aux=self.aux_state_proj(torch.cat([g,ph],-1));lord_state=aux[:,None,:];struct_states=aux[:,None,:].expand(B,8,-1);plan_state=torch.cat([lord_state,slot_states,struct_states],1)
        rp=self.produce_plan_tokens if return_plan_tokens is None else return_plan_tokens
        out={'anchor_logits':a,'context_residual_logits':r,'disease_logits':dlog,'disease_probabilities':dprob,'main_logits':main,'main_probabilities':torch.sigmoid(main),'structure_logits':struct,'structure_probabilities':torch.sigmoid(struct),'plan_state':plan_state,'cross_slot_gate':self.max_gate*torch.sigmoid(self.cross_gate_raw)}
        if rp:out['plan_tokens']=self.plan_token_proj(plan_state)
        return out
