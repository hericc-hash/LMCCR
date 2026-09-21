#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundedPlanAdapter(nn.Module):
    def __init__(self,hidden_size:int,bottleneck:int=512,dropout:float=.1,max_scale:float=.12):
        super().__init__();self.max_scale=float(max_scale);self.scale_logit=nn.Parameter(torch.tensor(-2.0));self.net=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,bottleneck),nn.GELU(),nn.Dropout(dropout),nn.Linear(bottleneck,hidden_size));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        scale=torch.sigmoid(self.scale_logit)*self.max_scale;return x.float()+scale*torch.tanh(self.net(x.float())),scale

class PlanOnlyReportRealizer(nn.Module):
    """Plan-faithful language realizer.

    Inputs are restricted to explicit P-derived continuous plan tokens, deterministic P-derived
    control-text token ids, and language prompt/targets. No raw MRI, A, R, C, task feature, or
    global visual token can enter this module.
    """
    def __init__(self,llm:nn.Module,hidden_size:int,adapter_hidden:int=512,adapter_max_scale:float=.12,dropout:float=.1):
        super().__init__();self.llm=llm;self.hidden_size=int(hidden_size)
        if int(llm.config.hidden_size)!=self.hidden_size:raise ValueError("plan/LLM hidden size mismatch")
        self.plan_adapter=BoundedPlanAdapter(hidden_size,adapter_hidden,dropout,adapter_max_scale)
        if hasattr(self.llm.config,"use_cache"):self.llm.config.use_cache=False

    def adapt_plan(self,plan_tokens):
        p,scale=self.plan_adapter(plan_tokens);cons=1-F.cosine_similarity(F.normalize(p,dim=-1),F.normalize(plan_tokens.detach(),dim=-1),dim=-1).mean();return p,scale,cons

    def _assemble(self,plan_tokens,control_input_ids,control_attention_mask,input_ids,attention_mask,labels=None):
        prefix,scale,cons=self.adapt_plan(plan_tokens);embw=self.llm.get_input_embeddings().weight;dev,dtype=embw.device,embw.dtype
        control_input_ids=control_input_ids.to(dev);control_attention_mask=control_attention_mask.to(dev);input_ids=input_ids.to(dev);attention_mask=attention_mask.to(dev)
        control_emb=self.llm.get_input_embeddings()(control_input_ids).to(dtype=dtype);text_emb=self.llm.get_input_embeddings()(input_ids).to(dtype=dtype);prefix=prefix.to(dev,dtype=dtype)
        inputs=torch.cat([prefix,control_emb,text_emb],1)
        mask=torch.cat([torch.ones(prefix.shape[:2],device=dev,dtype=attention_mask.dtype),control_attention_mask.to(dtype=attention_mask.dtype),attention_mask],1)
        full_labels=None
        if labels is not None:
            labels=labels.to(dev);pre=torch.full(prefix.shape[:2],-100,device=dev,dtype=labels.dtype);ctl=torch.full(control_input_ids.shape,-100,device=dev,dtype=labels.dtype);full_labels=torch.cat([pre,ctl,labels],1)
        return inputs,mask,full_labels,scale,cons

    def forward(self,plan_tokens,control_input_ids,control_attention_mask,input_ids,attention_mask,labels=None):
        inputs,mask,full_labels,scale,cons=self._assemble(plan_tokens,control_input_ids,control_attention_mask,input_ids,attention_mask,labels)
        out=self.llm(inputs_embeds=inputs,attention_mask=mask,labels=full_labels,use_cache=False,return_dict=True)
        return {"lm_loss":out.loss if getattr(out,"loss",None) is not None else inputs.sum()*0,"llm_logits":getattr(out,"logits",None),"plan_adapter_scale":scale,"plan_consistency_loss":cons}

    @torch.inference_mode()
    def generate_text(self,plan_tokens,control_input_ids,control_attention_mask,prompt_input_ids,prompt_attention_mask,max_new_tokens=512,eos_token_id=None,pad_token_id=None,do_sample=False,temperature=.7,top_p=.9,num_beams=1):
        inputs,mask,_,scale,cons=self._assemble(plan_tokens,control_input_ids,control_attention_mask,prompt_input_ids,prompt_attention_mask,None)
        kw=dict(inputs_embeds=inputs,attention_mask=mask,max_new_tokens=max_new_tokens,do_sample=do_sample,num_beams=num_beams,eos_token_id=eos_token_id,pad_token_id=pad_token_id,use_cache=True)
        if do_sample:kw.update(temperature=temperature,top_p=top_p)
        seq=self.llm.generate(**kw);return {"generated_ids":seq,"plan_adapter_scale":scale,"plan_consistency_loss":cons}

    def initialize_plan_adapter_from_state(self,state:dict):
        own=self.state_dict();copied=[]
        for k,v in state.items():
            if not k.startswith("plan_adapter."):continue
            if k in own and tuple(own[k].shape)==tuple(v.shape):own[k].copy_(v.to(dtype=own[k].dtype));copied.append(k)
        return copied

    def non_llm_state_dict(self):return {k:v.detach().cpu() for k,v in self.state_dict().items() if not k.startswith("llm.")}

