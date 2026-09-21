#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json,torch
from .planner import ExplicitClinicalPlanner
from .realizer import PlanOnlyReportRealizer

def safe_torch_load(path:str):
    try:return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:return torch.load(path,map_location="cpu")

def build_planner(global_dim,task_dim,coordinate_dim,hidden_size=4096,projector_hidden_dim=1024,plan_state_hidden=512,dropout=.1):
    return ExplicitClinicalPlanner(global_dim,task_dim,coordinate_dim,hidden_size,projector_hidden_dim,plan_state_hidden,dropout)

def save_planner_checkpoint(out_dir,model,config,metrics):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);p=out/"clinical_planner.pt";torch.save({"config":config,"metrics":metrics,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},p);return p

def load_planner_checkpoint(path):
    obj=safe_torch_load(path);c=obj["config"];m=build_planner(c["global_dim"],c["task_dim"],c["coordinate_dim"],c["hidden_size"],c.get("projector_hidden_dim",1024),c.get("plan_state_hidden",512),c.get("dropout",.1));m.load_state_dict(obj["state_dict"],strict=True);return m,obj

def build_tokenizer(llm_dir):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(llm_dir,use_fast=False,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    return tok

def build_llm_with_lora(llm_dir,dtype_name="float16",lora_r=16,lora_alpha=32,lora_dropout=.05,resume_lora_dir=""):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig,PeftModel,TaskType,get_peft_model
    dtype={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[dtype_name]
    base=AutoModelForCausalLM.from_pretrained(llm_dir,torch_dtype=dtype,low_cpu_mem_usage=True,local_files_only=True)
    if resume_lora_dir and Path(resume_lora_dir).is_dir():llm=PeftModel.from_pretrained(base,resume_lora_dir,is_trainable=True)
    else:
        cfg=LoraConfig(task_type=TaskType.CAUSAL_LM,r=int(lora_r),lora_alpha=int(lora_alpha),lora_dropout=float(lora_dropout),bias="none",target_modules=["q_proj","k_proj","v_proj","o_proj"]);llm=get_peft_model(base,cfg)
    if hasattr(llm,"gradient_checkpointing_enable"):llm.gradient_checkpointing_enable()
    if hasattr(llm,"enable_input_require_grads"):llm.enable_input_require_grads()
    if hasattr(llm.config,"use_cache"):llm.config.use_cache=False
    for p in llm.parameters():
        if p.requires_grad:p.data=p.data.float()
    return llm

def build_realizer(llm,hidden_size,adapter_hidden=512,adapter_max_scale=.12,dropout=.1,init_realizer_checkpoint=""):
    m=PlanOnlyReportRealizer(llm,hidden_size,adapter_hidden,adapter_max_scale,dropout);copied=[]
    if init_realizer_checkpoint and Path(init_realizer_checkpoint).is_file():
        obj=safe_torch_load(init_realizer_checkpoint);copied=m.initialize_plan_adapter_from_state(obj.get("non_llm_state_dict",{}))
    return m,copied

def save_realizer_checkpoint(out_dir,model,config,metrics):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);p=out/"report_realizer.pt";torch.save({"config":config,"metrics":metrics,"non_llm_state_dict":model.non_llm_state_dict()},p);ld=out/"lora";model.llm.save_pretrained(ld);(out/"checkpoint_manifest.json").write_text(json.dumps({"checkpoint":str(p.resolve()),"lora_dir":str(ld.resolve()),"config":config,"metrics":metrics},ensure_ascii=False,indent=2),encoding="utf-8");return p,ld

def load_realizer(checkpoint,llm_dir,lora_dir):
    obj=safe_torch_load(checkpoint);c=obj["config"];llm=build_llm_with_lora(llm_dir,c.get("dtype_name","float16"),c.get("lora_r",16),c.get("lora_alpha",32),c.get("lora_dropout",.05),lora_dir);m=PlanOnlyReportRealizer(llm,c["hidden_size"],c.get("adapter_hidden",512),c.get("adapter_max_scale",.12),c.get("dropout",.1));own=m.state_dict()
    for k,v in obj["non_llm_state_dict"].items():
        if k in own and tuple(own[k].shape)==tuple(v.shape):own[k].copy_(v.to(dtype=own[k].dtype))
    return m,obj

