from __future__ import annotations
import torch

def _dtype(device):return torch.bfloat16 if device.startswith('cuda') and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if device.startswith('cuda') and torch.cuda.is_available() else torch.float32)
def _clear(model):
    if hasattr(model,'peft_config'):
        try:delattr(model,'peft_config')
        except Exception:pass
    if hasattr(model,'_hf_peft_config_loaded'):
        try:model._hf_peft_config_loaded=False
        except Exception:pass
    return model
def _merge(model,adapter):
    from peft import PeftModel
    p=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    try:m=p.merge_and_unload(safe_merge=True)
    except TypeError:m=p.merge_and_unload()
    return _clear(m)
def load_v32(base_model,language_adapter,r31v1_adapter,scaffold_adapter,device):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(base_model,trust_remote_code=True,use_fast=False);dtype=_dtype(device)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(base_model,dtype=dtype,trust_remote_code=True,low_cpu_mem_usage=True)
    if device.startswith('cuda') and torch.cuda.is_available():m=m.to(device)
    m=_merge(m,language_adapter);m=_merge(m,r31v1_adapter)
    from peft import PeftModel
    m=PeftModel.from_pretrained(m,str(scaffold_adapter),is_trainable=False);m.eval();return tok,m
