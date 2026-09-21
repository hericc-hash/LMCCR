from __future__ import annotations
import torch

def _dtype(device):
    return torch.bfloat16 if device.startswith('cuda') and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if device.startswith('cuda') and torch.cuda.is_available() else torch.float32)

def _clear_peft_meta(model):
    if hasattr(model,'peft_config'):
        try:delattr(model,'peft_config')
        except Exception:pass
    if hasattr(model,'_hf_peft_config_loaded'):
        try:model._hf_peft_config_loaded=False
        except Exception:pass
    return model

def _merge_adapter(model,adapter):
    from peft import PeftModel
    p=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    try:m=p.merge_and_unload(safe_merge=True)
    except TypeError:m=p.merge_and_unload()
    return _clear_peft_meta(m)

def load_r31v1_merged_base(base_model,language_adapter,r31v1_adapter,device):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(base_model,trust_remote_code=True,use_fast=False);dtype=_dtype(device)
    base=AutoModelForCausalLM.from_pretrained(base_model,dtype=dtype,trust_remote_code=True,low_cpu_mem_usage=True)
    if device.startswith('cuda') and torch.cuda.is_available():base=base.to(device)
    base=_merge_adapter(base,language_adapter)
    base=_merge_adapter(base,r31v1_adapter)
    base.requires_grad_(False)
    return tok,base,dtype

def load_trainable_realizer(base_model,language_adapter,r31v1_adapter,device,lora_cfg,gradient_checkpointing=True):
    from peft import LoraConfig,get_peft_model
    tok,merged,dtype=load_r31v1_merged_base(base_model,language_adapter,r31v1_adapter,device)
    pc=LoraConfig(r=int(lora_cfg['r']),lora_alpha=int(lora_cfg['alpha']),lora_dropout=float(lora_cfg['dropout']),bias='none',task_type='CAUSAL_LM',target_modules=list(lora_cfg['target_modules']))
    model=get_peft_model(merged,pc)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable();model.enable_input_require_grads();model.config.use_cache=False
    tr=[(n,p) for n,p in model.named_parameters() if p.requires_grad];bad=[n for n,_ in tr if 'lora_' not in n.lower()]
    if bad or not tr:raise RuntimeError('R3.1-v3.2 LoRA trainable scope invalid '+str(bad[:5]))
    model.train()
    return tok,model,{'dtype':str(dtype),'trainable_tensors':len(tr),'trainable_params':sum(p.numel() for _,p in tr),'r':int(lora_cfg['r']),'alpha':int(lora_cfg['alpha']),'language_adapter_merged':str(language_adapter),'r31v1_adapter_merged':str(r31v1_adapter)}

def load_realizer_inference(base_model,language_adapter,r31v1_adapter,v31_adapter,device):
    from peft import PeftModel
    tok,merged,_=load_r31v1_merged_base(base_model,language_adapter,r31v1_adapter,device)
    model=PeftModel.from_pretrained(merged,str(v31_adapter),is_trainable=False);model.eval();return tok,model
