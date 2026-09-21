from __future__ import annotations
import random,torch
from .prompting import messages

def build_prompt(tok,row):
    msgs=messages(row['core_binary'],row['slot_valid'],row['linguistic_scaffold']);p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);x=tok(p,return_tensors='pt');return p,x

def _decode(tok,y,input_len):return tok.decode(y[input_len:],skip_special_tokens=True).strip()

def _inputs(tok,model,row):
    _,x=build_prompt(tok,row);dev=next(model.parameters()).device;x={k:v.to(dev) for k,v in x.items()};return x,x['input_ids'].shape[1]

def _common(tok,cfg):
    return {'max_new_tokens':int(cfg['max_new_tokens']),'repetition_penalty':float(cfg['repetition_penalty']),'pad_token_id':tok.eos_token_id,'eos_token_id':tok.eos_token_id}

def generate_greedy(tok,model,row,cfg):
    x,L=_inputs(tok,model,row)
    with torch.no_grad():y=model.generate(**x,**_common(tok,cfg),do_sample=False,num_beams=1)
    return ('greedy',_decode(tok,y[0],L))

def generate_samples(tok,model,row,cfg,seed,count=None):
    n=int(count if count is not None else cfg.get('sample_count',2));
    if n<=0:return []
    x,L=_inputs(tok,model,row);torch.manual_seed(int(seed));random.seed(int(seed))
    if torch.cuda.is_available():torch.cuda.manual_seed_all(int(seed))
    with torch.no_grad():
        ys=model.generate(**x,**_common(tok,cfg),do_sample=True,num_beams=1,num_return_sequences=n,temperature=float(cfg['temperature']),top_p=float(cfg['top_p']),top_k=int(cfg['top_k']))
    return [(f'sample{i}',_decode(tok,y,L)) for i,y in enumerate(ys)]

def dedupe_pool(pool):
    seen=set();ret=[]
    for tag,text in pool:
        text=str(text).strip()
        if not text or text in seen:continue
        seen.add(text);ret.append((str(tag),text))
    return ret
