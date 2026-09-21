from __future__ import annotations
import torch
from .prompting import messages
from .modeling import load_realizer_inference
from .eval_bridge import load_parser,parse,normalize
from .scaffold import build_scaffold
from .precision_patch import precision_finalize

def load_model(base_model,language_adapter,r31v1_adapter,v31_adapter,device):return load_realizer_inference(base_model,language_adapter,r31v1_adapter,v31_adapter,device)
def _gen(tok,model,msgs,g):
    p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);x=tok(p,return_tensors='pt');dev=next(model.parameters()).device;x={k:v.to(dev) for k,v in x.items()};kw={'max_new_tokens':int(g['max_new_tokens']),'do_sample':False,'num_beams':int(g['num_beams']),'repetition_penalty':float(g['repetition_penalty']),'pad_token_id':tok.eos_token_id,'eos_token_id':tok.eos_token_id}
    with torch.no_grad():y=model.generate(**x,**kw)
    return tok.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
def generate_one(tok,model,row,generation,lex,parser_module,scaffold_cfg,patch_cfg):
    mod=load_parser(parser_module);scaffold,sm=build_scaffold(row['direct_draft'],mod,parse,normalize,scaffold_cfg);raw=_gen(tok,model,messages(row['core_binary'],row['slot_valid'],scaffold),generation);final,fm=precision_finalize(raw,row['core_binary'],row['slot_valid'],lex,mod,parse,normalize,patch_cfg)
    return {'linguistic_scaffold':scaffold,'scaffold_meta':sm,'generated_raw':raw,'generated':final,'finalize_meta':fm}
