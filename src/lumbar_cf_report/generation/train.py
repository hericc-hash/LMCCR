from __future__ import annotations
import math,random,time
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from .prompting import messages
from .modeling import load_trainable_realizer

class DS(Dataset):
    def __init__(self,rows,tok,max_len):self.rows=rows;self.tok=tok;self.max_len=max_len
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];msgs=messages(r['core_binary'],r['slot_valid'],r['linguistic_scaffold']);target=str(r['target_text']).strip()
        prompt=self.tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);full=self.tok.apply_chat_template(msgs+[{'role':'assistant','content':target}],tokenize=False,add_generation_prompt=False)
        pids=self.tok(prompt,add_special_tokens=False)['input_ids'];fids=self.tok(full,add_special_tokens=False)['input_ids'];lcp=0
        for a,b in zip(pids,fids):
            if a!=b:break
            lcp+=1
        if lcp<max(8,int(.9*min(len(pids),len(fids)))):raise RuntimeError('chat prefix mismatch')
        fids=fids[:self.max_len]
        if lcp>=len(fids)-8:raise RuntimeError('target truncated away')
        labels=[-100]*len(fids)
        for j in range(lcp,len(fids)):labels[j]=fids[j]
        return {'input_ids':torch.tensor(fids),'attention_mask':torch.ones(len(fids),dtype=torch.long),'labels':torch.tensor(labels),'supervised_tokens':torch.tensor(len(fids)-lcp)}

def collate(b):
    if len(b)!=1:raise RuntimeError('batch_size must be1')
    return {k:v.unsqueeze(0) for k,v in b[0].items()}

def train(rows,base,language_adapter,r31v1_adapter,out,cfg,lora_cfg,device,seed):
    from transformers import get_linear_schedule_with_warmup
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    tok,model,meta=load_trainable_realizer(base,language_adapter,r31v1_adapter,device,lora_cfg,cfg['gradient_checkpointing']);ds=DS(rows,tok,int(cfg['max_seq_tokens']));g=torch.Generator().manual_seed(seed);dl=DataLoader(ds,batch_size=1,shuffle=True,collate_fn=collate,generator=g,num_workers=0)
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=float(cfg['lr']),weight_decay=float(cfg['weight_decay']));acc=int(cfg['grad_accum']);epochs=int(cfg['epochs']);total=max(1,math.ceil(len(dl)/acc)*epochs);sch=get_linear_schedule_with_warmup(opt,int(total*float(cfg['warmup_ratio'])),total);step=0;hist=[];opt.zero_grad(set_to_none=True)
    for ep in range(1,epochs+1):
        losses=[];t=time.time()
        for bi,b in enumerate(dl,1):
            b.pop('supervised_tokens');dev=next(model.parameters()).device;b={k:v.to(dev) for k,v in b.items()};loss=model(**b).loss
            if not torch.isfinite(loss):raise RuntimeError('non-finite scaffold loss')
            (loss/acc).backward();losses.append(float(loss.detach().cpu()))
            if bi%acc==0 or bi==len(dl):
                torch.nn.utils.clip_grad_norm_(params,float(cfg['max_grad_norm']));opt.step();sch.step();opt.zero_grad(set_to_none=True);step+=1
                if step%int(cfg['log_every'])==0:print(f'[R3.1-v3.2] epoch={ep} step={step}/{total} loss={sum(losses[-20:])/min(20,len(losses)):.4f}',flush=True)
        hist.append({'epoch':ep,'mean_train_loss':sum(losses)/len(losses),'seconds':time.time()-t})
    d=Path(out)/'final_epoch_2/scaffold_lora';d.mkdir(parents=True,exist_ok=True);model.save_pretrained(str(d));return hist,meta,str(d)
