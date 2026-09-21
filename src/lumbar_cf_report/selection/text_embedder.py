from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch

LEVELS=['L1/2','L2/3','L3/4','L4/5','L5/S1']
SLOTS=['lordosis']+[f'disc:{x}' for x in LEVELS]+[f'stenosis:{x}' for x in LEVELS]+[f'nerve:{x}' for x in LEVELS]


def _clean(x,limit=1600):
    s=str(x or '').strip().replace('\x00',' ')
    return s[:limit]


def planner_text(row):
    core=list(row.get('core_binary',[0]*16))[:16]
    probs=list(row.get('planner_probs',[0.5]*16))[:16]
    valid=list(row.get('slot_valid',[1]*16))[:16]
    out=[]
    for i,name in enumerate(SLOTS):
        if i>=len(valid) or not int(valid[i]):
            continue
        p=float(probs[i]) if i<len(probs) else .5
        out.append(f'{name}={int(core[i])}({p:.2f})')
    return '; '.join(out)


def build_prefix(row):
    # Deployment-safe only: Planner, Direct draft, and redacted scaffold.
    return (
        'Clinical planner states:\n'+planner_text(row)+'\n\n'
        'Patient direct draft:\n'+_clean(row.get('direct_draft',''))+'\n\n'
        'Core-redacted linguistic scaffold:\n'+_clean(row.get('linguistic_scaffold',''))+'\n\n'
        'Candidate report:\n'
    )


def input_fingerprint(row,cand):
    h=hashlib.sha256()
    h.update(build_prefix(row).encode('utf-8'))
    h.update(str(cand.get('tag','')).encode('utf-8'))
    h.update(_clean(cand.get('text','')).encode('utf-8'))
    return h.hexdigest()


def _backbone(model):
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    if hasattr(base,'model'):
        return base.model
    if hasattr(base,'transformer'):
        return base.transformer
    raise RuntimeError('Could not locate causal-LM backbone for hidden-state extraction')


def _encode_parts(tok,row,cand,prefix_tokens,candidate_tokens):
    prefix=build_prefix(row)
    text=_clean(cand.get('text',''))
    pids=tok(prefix,add_special_tokens=False)['input_ids']
    cids=tok(text,add_special_tokens=False)['input_ids']
    # Preserve the end of context (scaffold + Candidate marker) and the full beginning of report.
    pids=pids[-int(prefix_tokens):]
    cids=cids[:int(candidate_tokens)]
    if not cids:
        cids=[tok.eos_token_id]
    ids=pids+cids
    cm=[0]*len(pids)+[1]*len(cids)
    return ids,cm


def encode_pool(tok,model,rows,cfg,device='cuda'):
    ecfg=cfg['text_encoder'];batch_size=int(ecfg.get('batch_size',2));pt=int(ecfg.get('prefix_max_tokens',448));ct=int(ecfg.get('candidate_max_tokens',448))
    flat=[]
    for row in rows:
        for j,cand in enumerate(row.get('candidates',[])):
            ids,cm=_encode_parts(tok,row,cand,pt,ct)
            flat.append((str(row['serial']),str(cand.get('tag',f'candidate{j}')),j,input_fingerprint(row,cand),ids,cm))
    if not flat:raise RuntimeError('No candidates for text embedding')
    dev=torch.device(device if str(device).startswith('cuda') and torch.cuda.is_available() else 'cpu');model.eval();bb=_backbone(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    all_emb=[]
    with torch.inference_mode():
        for st in range(0,len(flat),batch_size):
            chunk=flat[st:st+batch_size];mx=max(len(x[4]) for x in chunk);ids=[];att=[];cmask=[]
            for _,_,_,_,seq,cm in chunk:
                n=mx-len(seq);ids.append(seq+[pad]*n);att.append([1]*len(seq)+[0]*n);cmask.append(cm+[0]*n)
            ids_t=torch.tensor(ids,dtype=torch.long,device=dev);att_t=torch.tensor(att,dtype=torch.long,device=dev);cm_t=torch.tensor(cmask,dtype=torch.bool,device=dev)
            out=bb(input_ids=ids_t,attention_mask=att_t,use_cache=False,return_dict=True)
            hs=out.last_hidden_state.float()
            den=cm_t.sum(1,keepdim=True).clamp_min(1).float();mean=(hs*cm_t.unsqueeze(-1)).sum(1)/den
            last_idx=(cm_t.long()*torch.arange(mx,device=dev).view(1,-1)).max(1).values
            last=hs[torch.arange(hs.shape[0],device=dev),last_idx]
            pooled=.70*mean+.30*last
            pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
            all_emb.append(pooled.cpu().numpy().astype(np.float16))
            if (st//batch_size)%25==0 or st+batch_size>=len(flat):
                print(f'[TextEmbed] {min(st+batch_size,len(flat))}/{len(flat)}',flush=True)
    E=np.concatenate(all_emb,axis=0)
    return {
        'serial':np.asarray([x[0] for x in flat],dtype=object),
        'tag':np.asarray([x[1] for x in flat],dtype=object),
        'candidate_index':np.asarray([x[2] for x in flat],dtype=np.int16),
        'fingerprint':np.asarray([x[3] for x in flat],dtype=object),
        'embedding':E,
    }


def save_embedding_cache(cache,path,meta=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**cache)
    if meta is not None:
        p.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


def load_embedding_cache(path):
    z=np.load(path,allow_pickle=True)
    return {k:z[k] for k in z.files}


def cache_to_case_embeddings(rows,cache,strict=True):
    mp={}
    for i,(s,t,j,fp) in enumerate(zip(cache['serial'],cache['tag'],cache['candidate_index'],cache['fingerprint'])):
        mp[(str(s),str(t),int(j),str(fp))]=cache['embedding'][i].astype(np.float32)
    out=[];missing=[]
    for row in rows:
        arr=[]
        for j,cand in enumerate(row.get('candidates',[])):
            key=(str(row['serial']),str(cand.get('tag',f'candidate{j}')),j,input_fingerprint(row,cand))
            if key not in mp:
                missing.append((str(row['serial']),str(cand.get('tag')),j));arr.append(None)
            else:arr.append(mp[key])
        if any(x is None for x in arr):out.append(None)
        else:out.append(np.stack(arr,axis=0))
    if strict and missing:raise RuntimeError('Text embedding cache mismatch/missing '+str(missing[:12]))
    return out
