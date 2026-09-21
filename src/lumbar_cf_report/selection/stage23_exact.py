from __future__ import annotations
from pathlib import Path
import json,torch
from .io import load_pt
from .planner import Stage23UnifiedClinicalPlanner
EXPECTED_VERSION='LMCCR-Stage2.3B-CF-Supervision-v1.7'
EXPECTED_MODEL={'task_dim':128,'global_dim':256,'coordinate_dim':47,'slot_hidden':96,'global_hidden':64,'coordinate_hidden':32,'embedding_dim':16,'dropout':0.1,'max_context_logit_residual':0.75,'max_cross_slot_gate':0.15,'initial_cross_slot_gate':0.02,'plan_state_dim':128,'plan_token_dim':4096,'produce_plan_tokens':False}

def load_runtime(root):
    root=Path(root);paths={'candidate':root/'05_stage2_3B_selected_candidate.pt','data':root/'01_stage23b_data.pt','config':root/'config_used.json','saved_probs':root/'07_internal_factual_probs.pt'}
    miss=[str(x) for x in paths.values() if not x.exists()]
    if miss:raise FileNotFoundError('Stage2.3-v1.7 runtime missing '+str(miss))
    ck=load_pt(paths['candidate']);data=load_pt(paths['data']);cfg=json.loads(paths['config'].read_text(encoding='utf-8'));saved=load_pt(paths['saved_probs'])
    if str(ck.get('version'))!=EXPECTED_VERSION or str(cfg.get('version'))!=EXPECTED_VERSION:raise RuntimeError('Stage2.3 version drift')
    mm={k:(v,cfg.get('model',{}).get(k)) for k,v in EXPECTED_MODEL.items() if cfg.get('model',{}).get(k)!=v}
    if mm:raise RuntimeError('Stage2.3 model config drift '+str(mm))
    return cfg,ck,data,saved,paths

def build_model(cfg,ck,device='cpu'):
    m=Stage23UnifiedClinicalPlanner(cfg,anchor_w=ck['anchor_weight'],anchor_b=ck['anchor_bias']);m.load_state_dict(ck['state_dict'],strict=True);return m.to(device).eval()

def replay_split(cfg,ck,data,split,device='cpu',batch=64):
    d=data[split];m=build_model(cfg,ck,device);outs=[]
    with torch.no_grad():
        for st in range(0,len(d['serials']),batch):
            sl=slice(st,min(st+batch,len(d['serials'])));kw=[d[k][sl].to(device) for k in ['global_source','raw_E','coordinate_state','task_quality','task_valid']];o=m(*kw);outs.append(o['main_probabilities'].detach().cpu())
    return torch.cat(outs,0)


def replay_serials(cfg,ck,data,split,serials,device='cpu',batch=64):
    d=data[split];idxmap={str(s):i for i,s in enumerate(d['serials'])};idx=[idxmap[str(s)] for s in serials];m=build_model(cfg,ck,device);outs=[]
    with torch.no_grad():
        for st in range(0,len(idx),batch):
            ii=idx[st:st+batch];kw=[d[k][ii].to(device) for k in ['global_source','raw_E','coordinate_state','task_quality','task_valid']];o=m(*kw);outs.append(o['main_probabilities'].detach().cpu())
    return torch.cat(outs,0)

def binary_from_probs(probs,thresholds,lordosis_threshold=.5):
    th=[float(lordosis_threshold)]+[float(thresholds[t]) for t in range(3) for _ in range(5)]
    return [[int(float(p)>=th[i]) for i,p in enumerate(row)] for row in probs.tolist()],th
