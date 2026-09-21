from __future__ import annotations
import importlib,torch

def load_frozen_planner(checkpoint_path,device='cpu'):
    from ..clinical.planner import ExplicitClinicalPlanner
    cls=ExplicitClinicalPlanner
    try:ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    except TypeError:ck=torch.load(checkpoint_path,map_location='cpu')
    cfg=ck['config'];model=cls(int(cfg['global_dim']),int(cfg['task_dim']),int(cfg['coordinate_dim']),hidden_size=int(cfg['hidden_size']),projector_hidden_dim=int(cfg['projector_hidden_dim']),plan_state_hidden=int(cfg['plan_state_hidden']),dropout=float(cfg['dropout']))
    state=ck.get('state_dict') or ck.get('model_state_dict');model.load_state_dict(state,strict=True);model.to(device).eval();return model,ck

def disease_logits(main):return torch.stack([main[:,1:6],main[:,6:11],main[:,11:16]],dim=2)

@torch.inference_mode()
def run_logits(model,data,evidence,device='cpu',batch_size=32):
    g=torch.as_tensor(data['global_source']).float();c=torch.as_tensor(data['coordinate_state']).float();q=torch.as_tensor(data['task_quality']).float();v=torch.as_tensor(data['task_valid']).float();e=torch.as_tensor(evidence).float();outs=[]
    for st in range(0,len(g),batch_size):
        en=min(len(g),st+batch_size);o=model(g[st:en].to(device),e[st:en].to(device),c[st:en].to(device),q[st:en].to(device),v[st:en].to(device));outs.append(o['main_logits'].detach().cpu())
    return disease_logits(torch.cat(outs,0))
