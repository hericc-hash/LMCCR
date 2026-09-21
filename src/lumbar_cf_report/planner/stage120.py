from __future__ import annotations
import importlib.util,torch
from pathlib import Path
KWARGS=dict(x_dim=256,c_dim=47,task_dim=128,a_dim=96,r_dim=48,task_embedding_dim=24,ar_hidden=256,hidden=256,dropout=0.1,transport_hidden=96,transport_rank=6,transport_mode='reversible_cayley',use_coordinate_conditioning=True,identity_tau=2.0,transport_rotation_scale=0.12,transport_strength_scale=0.15)

def load_frozen(model_py,checkpoint_path,device='cpu'):
    spec=importlib.util.spec_from_file_location('stage23b_frozen_stage120_model',Path(model_py));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=getattr(mod,'Stage120JointModel')(**KWARGS)
    try:ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    except TypeError:ck=torch.load(checkpoint_path,map_location='cpu')
    state=ck.get('model_state') or ck.get('state_dict') or ck.get('model_state_dict') or ck.get('model');model.load_state_dict(state,strict=True);model.to(device).eval();return model,ck

@torch.inference_mode()
def reconstruct(model,X,E,C,device='cpu',batch_size=32):
    X=torch.as_tensor(X).float();E=torch.as_tensor(E).float();C=torch.as_tensor(C).float();out=[]
    for st in range(0,len(X),batch_size):
        en=min(len(X),st+batch_size);o=model(X[st:en].to(device),E[st:en].to(device),C[st:en].to(device));out.append(o['reconstructed_task_features'].detach().cpu())
    return torch.cat(out,0)
