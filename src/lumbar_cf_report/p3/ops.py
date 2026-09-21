
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import torch

from .common import normalize, cosine
from .backend import extract_endpoint, extract_encoded_factual, state_logits

def load_pair_assets(cfg):
    pb=torch.load(cfg["paths"]["pair_bank"],map_location="cpu",weights_only=False)
    e0=torch.load(
        cfg["paths"]["frozen_old_factual_evidence"],
        map_location="cpu",weights_only=False
    )["E0"]
    return pb,e0

def load_protected_jacobian_cache(cfg):
    from .common import sha256_file
    p=Path(cfg["paths"]["protected_jacobian_cache"])
    if not p.is_file():
        raise FileNotFoundError(f"Missing P1 protected Jacobian cache: {p}")
    x=torch.load(p,map_location="cpu",weights_only=False)
    if x["pair_bank_sha256"]!=sha256_file(cfg["paths"]["pair_bank"]):
        raise RuntimeError("Protected Jacobian pair-bank SHA mismatch")
    if x["frozen_checkpoint_sha256"]!=sha256_file(cfg["paths"]["frozen_checkpoint"]):
        raise RuntimeError("Protected Jacobian frozen-checkpoint SHA mismatch")
    return x

def move_bank_to_device(bank,device):
    for k in ["X","E","C","y","quality","valid"]:
        setattr(bank,k,getattr(bank,k).to(device))
    return bank

def freeze_model_parameters(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bad=[n for n,p in model.named_parameters() if p.requires_grad]
    if bad:
        raise RuntimeError(f"Training-free firewall violated: {bad[:20]}")
    return model

def index_tensor(rows,key,device,dtype=torch.long):
    return torch.tensor([r[key] for r in rows],device=device,dtype=dtype)

def encode_pair(model,bank,rows,device,transport=True):
    ri=index_tensor(rows,"recipient_index",device)
    di=index_tensor(rows,"donor_index",device)
    li=index_tensor(rows,"level",device)
    ti=index_tensor(rows,"task",device)
    xr=bank.X[ri,li]; er=bank.E[ri,li]; cr=bank.C[ri,li]
    xd=bank.X[di,li]; ed=bank.E[di,li]; cd=bank.C[di,li]
    qr=bank.quality[ri,li]; qd=bank.quality[di,li]
    fa=model.encode_pair_level(xr,er,cr)
    fb=model.encode_pair_level(xd,ed,cd)
    alpha=torch.ones(len(rows),device=device)
    cf=model.counterfactual_from_encoded(
        fa,fb,ti,alpha,qa=qr,qb=qd,transport=transport
    )
    endpoint=extract_endpoint(cf)
    factual=extract_encoded_factual(fa)
    return ri,di,li,ti,cr,cd,qr,qd,fa,fb,factual,endpoint

def ordinary_project_direction(d_full,J,alpha,eps):
    """Reconstruct the exact P1.1 specificity anchor."""
    b=d_full.shape[0]
    d=normalize(d_full.reshape(b,-1),eps)
    JJt=torch.bmm(J,J.transpose(1,2))
    rhs=torch.bmm(J,d.unsqueeze(-1))
    if float(alpha)==0.0:
        coef=torch.bmm(torch.linalg.pinv(JJt),rhs)
    else:
        scale=JJt.diagonal(dim1=-2,dim2=-1).sum(-1)/2.0
        lam=float(alpha)*scale+eps
        eye=torch.eye(2,device=d.device,dtype=d.dtype).unsqueeze(0)
        coef=torch.linalg.solve(JJt+lam[:,None,None]*eye,rhs)
    removed=torch.bmm(J.transpose(1,2),coef).squeeze(-1)
    proj=d-removed
    return normalize(proj,eps).reshape(b,3,128)

def blend_direction(d_spec,d_stage,lam,eps):
    """Finite-step path candidate. lam=0 => specificity anchor; lam=1 => Stage A."""
    if not (0.0 <= float(lam) <= 1.0):
        raise ValueError(f"lambda outside [0,1]: {lam}")
    b=d_stage.shape[0]
    a=normalize(d_spec.reshape(b,-1),eps)
    z=normalize(d_stage.reshape(b,-1),eps)
    mix=(1.0-float(lam))*a+float(lam)*z
    return normalize(mix,eps).reshape(b,3,128)

def functional_mds_drift(model,direction,factual,cr,rows,beta,eps):
    b=len(rows)
    u=normalize(direction.reshape(b,-1),eps).reshape(b,3,128)
    z0=state_logits(model,factual[:,None],cr[:,None])[:,0]
    scales=torch.tensor(
        [float(r["local_scale"]) for r in rows],
        device=factual.device,dtype=factual.dtype
    )
    estar=factual+float(beta)*scales[:,None,None]*u
    z1=state_logits(model,estar[:,None],cr[:,None])[:,0]
    ar=torch.arange(b,device=factual.device)
    ti=torch.tensor([int(r["task"]) for r in rows],device=factual.device)
    sign=torch.tensor(
        [1.0 if int(r["minority_state"])==1 else -1.0 for r in rows],
        device=factual.device,dtype=factual.dtype
    )
    mds=sign*(z1[ar,ti]-z0[ar,ti])
    mask=torch.ones_like(z0); mask[ar,ti]=0
    drift=(torch.abs(z1-z0)*mask).sum(-1)/2.0
    return mds,drift

def evaluate_direction(model,direction,factual,cr,rows,betas,eps):
    out={}
    for beta in betas:
        mds,drift=functional_mds_drift(
            model,direction,factual,cr,rows,float(beta),eps
        )
        out[float(beta)]={
            "mds":mds,
            "mdc":(mds>0).float(),
            "drift":drift,
        }
    return out

def geometry_cosine(direction,rows,device,eps):
    b=len(rows)
    ti=torch.tensor([int(r["task"]) for r in rows],device=device)
    ar=torch.arange(b,device=device)
    target=direction[ar,ti]
    uemp=torch.stack([
        torch.as_tensor(r["u_emp"]).float() for r in rows
    ]).to(device)
    return cosine(target,uemp,eps)

def manifold_metrics(E0,direction,rows,beta,eps):
    b=len(rows)
    u=normalize(direction.reshape(b,-1),eps).reshape(b,3,128).cpu()
    centroid=[]; nearest=[]
    for j,r in enumerate(rows):
        i=int(r["recipient_index"]); l=int(r["level"]); t=int(r["task"])
        neigh=[int(x) for x in r["neighbor_indices"]]
        base=E0[i,l,t]
        step=base+float(beta)*float(r["local_scale"])*u[j,t]
        pool=E0[neigh,l,t]
        mu=pool.mean(0)
        centroid.append(float((step-mu).norm()))
        nearest.append(float((pool-step[None]).norm(dim=-1).min()))
    return torch.tensor(centroid),torch.tensor(nearest)

def adaptive_select(candidate_dirs,mds_grid,drift_grid,lambda_grid,stage_mds,rho,eps):
    """
    candidate_dirs: [L,B,3,128]
    mds_grid/drift_grid: [L,B]
    Select minimum finite-step drift satisfying the per-row Stage-A MDS retention constraint.
    """
    if candidate_dirs.ndim!=4:
        raise RuntimeError(f"candidate_dirs shape must be [L,B,3,128], got {tuple(candidate_dirs.shape)}")
    L,B=candidate_dirs.shape[:2]
    if mds_grid.shape!=(L,B) or drift_grid.shape!=(L,B):
        raise RuntimeError("functional grid shape mismatch")
    if len(lambda_grid)!=L:
        raise RuntimeError("lambda grid length mismatch")

    required=torch.where(
        stage_mds>0,
        float(rho)*stage_mds,
        torch.zeros_like(stage_mds)
    )
    feasible=mds_grid>=required[None,:]

    masked=torch.where(
        feasible,
        drift_grid,
        torch.full_like(drift_grid,float("inf"))
    )
    idx=masked.argmin(dim=0)
    any_feasible=feasible.any(dim=0)

    # lambda=1 is the exact Stage-A endpoint and is the frozen fallback.
    try:
        fallback_idx=min(
            range(L),key=lambda i:abs(float(lambda_grid[i])-1.0)
        )
    except Exception as e:
        raise RuntimeError("Could not resolve Stage-A fallback lambda") from e
    if abs(float(lambda_grid[fallback_idx])-1.0)>1e-9:
        raise RuntimeError("lambda grid must include 1.0 Stage-A fallback")

    idx=torch.where(
        any_feasible,
        idx,
        torch.full_like(idx,int(fallback_idx))
    )
    ar=torch.arange(B,device=candidate_dirs.device)
    selected=candidate_dirs[idx,ar]
    selected_mds=mds_grid[idx,ar]
    selected_drift=drift_grid[idx,ar]
    lam_tensor=torch.tensor(
        [float(x) for x in lambda_grid],
        device=candidate_dirs.device,
        dtype=selected_mds.dtype
    )
    selected_lambda=lam_tensor[idx]
    achieved=selected_mds>=required-eps

    return selected,{
        "selected_index":idx,
        "selected_lambda":selected_lambda,
        "required_mds":required,
        "search_mds":selected_mds,
        "search_drift":selected_drift,
        "any_feasible":any_feasible,
        "constraint_achieved":achieved,
        "fallback":~any_feasible,
        "mds_retention":selected_mds/(stage_mds.abs()+eps),
    }

def write_csv(rows,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)
