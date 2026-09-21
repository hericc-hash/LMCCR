#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lumbar_cf_report.ccmrt.config import deep_update,parse_overrides
from lumbar_cf_report.ccmrt.data import FeatureBank,FactualCaseDataset
from lumbar_cf_report.ccmrt.factory import make_model
from lumbar_cf_report.ccmrt.io import load_yaml,set_seed,ensure_dir,to_device,append_jsonl,save_json
from lumbar_cf_report.ccmrt.losses import factual_loss,intervention_loss
from lumbar_cf_report.ccmrt.pairs import build_pair_rows,CounterfactualPairDataset


def collate(batch):
    out={}
    for k in batch[0]:
        v=[x[k] for x in batch]
        if torch.is_tensor(v[0]): out[k]=torch.stack(v)
        elif isinstance(v[0],(int,float)): out[k]=torch.tensor(v)
        else: out[k]=v
    return out

def mean_parts(ps):
    if not ps:return {}
    return {k:float(np.mean([float(p[k]) for p in ps])) for k in ps[0]}

def save_ckpt(model,cfg,path,phase,epoch,metrics,init_info):
    torch.save({"stage":"CCMRT","version":"CCMRT-Reversible-v1","phase":phase,"epoch":epoch,"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "config":cfg,"metrics":metrics,"init_info":init_info},path)

def unique_params(params):
    seen=set(); out=[]
    for p in params:
        if id(p) not in seen: seen.add(id(p)); out.append(p)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--override",action="append",default=[]); args=ap.parse_args()
    cfg=deep_update(load_yaml(args.config),parse_overrides(args.override)); set_seed(int(cfg["seed"])); out=ensure_dir(cfg["paths"]["output_root"]); ckdir=ensure_dir(out/"checkpoints")
    log=out/"train_log.jsonl"; 
    if log.exists() and cfg["training"].get("overwrite_log",True): log.unlink()
    bank=FeatureBank(cfg["paths"]["feature_bank_dev"]); model=make_model(bank,cfg)
    init_info={"mode":str(cfg["training"].get("init_mode","pretrained")),"checkpoint":cfg["paths"].get("ar_init_checkpoint","")}
    if init_info["mode"].lower()=="pretrained":
        p=str(init_info["checkpoint"])
        if not p or not Path(p).exists(): raise FileNotFoundError(f"APREB initialization checkpoint not found: {p}")
        obj=model.load_apreb_checkpoint(p,strict=True); init_info["initial_metrics"]=obj.get("metrics",{})
        print(f"[CCMRT] A/R initialized from {p}; parameters remain trainable")
    elif init_info["mode"].lower()=="scratch": print("[CCMRT] A/R random initialization")
    else: raise ValueError("training.init_mode must be pretrained or scratch")
    dev=torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("device","cuda")!="cpu" else "cpu"); model.to(dev)
    fact_ds=FactualCaseDataset(bank,cfg["splits"]["train"]); fact_loader=DataLoader(fact_ds,batch_size=int(cfg["training"]["factual_batch_size"]),shuffle=True,num_workers=int(cfg["training"]["num_workers"]),collate_fn=collate)

    # Phase A: factual warmup. A/R is NOT frozen; guard/composer learns a stable audit interface.
    model.set_factual_warmup_trainable()
    ar_params=unique_params(list(model.ar.parameters())); guard_params=unique_params([p for n,p in model.named_parameters() if not n.startswith("ar.") and not n.startswith("transporter.")])
    opt=torch.optim.AdamW([{"params":ar_params,"lr":float(cfg["training"]["factual_ar_lr"])},{"params":guard_params,"lr":float(cfg["training"]["factual_guard_lr"])}],weight_decay=float(cfg["training"]["weight_decay"]))
    best=float("inf")
    for ep in range(1,int(cfg["training"]["factual_epochs"])+1):
        model.train(); ls=[]; ps=[]
        for b in fact_loader:
            b=to_device(b,dev); loss,parts,_=factual_loss(model,b,cfg["loss"]["factual"]); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(unique_params(ar_params+guard_params),float(cfg["training"]["grad_clip"])); opt.step(); ls.append(float(loss.detach())); ps.append({k:float(v.cpu()) for k,v in parts.items()})
        rec={"phase":"factual_joint_warmup","epoch":ep,"loss":float(np.mean(ls)),"parts":mean_parts(ps)}; append_jsonl(rec,log); print(f"[CCMRT factual {ep:03d}] loss={rec['loss']:.5f} parts={rec['parts']}")
        if rec["loss"]<best: best=rec["loss"]; save_ckpt(model,cfg,ckdir/"factual_warmup.pt","factual",ep,rec,init_info)

    best_obj=torch.load(ckdir/"factual_warmup.pt",map_location=dev,weights_only=False); model.load_state_dict(best_obj["model_state"])
    # Phase B: A/R + Transport joint learning. Only audit guard/composer is frozen.
    model.set_joint_trainable()
    residual=unique_params(model.residual_parameters()); anatomy=unique_params(model.anatomy_parameters()); trans=unique_params(list(model.transporter.parameters()))
    # verify parameter groups do not overlap
    ids=[id(p) for p in residual+anatomy+trans]
    if len(ids)!=len(set(ids)): raise RuntimeError("optimizer parameter groups overlap")
    optj=torch.optim.AdamW([{"params":residual,"lr":float(cfg["training"]["joint_residual_lr"])},{"params":anatomy,"lr":float(cfg["training"]["joint_anatomy_lr"])},{"params":trans,"lr":float(cfg["training"]["transport_lr"])}],weight_decay=float(cfg["training"]["weight_decay"]))
    pairs=build_pair_rows(bank,recipient_splits=cfg["splits"]["train"],donor_splits=cfg["splits"]["donor"],quality_weight=float(cfg["donor_matching"]["quality_weight"]),topk=int(cfg["donor_matching"]["topk"]),deterministic_nearest=False,max_pairs_per_slot=int(cfg["donor_matching"]["train_pairs_per_slot"]),seed=int(cfg["seed"]))
    pds=CounterfactualPairDataset(bank,pairs,alphas=cfg["training"]["alphas"],seed=int(cfg["seed"])); pl=DataLoader(pds,batch_size=int(cfg["training"]["pair_batch_size"]),shuffle=True,num_workers=int(cfg["training"]["num_workers"]),collate_fn=collate)
    best_score=-1e9
    for ep in range(1,int(cfg["training"]["joint_epochs"])+1):
        model.train(); ls=[]; ps=[]
        for b in pl:
            b=to_device(b,dev); loss,parts=intervention_loss(model,b,cfg["loss"]["intervention"],cfg["loss"]["joint_factual_anchor"],target_margin=float(cfg["training"]["target_margin"]),effect_floor_ratio=float(cfg["training"]["effect_floor_ratio"]),monotonic_alphas=tuple(cfg["training"]["alphas"]),monotonic_margin=float(cfg["training"].get("monotonic_margin",0.0)))
            optj.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(residual+anatomy+trans,float(cfg["training"]["grad_clip"])); optj.step()
            ls.append(float(loss.detach())); ps.append({k:float(v.cpu()) for k,v in parts.items()})
        mp=mean_parts(ps); score=float(mp.get("transport_signed",0)-1.75*mp.get("anatomy",0)-1.10*mp.get("off_target",0)-0.75*mp.get("effect_floor",0)-0.20*mp.get("cycle",0)-0.50*mp.get("monotonic",0)-0.25*mp.get("factual_anchor",0)-0.75*mp.get("factual_x_cos",0)-0.25*mp.get("factual_e_cos",0))
        rec={"phase":"joint_ar_transport","epoch":ep,"loss":float(np.mean(ls)),"selection_score":score,"parts":mp}; append_jsonl(rec,log); print(f"[CCMRT joint {ep:03d}] loss={rec['loss']:.5f} score={score:.5f} parts={mp}")
        if score>best_score: best_score=score; save_ckpt(model,cfg,ckdir/"ccmrt.pt","joint",ep,rec,init_info)
    summary={"status":"finished","stage":"CCMRT","branch":"Joint Coordinate-Conditioned Minimal Residual Transport + Factual Preservation + Reversible Cayley","upstream_frozen":"Frozen segment, evidence, and coordinate source tensors",
             "trainable_joint":["A/R coordinate+anatomy path (low LR)","A/R residual path","A/R pathology head","ResidualTransporter"],"frozen_during_joint":["evidence composer","state task heads","anatomy/coordinate audit probes"],
             "init_info":init_info,"best_factual_loss":best,"best_joint_training_score":best_score,"n_training_pairs":len(pairs),"checkpoints":{"factual":str(ckdir/"factual_warmup.pt"),"joint":str(ckdir/"ccmrt.pt")}}
    save_json(summary,out/"train_summary.json"); print(f"[CCMRT] finished -> {out}")
if __name__=="__main__": main()


