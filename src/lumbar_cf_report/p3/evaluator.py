
from __future__ import annotations
from pathlib import Path
import csv,json
import numpy as np
import torch

from .common import TASKS,load_cfg,dump_json,sha256_file
from .backend import load_bank,load_model
from .ops import (
    load_pair_assets,load_protected_jacobian_cache,move_bank_to_device,
    freeze_model_parameters,encode_pair,ordinary_project_direction,
    blend_direction,functional_mds_drift,evaluate_direction,geometry_cosine,
    manifold_metrics,adaptive_select,write_csv
)

def _load_stageA_checkpoints(cfg):
    x=json.loads(Path(cfg["paths"]["stageA_selection"]).read_text(encoding="utf-8"))
    bp=x.get("best_profile",{})
    if bp.get("profile")!=cfg["data"]["expected_stageA_profile"]:
        raise RuntimeError(f"Stage A best profile mismatch: {bp.get('profile')}")
    cps=[Path(p) for p in bp.get("checkpoints",[])]
    if len(cps)!=3:
        raise RuntimeError(f"Expected 3 Stage A checkpoints, found {len(cps)}")
    info=[]
    for p in cps:
        if not p.is_file(): raise FileNotFoundError(p)
        ck=torch.load(p,map_location="cpu",weights_only=False)
        if ck.get("stage")!="A" or ck.get("profile")!="empirical":
            raise RuntimeError(f"Unexpected Stage A checkpoint metadata: {p}")
        info.append({"path":p,"seed":int(ck["seed"]),"epoch":int(ck["epoch"])})
    got=sorted(x["seed"] for x in info)
    exp=sorted(int(x) for x in cfg["data"]["expected_stageA_seeds"])
    if got!=exp:
        raise RuntimeError(f"Stage A seeds mismatch: got={got} expected={exp}")
    return sorted(info,key=lambda z:z["seed"])

def _safe_mean(vals):
    xs=[]
    for x in vals:
        if x in (None,""): continue
        try: y=float(x)
        except (TypeError,ValueError): continue
        if np.isfinite(y): xs.append(y)
    return float(np.mean(xs)) if xs else None

def _record(records,seed,method,param_name,param_value,rows,cos,func,cent,near,diag=None):
    beta=0.25
    for j,r in enumerate(rows):
        mds=float(func[beta]["mds"][j])
        drift=float(func[beta]["drift"][j])
        rec={
          "seed":seed,"method":method,
          "parameter_name":param_name,"parameter_value":param_value,
          "recipient_serial":r["recipient_serial"],
          "level":r["level_name"],"task":r["task_name"],
          "cosine":float(cos[j]),
          "mds":mds,"mdc":float(func[beta]["mdc"][j]),
          "drift":drift,"cps":mds/(drift+1e-8),
          "centroid_distance":float(cent[j]),
          "nearest_distance":float(near[j]),
        }
        for b in [0.10,0.25,0.50]:
            rec[f"mds_{b}"]=float(func[b]["mds"][j])
            rec[f"mdc_{b}"]=float(func[b]["mdc"][j])
            rec[f"drift_{b}"]=float(func[b]["drift"][j])
        if diag is not None:
            for k,v in diag.items():
                if torch.is_tensor(v):
                    vv=v[j]
                    rec[k]=bool(vv) if vv.dtype==torch.bool else float(vv)
                else:
                    rec[k]=v
        records.append(rec)

def _evaluate_and_record(records,seed,method,param_name,param_value,direction,
                         model,factual,cr,rows,E0,betas,mbeta,eps,diag=None):
    cos=geometry_cosine(direction,rows,factual.device,eps).detach().cpu()
    func=evaluate_direction(model,direction,factual,cr,rows,betas,eps)
    func={b:{k:v.detach().cpu() for k,v in x.items()} for b,x in func.items()}
    cent,near=manifold_metrics(E0,direction.detach().cpu(),rows,mbeta,eps)
    if diag is not None:
        diag={k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in diag.items()}
    _record(
        records,seed,method,param_name,param_value,rows,cos,func,cent,near,diag
    )

def run_search(cfg_path):
    cfg=load_cfg(cfg_path)
    root=Path(cfg["paths"]["output_root"])
    out=root/"02_functional_search"
    out.mkdir(parents=True,exist_ok=True)

    pb,E0=load_pair_assets(cfg)
    rows=pb["internal"]
    J=load_protected_jacobian_cache(cfg)["J"]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank=move_bank_to_device(load_bank(cfg["paths"]["feature_bank"]),device)
    base,_=load_model(cfg,device); base=freeze_model_parameters(base)

    checkpoints=_load_stageA_checkpoints(cfg)
    eps=float(cfg["functional_search"]["eps"])
    beta_sel=float(cfg["functional_search"]["selection_beta"])
    betas=[float(x) for x in cfg["evaluation"]["betas"]]
    mbeta=float(cfg["evaluation"]["manifold_beta"])
    lambda_grid=[float(x) for x in cfg["functional_search"]["lambda_grid"]]
    rhos=[float(x) for x in cfg["functional_search"]["adaptive_rho_grid"]]
    spec_alpha=float(cfg["specificity_anchor"]["alpha"])

    with torch.no_grad():
        *_,cr,cd,qr,qd,fa,fb,factual,old_endpoint=encode_pair(
            base,bank,rows,device,transport=True
        )
        *_,direct_endpoint=encode_pair(
            base,bank,rows,device,transport=False
        )

    factual_e0=torch.stack([
        E0[int(r["recipient_index"]),int(r["level"])] for r in rows
    ],0).to(device)
    max_e0_diff=float((factual-factual_e0).abs().max())
    if max_e0_diff>1e-5:
        raise RuntimeError(
            f"Finite-step factual anchor mismatch: max |factual-E0|={max_e0_diff}"
        )

    old_dir=old_endpoint-factual
    direct_dir=direct_endpoint-factual
    Jdev=J.to(device)

    records=[]
    _evaluate_and_record(
        records,-1,"old_ccmrt",None,None,old_dir,
        base,factual,cr,rows,E0,betas,mbeta,eps
    )
    _evaluate_and_record(
        records,-1,"direct",None,None,direct_dir,
        base,factual,cr,rows,E0,betas,mbeta,eps
    )

    for info in checkpoints:
        model,_=load_model(cfg,device,checkpoint_override=str(info["path"]))
        model=freeze_model_parameters(model)
        with torch.no_grad():
            *_,crA,cdA,qrA,qdA,faA,fbA,factualA,stage_endpoint=encode_pair(
                model,bank,rows,device,transport=True
            )
        factual_diff=float((factualA-factual).abs().max())
        if factual_diff>1e-5:
            raise RuntimeError(
                f"Stage A factual evidence changed: max diff={factual_diff}"
            )
        stage_dir=stage_endpoint-factualA
        spec_dir=ordinary_project_direction(stage_dir,Jdev,spec_alpha,eps)

        _evaluate_and_record(
            records,info["seed"],"stageA",None,None,stage_dir,
            base,factual,cr,rows,E0,betas,mbeta,eps
        )
        _evaluate_and_record(
            records,info["seed"],"specificity_anchor","alpha",spec_alpha,spec_dir,
            base,factual,cr,rows,E0,betas,mbeta,eps
        )

        # Build the entire one-dimensional path once for this Stage-A seed.
        dirs=[]
        mds_list=[]; drift_list=[]
        for lam in lambda_grid:
            d=blend_direction(spec_dir,stage_dir,lam,eps)
            mds,drift=functional_mds_drift(
                base,d,factual,cr,rows,beta_sel,eps
            )
            dirs.append(d)
            mds_list.append(mds)
            drift_list.append(drift)

            # Global lambda candidates are a first-class, simpler candidate family.
            _evaluate_and_record(
                records,info["seed"],"global_path","lambda",lam,d,
                base,factual,cr,rows,E0,betas,mbeta,eps,
                diag={
                  "selected_lambda":torch.full(
                      (len(rows),),float(lam),device=device
                  )
                }
            )

        dirs=torch.stack(dirs,0)          # [L,B,3,128]
        mds_grid=torch.stack(mds_list,0)  # [L,B]
        drift_grid=torch.stack(drift_list,0)

        # Stage-A actual finite-step response is lambda=1 endpoint.
        stage_mds,_=functional_mds_drift(
            base,stage_dir,factual,cr,rows,beta_sel,eps
        )

        for rho in rhos:
            selected,diag=adaptive_select(
                dirs,mds_grid,drift_grid,lambda_grid,stage_mds,rho,eps
            )
            diag["rho"]=float(rho)
            _evaluate_and_record(
                records,info["seed"],"adaptive_constraint","rho",rho,selected,
                base,factual,cr,rows,E0,betas,mbeta,eps,diag=diag
            )

        print(f"[P3] seed={info['seed']} complete")

    write_csv(records,out/"P3_CASE_LEVEL_TASK_METRICS.csv")

    # Compact seed/task summaries.
    groups={}
    for r in records:
        key=(r["seed"],r["method"],r["parameter_name"],r["parameter_value"])
        groups.setdefault(key,[]).append(r)

    def summarize(xs):
        keys=[
          "cosine","mds","mdc","drift","cps","centroid_distance","nearest_distance",
          "selected_lambda","required_mds","search_mds","search_drift",
          "mds_retention"
        ]
        out={}
        for k in keys:
            m=_safe_mean([x.get(k,None) for x in xs])
            if m is not None: out[k+"_mean"]=m
        for k in ["any_feasible","constraint_achieved","fallback"]:
            vals=[x.get(k,None) for x in xs if x.get(k,None) not in (None,"")]
            if vals:
                out[k+"_fraction"]=float(np.mean([
                    1.0 if str(v).lower() in {"true","1","1.0"} else 0.0
                    for v in vals
                ]))
        return out

    seed_rows=[]; task_rows=[]
    for (seed,method,pn,pv),xs in groups.items():
        seed_rows.append({
          "seed":seed,"method":method,"parameter_name":pn,
          "parameter_value":pv,"n":len(xs),**summarize(xs)
        })
        for task in TASKS:
            ys=[x for x in xs if x["task"]==task]
            task_rows.append({
              "seed":seed,"method":method,"parameter_name":pn,
              "parameter_value":pv,"task":task,"n":len(ys),**summarize(ys)
            })
    write_csv(seed_rows,out/"P3_SEED_SUMMARY.csv")
    write_csv(task_rows,out/"P3_TASK_SUMMARY.csv")

    dump_json({
      "stageA_checkpoints":[
        {"seed":x["seed"],"epoch":x["epoch"],"path":str(x["path"])}
        for x in checkpoints
      ],
      "specificity_anchor_alpha":spec_alpha,
      "lambda_grid":lambda_grid,
      "adaptive_rho_grid":rhos,
      "selection_beta":beta_sel,
      "n_internal_rows":len(rows),
      "max_e0_factual_diff":max_e0_diff,
      "protected_jacobian_shape":list(J.shape),
      "pair_bank_sha256":sha256_file(cfg["paths"]["pair_bank"]),
      "frozen_checkpoint_sha256":sha256_file(cfg["paths"]["frozen_checkpoint"]),
      "protected_jacobian_sha256":sha256_file(cfg["paths"]["protected_jacobian_cache"]),
    },out/"P3_RUN_MANIFEST.json")
    print(out)

def _aggregate(rows,method,param_name=None,param_value=None):
    xs=[r for r in rows if r["method"]==method]
    if param_name is not None:
        xs=[r for r in xs if r.get("parameter_name","")==param_name]
    if param_value is not None:
        xs=[
          r for r in xs
          if r.get("parameter_value","") not in ("",None)
          and abs(float(r["parameter_value"])-float(param_value))<1e-12
        ]
    if not xs: return None

    by={}
    for r in xs:
        by.setdefault(int(r["seed"]),[]).append(r)

    seed_summ=[]
    for seed,ys in by.items():
        d={
          "seed":seed,
          "cosine":_safe_mean([y["cosine"] for y in ys]),
          "mds":_safe_mean([y["mds"] for y in ys]),
          "mdc":_safe_mean([y["mdc"] for y in ys]),
          "drift":_safe_mean([y["drift"] for y in ys]),
          "cps":_safe_mean([y["cps"] for y in ys]),
          "centroid_distance":_safe_mean([y["centroid_distance"] for y in ys]),
          "nearest_distance":_safe_mean([y["nearest_distance"] for y in ys]),
          "selected_lambda":_safe_mean([y.get("selected_lambda",None) for y in ys]),
          "constraint_achieved_fraction":None,
          "fallback_fraction":None,
        }
        for src,dst in [
            ("constraint_achieved","constraint_achieved_fraction"),
            ("fallback","fallback_fraction")
        ]:
            vals=[y.get(src,None) for y in ys if y.get(src,None) not in (None,"")]
            if vals:
                d[dst]=float(np.mean([
                    1.0 if str(v).lower() in {"true","1","1.0"} else 0.0
                    for v in vals
                ]))
        seed_summ.append(d)

    out={"n_seeds":len(seed_summ)}
    for k in [
      "cosine","mds","mdc","drift","cps","centroid_distance","nearest_distance",
      "selected_lambda","constraint_achieved_fraction","fallback_fraction"
    ]:
        vals=[x[k] for x in seed_summ if x[k] is not None]
        if vals: out[k]=float(np.mean(vals))
    return out

def _task_repairs(rows,method,param_name,param_value,old):
    repairs={}
    for task in ["stenosis","nerve"]:
        xs=[
          r for r in rows if r["method"]==method and r["task"]==task
          and (param_name is None or r.get("parameter_name","")==param_name)
          and (param_value is None or (
              r.get("parameter_value","") not in ("",None)
              and abs(float(r["parameter_value"])-float(param_value))<1e-12
          ))
        ]
        ox=[r for r in rows if r["method"]=="old_ccmrt" and r["task"]==task]
        repairs[task]={
          "delta_cos":_safe_mean([x["cosine"] for x in xs])
                      -_safe_mean([x["cosine"] for x in ox]),
          "delta_mds":_safe_mean([x["mds"] for x in xs])
                      -_safe_mean([x["mds"] for x in ox]),
        }
    return repairs

def select_p3(cfg_path):
    cfg=load_cfg(cfg_path)
    root=Path(cfg["paths"]["output_root"])
    p=root/"02_functional_search"/"P3_CASE_LEVEL_TASK_METRICS.csv"
    if not p.is_file(): raise FileNotFoundError(p)
    with p.open("r",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))

    old=_aggregate(rows,"old_ccmrt")
    direct=_aggregate(rows,"direct")
    stage=_aggregate(rows,"stageA")
    spec=_aggregate(
        rows,"specificity_anchor","alpha",
        float(cfg["specificity_anchor"]["alpha"])
    )

    sg=cfg["evaluation"]["strong_go"]; gg=cfg["evaluation"]["go"]

    def assess(method,param_name,param_value,family):
        agg=_aggregate(rows,method,param_name,param_value)
        repairs=_task_repairs(rows,method,param_name,param_value,old)
        task_ok=all(
            v["delta_cos"]>0 and v["delta_mds"]>0 for v in repairs.values()
        )
        strong=(
            agg["cosine"]>=sg["cos_min"] and
            agg["mds"]>=sg["mds_min"] and
            agg["mdc"]>=sg["mdc_min"] and
            agg["drift"]<=sg["drift_max"] and task_ok
        )
        go=(
            agg["cosine"]>=gg["cos_min"] and
            agg["mds"]>=gg["mds_min"] and
            agg["mdc"]>=gg["mdc_min"] and
            agg["drift"]<=gg["drift_max"] and task_ok
        )
        orient=(agg["cosine"]-old["cosine"])/(direct["cosine"]-old["cosine"]+1e-8)
        mdsprog=(agg["mds"]-old["mds"])/(direct["mds"]-old["mds"]+1e-8)
        specpres=(direct["drift"]-agg["drift"])/(direct["drift"]-old["drift"]+1e-8)
        score=.35*orient+.40*mdsprog+.25*specpres
        out={
          "family":family,"method":method,
          "parameter_name":param_name,"parameter_value":param_value,
          **agg,
          "orientation_progress":orient,
          "mds_progress":mdsprog,
          "specificity_preservation":specpres,
          "mds_retention_vs_stageA":agg["mds"]/(stage["mds"]+1e-8),
          "drift_retention_vs_stageA":agg["drift"]/(stage["drift"]+1e-8),
          "task_repairs":repairs,
          "decision":"STRONG_GO" if strong else ("GO" if go else "FAIL"),
          "pareto_score":score,
        }
        return out

    global_candidates=[
        assess("global_path","lambda",float(lam),"global")
        for lam in cfg["functional_search"]["lambda_grid"]
    ]
    adaptive_candidates=[
        assess("adaptive_constraint","rho",float(rho),"adaptive")
        for rho in cfg["functional_search"]["adaptive_rho_grid"]
    ]
    allc=global_candidates+adaptive_candidates

    # Highest gate tier first. Within equal tier, prefer global fixed lambda.
    # Score only tie-breaks after scientific gate and simplicity.
    rank={"STRONG_GO":2,"GO":1,"FAIL":0}
    simplicity={"global":1,"adaptive":0}
    allc.sort(
        key=lambda x:(
            rank[x["decision"]],
            simplicity[x["family"]],
            x["pareto_score"]
        ),
        reverse=True
    )
    best=allc[0]

    if best["decision"] in {"STRONG_GO","GO"}:
        action=(
            "FREEZE_STAGE1_21_P3_GLOBAL"
            if best["family"]=="global"
            else "FREEZE_STAGE1_21_P3_ADAPTIVE"
        )
    else:
        action=cfg["selection_policy"]["if_all_fail"]

    result={
      "stage":"Stage1.21-P3",
      "old_ccmrt":old,
      "direct":direct,
      "stageA_seed_mean":stage,
      "specificity_anchor_seed_mean":spec,
      "best_candidate":best,
      "global_candidates":global_candidates,
      "adaptive_candidates":adaptive_candidates,
      "stage_action":action,
      "selection_rule":cfg["selection_policy"]["primary_rule"],
      "interpretation":(
          "P3 evaluates finite-step frozen-head function directly at beta=0.25; "
          "no first-order gradient proxy is used for candidate selection."
      )
    }
    d=root/"03_summary"; d.mkdir(parents=True,exist_ok=True)
    dump_json(result,d/"P3_SELECTION.json")
    write_csv(global_candidates,d/"P3_GLOBAL_PATH.csv")
    write_csv(adaptive_candidates,d/"P3_ADAPTIVE_PROFILES.csv")
    print(action)
    print(best)
    return result
