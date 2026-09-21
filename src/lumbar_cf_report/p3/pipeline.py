
from __future__ import annotations
import argparse,json,zipfile,hashlib
from pathlib import Path
import torch

from .common import load_cfg,dump_json,sha256_file
from .backend import load_bank,load_model
from .ops import (
    load_pair_assets,load_protected_jacobian_cache,freeze_model_parameters,
    adaptive_select
)
from .evaluator import run_search,select_p3

def _state_dict(payload):
    return payload.get(
        "model_state",
        payload.get("model_state_dict",payload.get("state_dict",payload))
    )

def _tensor_hash(sd,exclude_prefix=None):
    h=hashlib.sha256()
    for n in sorted(sd):
        if exclude_prefix and n.startswith(exclude_prefix):
            continue
        t=sd[n].detach().cpu().contiguous()
        h.update(n.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def _validate_config_schema(cfg):
    for key in [
      "paths","frozen_sha256","model","data","specificity_anchor",
      "functional_search","evaluation","selection_policy","firewall"
    ]:
        if key not in cfg:
            raise RuntimeError(f"CONFIG_SCHEMA_FAIL missing top-level key: {key}")
    for key in [
      "feature_bank","frozen_checkpoint","frozen_config","frozen_model_py",
      "pair_bank","frozen_old_factual_evidence","stageA_selection",
      "protected_jacobian_cache","prior_p1_1_selection","prior_p2_selection",
      "output_root"
    ]:
        if key not in cfg["paths"]:
            raise RuntimeError(f"CONFIG_SCHEMA_FAIL missing path: {key}")

def preflight(cfg_path):
    cfg=load_cfg(cfg_path); _validate_config_schema(cfg)
    out=Path(cfg["paths"]["output_root"])
    d=out/"00_preflight"; d.mkdir(parents=True,exist_ok=True)
    errors=[]; checks={}

    for k in [
      "feature_bank","frozen_checkpoint","frozen_config","frozen_model_py",
      "pair_bank","frozen_old_factual_evidence","stageA_selection",
      "protected_jacobian_cache","prior_p1_1_selection","prior_p2_selection"
    ]:
        p=Path(cfg["paths"][k])
        checks[k]={"path":str(p),"exists":p.exists()}
        if not p.exists(): errors.append(f"missing:{k}:{p}")

    if not errors:
        for k,ek in [("frozen_checkpoint","checkpoint"),("frozen_config","config")]:
            got=sha256_file(cfg["paths"][k]); exp=cfg["frozen_sha256"][ek]
            checks[k+"_sha256"]={"got":got,"expected":exp,"match":got==exp}
            if got!=exp: errors.append(f"{k} SHA mismatch")

        bank=load_bank(cfg["paths"]["feature_bank"])
        dev=bank.indices(cfg["data"]["development_split_name"])
        val=bank.indices(cfg["data"]["internal_split_name"])
        checks["cohort_counts"]={"development":len(dev),"internal":len(val)}
        if len(dev)!=cfg["data"]["expected_development"]:
            errors.append("Development count mismatch")
        if len(val)!=cfg["data"]["expected_internal"]:
            errors.append("Internal count mismatch")

        pb,E0=load_pair_assets(cfg)
        checks["pair_bank"]={
          "train_rows":len(pb["train"]),"internal_rows":len(pb["internal"])
        }
        if len(pb["train"])!=cfg["data"]["expected_train_rows"]:
            errors.append("pair train row mismatch")
        if len(pb["internal"])!=cfg["data"]["expected_internal_rows"]:
            errors.append("pair internal row mismatch")
        checks["E0_shape"]=list(E0.shape)

        # Bind the method chain: P1.1 low-drift endpoint and P2 nonlinear NO-GO.
        p11=json.loads(
            Path(cfg["paths"]["prior_p1_1_selection"]).read_text(encoding="utf-8")
        )
        b11=p11.get("best_projected_stageA",{})
        checks["prior_p1_1"]={
          "stage_action":p11.get("stage_action"),
          "best_alpha":b11.get("alpha"),
          "decision":b11.get("decision")
        }
        if p11.get("stage_action")!=cfg["specificity_anchor"]["expected_prior_action"]:
            errors.append("Prior P1.1 action mismatch")
        if abs(float(b11.get("alpha",-999))
               -float(cfg["specificity_anchor"]["expected_prior_best_alpha"]))>1e-12:
            errors.append("Prior P1.1 best alpha mismatch")

        p2=json.loads(
            Path(cfg["paths"]["prior_p2_selection"]).read_text(encoding="utf-8")
        )
        b2=p2.get("best_target_preserving_projection",{})
        checks["prior_p2"]={
          "stage_action":p2.get("stage_action"),
          "best_alpha":b2.get("alpha"),
          "decision":b2.get("decision"),
          "mds":b2.get("mds"),"drift":b2.get("drift"),
          "target_first_order_global_retention":b2.get(
              "target_first_order_global_retention"
          )
        }
        if p2.get("stage_action")!=cfg["evaluation"]["prior_p2_required_action"]:
            errors.append("Prior P2 action mismatch")
        if abs(float(b2.get("alpha",-999))
               -float(cfg["evaluation"]["prior_p2_expected_best_alpha"]))>1e-12:
            errors.append("Prior P2 best alpha mismatch")

        pj=load_protected_jacobian_cache(cfg)
        checks["protected_jacobian"]={
          "shape":list(pj["J"].shape),
          "internal_rows":int(pj["internal_rows"])
        }
        if list(pj["J"].shape)!=[cfg["data"]["expected_internal_rows"],2,384]:
            errors.append("Protected Jacobian shape mismatch")

        sel=json.loads(Path(cfg["paths"]["stageA_selection"]).read_text(encoding="utf-8"))
        bp=sel.get("best_profile",{})
        cps=[Path(x) for x in bp.get("checkpoints",[])]
        checks["stageA"]={
          "profile":bp.get("profile"),"decision":bp.get("decision"),
          "n_checkpoints":len(cps)
        }
        if bp.get("profile")!=cfg["data"]["expected_stageA_profile"]:
            errors.append("Stage A profile mismatch")
        if len(cps)!=3:
            errors.append("Expected 3 Stage A checkpoints")

        base_payload=torch.load(
            cfg["paths"]["frozen_checkpoint"],map_location="cpu",weights_only=False
        )
        base_nonT=_tensor_hash(_state_dict(base_payload),"transporter.")
        seeds=[]
        for p in cps:
            if not p.is_file():
                errors.append(f"missing Stage A checkpoint:{p}"); continue
            ck=torch.load(p,map_location="cpu",weights_only=False)
            if _tensor_hash(_state_dict(ck),"transporter.")!=base_nonT:
                errors.append(f"Non-transporter state differs:{p}")
            seeds.append(int(ck.get("seed",-1)))
        checks["stageA_seeds"]=sorted(seeds)
        if sorted(seeds)!=sorted(cfg["data"]["expected_stageA_seeds"]):
            errors.append("Stage A seed mismatch")

        lg=[float(x) for x in cfg["functional_search"]["lambda_grid"]]
        rg=[float(x) for x in cfg["functional_search"]["adaptive_rho_grid"]]
        checks["search_grid"]={
          "lambda_n":len(lg),"lambda_min":min(lg),"lambda_max":max(lg),
          "rho_grid":rg
        }
        if lg!=sorted(lg) or abs(lg[0])>1e-12 or abs(lg[-1]-1.0)>1e-12:
            errors.append("lambda grid must be sorted and include exact 0/1 endpoints")
        if len(set(lg))!=len(lg):
            errors.append("duplicate lambda values")
        if any(not (0.0<rho<=1.0) for rho in rg):
            errors.append("rho outside (0,1]")

        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model,_=load_model(cfg,device)
        freeze_model_parameters(model)
        checks["runtime_trainable_parameters"]=sum(
            int(p.requires_grad) for p in model.parameters()
        )
        if checks["runtime_trainable_parameters"]!=0:
            errors.append("Runtime parameter firewall failed")

    result={
      "version":cfg["version"],
      "status":"PASS" if not errors else "FAIL",
      "checks":checks,"errors":errors,"firewall":cfg["firewall"]
    }
    dump_json(result,d/"P3_PRECHECK.json")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(2)

def smoke(cfg_path):
    cfg=load_cfg(cfg_path)
    root=Path(cfg["paths"]["output_root"])
    d=root/"01_smoke"; d.mkdir(parents=True,exist_ok=True)

    # Synthetic finite-step selector regression test.
    lambdas=[0.0,0.5,1.0]
    # [L,B]
    mds=torch.tensor([
      [0.010,0.000,-0.010],
      [0.032,0.018, 0.005],
      [0.040,0.020,-0.005],
    ])
    drift=torch.tensor([
      [0.002,0.001,0.001],
      [0.010,0.006,0.004],
      [0.020,0.015,0.020],
    ])
    dirs=torch.zeros(3,3,3,128)
    dirs[:,:,0,0]=1.0
    stage_mds=mds[-1]
    selected,diag=adaptive_select(
        dirs,mds,drift,lambdas,stage_mds,0.90,1e-8
    )
    # row0 requires 0.036 -> must use Stage A lambda=1
    # row1 requires 0.018 -> lambda=.5 is lower drift and feasible
    # row2 StageA MDS<0 -> requires >=0 -> lambda=.5 is feasible and lower drift
    got=[float(x) for x in diag["selected_lambda"]]
    exp=[1.0,0.5,0.5]
    if any(abs(a-b)>1e-8 for a,b in zip(got,exp)):
        raise RuntimeError(f"Adaptive selector regression failed: got={got} expected={exp}")
    result={
      "status":"P3_SMOKE_OK",
      "selected_lambda":got,
      "constraint_achieved":[bool(x) for x in diag["constraint_achieved"]],
      "fallback":[bool(x) for x in diag["fallback"]]
    }
    dump_json(result,d/"P3_SMOKE.json")
    print(result)

def pack(cfg_path):
    cfg=load_cfg(cfg_path); root=Path(cfg["paths"]["output_root"])
    z=root.parent/f"{root.name}_RESULTS.zip"
    if z.exists(): z.unlink()
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as f:
        for p in root.rglob("*"):
            if p.is_file() and not p.name.endswith(".pt"):
                f.write(p,p.relative_to(root.parent))
    print(z)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
      "command",choices=["preflight","smoke","run","select","pack"]
    )
    ap.add_argument("--config",default="configs/p3.json")
    a=ap.parse_args()
    if a.command=="preflight": preflight(a.config)
    elif a.command=="smoke": smoke(a.config)
    elif a.command=="run": run_search(a.config)
    elif a.command=="select": select_p3(a.config)
    elif a.command=="pack": pack(a.config)

if __name__=="__main__":
    main()
