from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from lumbar_cf_report.selection.eval_bridge import load_parser
from lumbar_cf_report.selection.metrics import bleu_one, case_counts, eval_texts, rouge_one


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cases(out: Path, pool_path: Path, parser_mod):
    lab_path = out / "05_selection_pool/SELECTION50_EVAL_LABELS.jsonl"
    if not pool_path.is_file():
        raise FileNotFoundError(pool_path)
    if not lab_path.is_file():
        raise FileNotFoundError(lab_path)
    pool = read_jsonl(pool_path)
    labs = {str(r["serial"]): r for r in read_jsonl(lab_path)}
    if len(pool) != 50 or len(labs) != 50:
        raise RuntimeError(f"Expected selection50 pool/labels: pool={len(pool)} labels={len(labs)}")
    cases = []
    for row in pool:
        s = str(row["serial"])
        lab = labs.get(s)
        if lab is None:
            raise RuntimeError(f"Missing label serial={s}")
        ref = str(lab["reference_raw"])
        gt = np.asarray(lab["slot_labels"], dtype=int).reshape(-1)[:16]
        valid = np.asarray(lab["slot_valid"], dtype=int).reshape(-1)[:16]
        core = np.asarray(row["core_binary"], dtype=int).reshape(-1)[:16]
        cands = []
        for j, cand in enumerate(row.get("candidates", [])):
            pv = np.asarray(cand["parser_vec"], dtype=int).reshape(-1)[:16]
            tp, fp, fn, tn = case_counts(pv, gt, valid)
            cands.append({
                "index": j,
                "tag": str(cand.get("tag", f"candidate{j}")),
                "text": str(cand["text"]),
                "parser_vec": pv.tolist(),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "gt_pos": int(tp + fn),
                "rouge": float(rouge_one(parser_mod, cand["text"], ref)),
                "bleu": float(bleu_one(cand["text"], ref)),
            })
        if not cands:
            raise RuntimeError(f"No candidates serial={s}")
        cases.append({
            "serial": s,
            "reference_raw": ref,
            "slot_labels": gt.tolist(),
            "slot_valid": valid.tolist(),
            "core_binary": core.tolist(),
            "candidates": cands,
        })
    return cases


def state_shape(cases):
    gp = sum(sum(np.asarray(c["slot_labels"], int) * (np.asarray(c["slot_valid"], int) > 0)) for c in cases)
    neg = sum(sum((np.asarray(c["slot_labels"], int) == 0) * (np.asarray(c["slot_valid"], int) > 0)) for c in cases)
    return int(gp), int(neg)


def dp_optimize(cases, objective, f1_floor=None, balanced_weights=(0.45, 0.55)):
    gp, neg = state_shape(cases)
    wr, wb = balanced_weights
    dp = np.full((gp + 1, neg + 1), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    choices = []

    def val(c):
        if objective == "rouge": return c["rouge"]
        if objective == "bleu": return c["bleu"]
        if objective == "balanced": return wr * c["rouge"] + wb * c["bleu"]
        raise ValueError(objective)

    for case in cases:
        new = np.full_like(dp, -np.inf)
        ch = np.full(dp.shape, -1, dtype=np.int16)
        for j, cand in enumerate(case["candidates"]):
            dtp, dfp = int(cand["tp"]), int(cand["fp"])
            src = dp[: gp + 1 - dtp, : neg + 1 - dfp]
            dst = new[dtp:, dfp:]
            scores = src + val(cand)
            mask = scores > dst + 1e-15
            dst[mask] = scores[mask]
            chv = ch[dtp:, dfp:]
            chv[mask] = j
        dp = new
        choices.append(ch)

    finite = np.isfinite(dp)
    tp_grid = np.arange(gp + 1, dtype=np.float64)[:, None]
    fp_grid = np.arange(neg + 1, dtype=np.float64)[None, :]
    f1 = np.where((tp_grid + fp_grid + gp) > 0, 2.0 * tp_grid / (tp_grid + fp_grid + gp), 0.0)
    feasible = finite.copy()
    if f1_floor is not None:
        feasible &= f1 + 1e-12 >= float(f1_floor)
    if not feasible.any():
        return None
    masked = np.where(feasible, dp, -np.inf)
    flat = int(np.argmax(masked))
    tp, fp = np.unravel_index(flat, dp.shape)
    picks = [None] * len(cases)
    ctp, cfp = int(tp), int(fp)
    for i in range(len(cases) - 1, -1, -1):
        j = int(choices[i][ctp, cfp])
        if j < 0:
            raise RuntimeError(f"DP backtrack failure i={i} state={(ctp,cfp)}")
        picks[i] = j
        cand = cases[i]["candidates"][j]
        ctp -= int(cand["tp"]); cfp -= int(cand["fp"])
    if ctp != 0 or cfp != 0:
        raise RuntimeError("DP backtrack origin mismatch")
    return {"picks": picks, "surrogate_objective": float(masked[tp, fp])}


def evaluate(cases, picks, parser_mod):
    texts=[]; refs=[]; gt=[]; valid=[]; core=[]; dist=Counter(); rows=[]
    tp=fp=fn=0
    for case, j in zip(cases, picks):
        cand = case["candidates"][int(j)]
        texts.append(cand["text"]); refs.append(case["reference_raw"]); gt.append(case["slot_labels"]); valid.append(case["slot_valid"]); core.append(case["core_binary"])
        dist[cand["tag"]] += 1
        tp += cand["tp"]; fp += cand["fp"]; fn += cand["fn"]
        rows.append({"serial":case["serial"],"tag":cand["tag"],"candidate_index":int(j),"candidate_rouge":cand["rouge"],"candidate_bleu":cand["bleu"],"candidate_fact_counts":{"tp":cand["tp"],"fp":cand["fp"],"fn":cand["fn"]}})
    m = eval_texts(parser_mod,texts,refs,gt,valid,core)
    m.update({"tag_distribution":dict(dist),"micro_counts":{"tp":int(tp),"fp":int(fp),"fn":int(fn),"gt_pos":int(tp+fn)}})
    return m, rows


def max_clinical(cases, parser_mod, bw=(0.45,0.55)):
    gp, neg = state_shape(cases); wr, wb = bw
    dp=np.full((gp+1,neg+1),-np.inf);dp[0,0]=0.0;choices=[]
    for case in cases:
        new=np.full_like(dp,-np.inf);ch=np.full(dp.shape,-1,dtype=np.int16)
        for j,cand in enumerate(case["candidates"]):
            dtp,dfp=int(cand["tp"]),int(cand["fp"]);src=dp[:gp+1-dtp,:neg+1-dfp];dst=new[dtp:,dfp:]
            sc=src+wr*cand["rouge"]+wb*cand["bleu"];mask=sc>dst+1e-15;dst[mask]=sc[mask];cv=ch[dtp:,dfp:];cv[mask]=j
        dp=new;choices.append(ch)
    finite=np.isfinite(dp);tpg=np.arange(gp+1)[:,None].astype(float);fpg=np.arange(neg+1)[None,:].astype(float);f1=2*tpg/(tpg+fpg+gp)
    mf=float(np.max(np.where(finite,f1,-1)));states=finite&(np.abs(f1-mf)<=1e-12);masked=np.where(states,dp,-np.inf);flat=int(np.argmax(masked));tp,fp=np.unravel_index(flat,dp.shape)
    picks=[None]*len(cases);ctp,cfp=int(tp),int(fp)
    for i in range(len(cases)-1,-1,-1):
        j=int(choices[i][ctp,cfp]);picks[i]=j;cand=cases[i]["candidates"][j];ctp-=int(cand["tp"]);cfp-=int(cand["fp"])
    return evaluate(cases,picks,parser_mod)


def exact_joint_milp(cases, parser_mod, targets):
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as e:
        return {"available":False,"feasible":None,"reason":repr(e)}
    offsets=[];flat=[];nvar=0
    for i,case in enumerate(cases):
        ids=[]
        for j,cand in enumerate(case["candidates"]):
            ids.append(nvar);flat.append((i,j,cand));nvar+=1
        offsets.append(ids)
    gp,_=state_shape(cases);ncase=len(cases);A=lil_matrix((ncase+3,nvar),dtype=float);lb=np.full(ncase+3,-np.inf);ub=np.full(ncase+3,np.inf)
    for i,ids in enumerate(offsets):A[i,ids]=1.0;lb[i]=1.0;ub[i]=1.0
    tf,tr,tb=float(targets["clinical_f1"]),float(targets["rouge_l"]),float(targets["bleu4"])
    for k,(_,_,cand) in enumerate(flat):
        A[ncase,k]=(2.0-tf)*cand["tp"]-tf*cand["fp"];A[ncase+1,k]=cand["rouge"];A[ncase+2,k]=cand["bleu"]
    lb[ncase]=tf*gp;lb[ncase+1]=tr*ncase;lb[ncase+2]=tb*ncase
    c=np.zeros(nvar)
    for k,(_,_,cand) in enumerate(flat):c[k]=-(0.45*cand["rouge"]+0.55*cand["bleu"])
    try:
        res=milp(c=c,integrality=np.ones(nvar,dtype=np.int8),bounds=Bounds(np.zeros(nvar),np.ones(nvar)),constraints=LinearConstraint(A.tocsr(),lb,ub),options={"time_limit":90.0,"mip_rel_gap":0.0})
    except Exception as e:
        return {"available":True,"feasible":None,"solver_error":repr(e)}
    if res.x is None:
        return {"available":True,"feasible":False,"solver_status":int(res.status),"solver_message":str(res.message)}
    picks=[]
    for ids in offsets:
        picks.append(int(np.argmax([res.x[k] for k in ids])))
    m,rows=evaluate(cases,picks,parser_mod);ok=m["clinical_f1"]+1e-12>=tf and m["rouge_l"]+1e-12>=tr and m["bleu4"]+1e-12>=tb
    return {"available":True,"feasible":bool(ok),"solver_status":int(res.status),"solver_message":str(res.message),"metrics":m,"rows":rows}


def run_floor(cases, parser_mod, floor):
    out={}
    for name,obj in [("max_rouge","rouge"),("max_bleu","bleu"),("balanced","balanced")]:
        d=dp_optimize(cases,obj,floor,(0.45,0.55))
        if d is None:out[name]={"feasible":False}
        else:
            m,_=evaluate(cases,d["picks"],parser_mod);out[name]={"feasible":True,"metrics":m}
    return out


def selftest():
    cases=[
        {"slot_labels":[1],"slot_valid":[1],"candidates":[{"tp":1,"fp":0,"rouge":0.4,"bleu":0.4},{"tp":0,"fp":0,"rouge":0.9,"bleu":0.9}]},
        {"slot_labels":[0],"slot_valid":[1],"candidates":[{"tp":0,"fp":0,"rouge":0.5,"bleu":0.5},{"tp":0,"fp":1,"rouge":0.95,"bleu":0.95}]},
    ]
    d=dp_optimize(cases,"balanced",0.70,(0.45,0.55))
    if d is None or d["picks"] != [0,0]:raise AssertionError(d)
    print("EXPANDED_SELECTION50_ORACLE_SELFTEST_OK")


def main():
    import os
    out=Path(os.environ['OUT']).resolve();pool=out/'05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl'
    cfg=read_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));parser_mod=load_parser(cfg['parser']['module']);cases=load_cases(out,pool,parser_mod)
    floors={}
    for floor in [0.70,0.71,0.72]:floors[f'f1_{int(round(floor*100)):02d}']=run_floor(cases,parser_mod,floor)
    mc,mcrows=max_clinical(cases,parser_mod);targets={'clinical_f1':0.70,'rouge_l':0.55,'bleu4':0.45};joint=exact_joint_milp(cases,parser_mod,targets)
    stats={'cases':50,'mean_candidates':float(np.mean([len(c['candidates']) for c in cases])),'min_candidates':int(min(len(c['candidates']) for c in cases)),'max_candidates':int(max(len(c['candidates']) for c in cases)),'tag_counts':dict(Counter(x['tag'] for c in cases for x in c['candidates']))}
    rep={'status':'PASS','diagnostic_only':True,'pool':stats,'targets':targets,'max_clinical_ceiling':mc,'floor_sweep_070_071_072':floors,'joint_gate_exact_milp':{k:v for k,v in joint.items() if k!='rows'},'generator_calls':0,'reranker_training':False,'holdout_accessed':False,'independent49_accessed':False}
    od=out/'06_selection_oracle_audit';write_json(rep,od/'SELECTION50_S4_ORACLE_CEILING.json');write_jsonl(mcrows,od/'MAX_CLINICAL_SELECTIONS.jsonl')
    if joint.get('rows'):write_jsonl(joint['rows'],od/'JOINT_GATE_SELECTIONS.jsonl')
    print(json.dumps(rep,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
