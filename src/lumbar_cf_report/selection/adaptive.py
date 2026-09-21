from __future__ import annotations
import math
import numpy as np

TASK_SLICES={"lordosis":slice(0,1),"disc":slice(1,6),"stenosis":slice(6,11),"nerve":slice(11,16)}

def _rep_unique_bigram(text):
    x=list(str(text));g=[tuple(x[i:i+2]) for i in range(max(0,len(x)-1))]
    return len(set(g))/max(1,len(g))

def normalized_confidence(probs,thresholds):
    probs=np.asarray(probs,float);ths=np.asarray(thresholds,float)
    denom=np.where(probs>=ths,np.maximum(1e-6,1-ths),np.maximum(1e-6,ths))
    return np.clip(np.abs(probs-ths)/denom,0,1)

def _candidate_stats(row,cand,cfg):
    pv=np.asarray(cand["parser_vec"],int);core=np.asarray(row["core_binary"],int);valid=np.asarray(row["slot_valid"],int)>0
    conf=normalized_confidence(row["planner_probs"],row["planner_thresholds"]);mis=(pv!=core)&valid
    sn=np.zeros(16,dtype=bool);sn[6:16]=True
    dl=np.zeros(16,dtype=bool);dl[0:6]=True
    text=str(cand.get("text",""));path=int(len(text)<int(cfg["text_min_chars"]) or len(text)>int(cfg["text_max_chars"]) or _rep_unique_bigram(text)<float(cfg["repeat_unique_bigram_min"]))
    return {
        "mismatch":int(mis.sum()),
        "mismatch_conf_sum":float(conf[mis].sum()),
        "high_conf_mismatch":int((mis&(conf>=float(cfg["high_confidence"]))).sum()),
        "sn_high_conf_mismatch":int((mis&sn&(conf>=float(cfg["hard_stenosis_nerve_confidence"]))).sum()),
        "sn_mismatch_conf_sum":float(conf[mis&sn].sum()),
        "dl_low_conf_mismatch":int((mis&dl&(conf<float(cfg["low_confidence"]))).sum()),
        "text_pathology":path,
    }

def difficulty(row,base_candidates,cfg):
    if not base_candidates: raise ValueError("base_candidates is empty")
    stats=[_candidate_stats(row,c,cfg) for c in base_candidates]
    best=min(stats,key=lambda z:(z["mismatch"],z["high_conf_mismatch"],z["text_pathology"]))
    pvs=[np.asarray(c["parser_vec"],int) for c in base_candidates]
    valid=np.asarray(row["slot_valid"],int)>0
    parser_disagree=0
    if len(pvs)>=2: parser_disagree=int(((pvs[0]!=pvs[1])&valid).sum())
    core=np.asarray(row["core_binary"],int);direct=np.asarray(row.get("direct_parser_vec",[0]*16),int);conf=normalized_confidence(row["planner_probs"],row["planner_thresholds"])
    planner_direct_high=int(((core!=direct)&valid&(conf>=float(cfg["high_confidence"]))).sum())
    both_unresolved=int(len(stats)>=2 and all(z["mismatch"]>=2 for z in stats[:2]))
    pathology=max(z["text_pathology"] for z in stats)
    dl_only=int(best["mismatch"]>0 and best["sn_high_conf_mismatch"]==0 and best["high_conf_mismatch"]==0 and best["dl_low_conf_mismatch"]==best["mismatch"])
    w=cfg["score_weights"]
    score=(float(w["best_core_mismatch"])*best["mismatch"]+
           float(w.get("best_mismatch_conf_sum",0))*best["mismatch_conf_sum"]+
           float(w["best_high_conf_mismatch"])*best["high_conf_mismatch"]+
           float(w["best_stenosis_nerve_high_conf_mismatch"])*best["sn_high_conf_mismatch"]+
           float(w.get("best_stenosis_nerve_mismatch_conf_sum",0))*best["sn_mismatch_conf_sum"]+
           float(w["raw_final_parser_disagreement"])*parser_disagree+
           float(w["planner_direct_high_conf_disagreement"])*planner_direct_high+
           float(w["both_candidates_unresolved"])*both_unresolved+
           float(w["text_pathology"])*pathology+
           float(w["disc_lordosis_low_conf_only_discount"])*dl_only)
    exact_healthy=(best["mismatch"]==0 and pathology==0)
    hard_override=(best["sn_high_conf_mismatch"]>0 or (best["high_conf_mismatch"]>=2 and best["mismatch"]>=2))
    return {
        "score":float(score),"exact_core_and_healthy":bool(exact_healthy),"hard_override":bool(hard_override),
        "best_core_mismatch":best["mismatch"],"best_mismatch_conf_sum":best["mismatch_conf_sum"],"best_high_conf_mismatch":best["high_conf_mismatch"],
        "best_stenosis_nerve_high_conf_mismatch":best["sn_high_conf_mismatch"],"best_stenosis_nerve_mismatch_conf_sum":best["sn_mismatch_conf_sum"],"raw_final_parser_disagreement":parser_disagree,
        "planner_direct_high_conf_disagreement":planner_direct_high,"both_candidates_unresolved":both_unresolved,
        "text_pathology":pathology,"disc_lordosis_low_conf_only":dl_only
    }

def calibrate_threshold(metas,cfg):
    if not metas: raise ValueError("no difficulty metas")
    scores=np.asarray([float(m["score"]) for m in metas],float)
    n=len(scores);target=max(int(cfg.get("minimum_development_expand_cases",0)),int(round(float(cfg["development_target_expand_fraction"])*n)))
    cap=max(1,int(math.floor(float(cfg["maximum_development_expand_fraction"])*n)))
    target=min(max(1,target),cap,n)
    order=np.argsort(-scores,kind="stable");cut=float(scores[order[target-1]])
    # Tie-safe threshold uses >=; report realized count so the run can audit any tie inflation.
    realized=int((scores>=cut).sum())
    return {"threshold":cut,"target_cases":target,"realized_cases_by_score":realized,"cases":n,"target_fraction":target/n,"realized_fraction_by_score":realized/n}

def should_expand(meta,threshold,cfg):
    if bool(cfg.get("always_easy_if_exact_core_and_healthy",True)) and meta.get("exact_core_and_healthy",False): return False
    if meta.get("hard_override",False): return True
    return float(meta["score"])>=float(threshold)
