from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def aggregate_factual(rows: List[Dict]) -> Dict:
    by_slot = defaultdict(lambda: {"y": [], "p": []})
    for r in rows:
        for t in range(3):
            by_slot[(r["level_index"], t)]["y"].append(r[f"y{t}"])
            by_slot[(r["level_index"], t)]["p"].append(r[f"p{t}"])
    aucs = []
    per_slot = {}
    for k, v in sorted(by_slot.items()):
        auc = safe_auc(v["y"], v["p"])
        per_slot[f"{k[0]}:{k[1]}"] = auc
        if auc is not None:
            aucs.append(auc)
    return {"macro_auc": float(np.mean(aucs)) if aucs else None, "per_slot_auc": per_slot}


def _mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else None


def aggregate_cf(rows: List[Dict]) -> Dict:
    return {
        "pairs": len(rows),
        "recipients": len(set(r["recipient_id"] for r in rows)),
        "directional_accuracy": _mean(rows, "direction_correct"),
        "mean_signed_target_shift": _mean(rows, "signed_target_shift"),
        "mean_abs_target_shift": _mean(rows, "abs_target_shift"),
        "mean_cross_task_drift": _mean(rows, "cross_task_drift"),
        "mean_anatomy_drift": _mean(rows, "anatomy_drift"),
        "mean_coordinate_drift": _mean(rows, "coordinate_drift"),
        "mean_coordinate_distance": _mean(rows, "coordinate_distance"),
        "mean_quality_distance": _mean(rows, "quality_distance"),
        "mean_residual_displacement": _mean(rows, "residual_displacement"),
        "mean_cycle_error": _mean(rows, "cycle_error"),
        "mean_identity_error": _mean(rows, "identity_error"),
    }


def patient_cluster_bootstrap(rows: List[Dict], metric_key: str, n_boot=5000, seed=20260823):
    rng = np.random.default_rng(seed)
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["recipient_id"]].append(r)
    pats = list(by_patient)
    if not pats:
        return {"estimate": None, "ci95": [None, None], "n_patients": 0}
    vals0 = [float(r[metric_key]) for r in rows if r.get(metric_key) is not None]
    estimate = float(np.mean(vals0)) if vals0 else None
    boots = []
    for _ in range(int(n_boot)):
        sample = rng.choice(pats, size=len(pats), replace=True)
        vals = []
        for p in sample:
            vals.extend(float(r[metric_key]) for r in by_patient[p] if r.get(metric_key) is not None)
        if vals:
            boots.append(float(np.mean(vals)))
    if not boots:
        return {"estimate": estimate, "ci95": [None, None], "n_patients": len(pats)}
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"estimate": estimate, "ci95": [float(lo), float(hi)], "n_patients": len(pats)}


def trajectory_monotonicity(rows: List[Dict], tol: float = 1e-8) -> Dict:
    """Pair-level monotonicity over alpha for signed target effects.

    A trajectory passes when signed_target_shift never decreases by more than tol as alpha grows.
    This mirrors the APREB multi-strength logic while remaining threshold-independent.
    """
    groups = defaultdict(list)
    for r in rows:
        key = (r["method"], r["recipient_id"], r["donor_id"], r["level_index"], r["task_index"])
        groups[key].append(r)
    by_method = defaultdict(list)
    for key, rs in groups.items():
        method = key[0]
        rs = sorted(rs, key=lambda x: float(x["alpha"]))
        vals = [float(x["signed_target_shift"]) for x in rs]
        ok = all(vals[i+1] + tol >= vals[i] for i in range(len(vals)-1)) if len(vals) >= 2 else False
        by_method[method].append(float(ok))
    return {
        m: {"fraction_monotonic": float(np.mean(v)) if v else None, "n_trajectories": len(v)}
        for m, v in by_method.items()
    }

