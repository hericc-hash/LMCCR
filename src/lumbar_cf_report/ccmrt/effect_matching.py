from __future__ import annotations

from typing import Dict, List
import numpy as np


def effect_match(curves: Dict[str, List[Dict]], x_key="mean_signed_target_shift",
                 y_keys=("mean_anatomy_drift", "mean_cross_task_drift"), n_points=5):
    """Interpolate each method on a common target-effect grid.

    This avoids claiming a method is more preserving merely because it produces a
    weaker target intervention.
    """
    valid = {}
    for name, rows in curves.items():
        rs = [r for r in rows if r.get(x_key) is not None and np.isfinite(r[x_key])]
        rs = sorted(rs, key=lambda r: r[x_key])
        if len(rs) >= 2:
            valid[name] = rs
    if len(valid) < 2:
        return {"status": "insufficient", "matched": []}
    lo = max(min(r[x_key] for r in rs) for rs in valid.values())
    hi = min(max(r[x_key] for r in rs) for rs in valid.values())
    if hi <= lo:
        return {"status": "no_overlap", "matched": []}
    grid = np.linspace(lo, hi, int(n_points))
    matched = []
    for g in grid:
        row = {"target_effect": float(g)}
        for name, rs in valid.items():
            xs = np.asarray([r[x_key] for r in rs], dtype=float)
            for yk in y_keys:
                ys = np.asarray([r[yk] for r in rs], dtype=float)
                row[f"{name}:{yk}"] = float(np.interp(g, xs, ys))
        matched.append(row)
    return {"status": "ok", "effect_range": [float(lo), float(hi)], "matched": matched}

