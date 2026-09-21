from __future__ import annotations
from typing import Dict
import math
import torch
import torch.nn.functional as F


def cosine_loss(a, b):
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def _bce(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y)


def factual_loss(model, batch, weights: Dict[str, float]):
    out = model(batch["x"], batch["e"], batch["c"])

    task_bce = _bce(out["state_logits"], batch["y"])
    residual_bce = _bce(out["pathology_logits"], batch["y"])

    e_rec = F.smooth_l1_loss(out["reconstructed_task_features"], batch["e"])
    e_cos = cosine_loss(out["reconstructed_task_features"], batch["e"])
    x_rec = F.smooth_l1_loss(out["xhat"], batch["x"])
    x_cos = cosine_loss(out["xhat"], batch["x"])

    seg_rec = F.smooth_l1_loss(out["reconstructed_segment_base"], batch["x"])
    coord_rec = F.smooth_l1_loss(out["reconstructed_coordinate"], batch["c"])
    probe_a = F.smooth_l1_loss(out["ahat_probe"], out["anatomy"].detach())
    probe_c = F.smooth_l1_loss(out["chat_probe"], batch["c"])

    legacy_a = torch.zeros((), device=batch["x"].device)
    legacy_r = torch.zeros_like(legacy_a)
    if "legacy_a" in batch:
        legacy_a = F.smooth_l1_loss(out["anatomy"], batch["legacy_a"])
    if "legacy_r" in batch:
        legacy_r = F.smooth_l1_loss(out["residual"], batch["legacy_r"])

    r_norm = torch.linalg.vector_norm(out["residual"], dim=-1).mean() / math.sqrt(out["residual"].shape[-1])

    total = (
        weights.get("task_bce", 1.0) * task_bce
        + weights.get("residual_bce", 0.5) * residual_bce
        + weights.get("e_rec", 1.0) * e_rec
        + weights.get("e_cos", 0.5) * e_cos
        + weights.get("x_rec", 1.0) * x_rec
        + weights.get("x_cos", 1.0) * x_cos
        + weights.get("segment_rec", 0.25) * seg_rec
        + weights.get("coordinate_rec", 0.15) * coord_rec
        + weights.get("probe_a", 0.15) * probe_a
        + weights.get("probe_c", 0.15) * probe_c
        + weights.get("legacy_a", 0.0) * legacy_a
        + weights.get("legacy_r", 0.0) * legacy_r
        + weights.get("residual_norm", 1e-4) * r_norm
    )

    parts = {
        "task_bce": task_bce.detach(),
        "residual_bce": residual_bce.detach(),
        "e_rec": e_rec.detach(),
        "e_cos": e_cos.detach(),
        "x_rec": x_rec.detach(),
        "x_cos": x_cos.detach(),
        "segment_rec": seg_rec.detach(),
        "coordinate_rec": coord_rec.detach(),
        "probe_a": probe_a.detach(),
        "probe_c": probe_c.detach(),
        "legacy_a": legacy_a.detach(),
        "legacy_r": legacy_r.detach(),
        "r_norm": r_norm.detach(),
    }
    return total, parts, out


def _pair_factual_anchor(fa, fb, batch, weights):
    totals = []
    e_cos_vals = []
    x_cos_vals = []
    e_rec_vals = []
    x_rec_vals = []

    for f, prefix in [(fa, "a"), (fb, "b")]:
        x = batch["x" + prefix]
        e = batch["e" + prefix]
        c = batch["c" + prefix]
        y = batch["y" + prefix]

        e_rec = F.smooth_l1_loss(f["reconstructed_task_features"], e)
        e_cos = cosine_loss(f["reconstructed_task_features"], e)
        x_rec = F.smooth_l1_loss(f["xhat"], x)
        x_cos = cosine_loss(f["xhat"], x)

        total = (
            weights.get("task_bce", 1.0) * _bce(f["state_logits"], y)
            + weights.get("residual_bce", 0.5) * _bce(f["pathology_logits"], y)
            + weights.get("e_rec", 1.0) * e_rec
            + weights.get("e_cos", 0.5) * e_cos
            + weights.get("x_rec", 0.75) * x_rec
            + weights.get("x_cos", 1.0) * x_cos
            + weights.get("segment_rec", 0.2) * F.smooth_l1_loss(f["reconstructed_segment_base"], x)
            + weights.get("coordinate_rec", 0.1) * F.smooth_l1_loss(f["reconstructed_coordinate"], c)
        )
        totals.append(total)
        e_cos_vals.append(e_cos)
        x_cos_vals.append(x_cos)
        e_rec_vals.append(e_rec)
        x_rec_vals.append(x_rec)

    diag = {
        "factual_e_cos": torch.stack(e_cos_vals).mean(),
        "factual_x_cos": torch.stack(x_cos_vals).mean(),
        "factual_e_rec": torch.stack(e_rec_vals).mean(),
        "factual_x_rec": torch.stack(x_rec_vals).mean(),
    }
    return sum(totals) / 2.0, diag


def intervention_loss(
    model,
    batch,
    weights: Dict[str, float],
    factual_anchor_weights: Dict[str, float],
    target_margin=0.03,
    effect_floor_ratio=0.95,
    monotonic_alphas=(0.25, 0.5, 0.75, 1.0),
    monotonic_margin=0.0,
):
    fa = model.encode_pair_level(batch["xa"], batch["ea"], batch["ca"])
    fb = model.encode_pair_level(batch["xb"], batch["eb"], batch["cb"])

    t = batch["task_index"].long()
    ids = torch.arange(t.shape[0], device=t.device)
    alpha = batch["alpha"]

    direct = model.counterfactual_from_encoded(
        fa, fb, t, alpha, qa=batch.get("qa"), qb=batch.get("qb"), transport=False
    )
    trans = model.counterfactual_from_encoded(
        fa, fb, t, alpha, qa=batch.get("qa"), qb=batch.get("qb"), transport=True
    )

    p0 = torch.sigmoid(fa["state_logits"])
    pd = torch.sigmoid(direct["state_logits"])
    pt = torch.sigmoid(trans["state_logits"])

    ya = batch["ya"][ids, t]
    yb = batch["yb"][ids, t]
    sign = torch.where(yb > ya, torch.ones_like(ya), -torch.ones_like(ya))

    direct_signed = sign * (pd[ids, t] - p0[ids, t])
    trans_signed = sign * (pt[ids, t] - p0[ids, t])

    target_hinge = F.relu(float(target_margin) * alpha - trans_signed).mean()
    floor = float(effect_floor_ratio) * F.relu(direct_signed.detach())
    effect_floor = F.relu(floor - trans_signed).mean()

    donor_bce = F.binary_cross_entropy_with_logits(trans["state_logits"][ids, t], yb)
    endpoint = model.counterfactual_from_encoded(
        fa,
        fb,
        t,
        torch.ones_like(alpha),
        qa=batch.get("qa"),
        qb=batch.get("qb"),
        transport=True,
    )
    endpoint_bce = F.binary_cross_entropy_with_logits(endpoint["state_logits"][ids, t], yb)

    mask = torch.ones_like(p0, dtype=torch.bool)
    mask[ids, t] = False
    off = (pt[mask] - p0[mask]).abs().mean()

    anatomy = F.smooth_l1_loss(trans["ahat_probe"], fa["ahat_probe"].detach())
    coordinate = F.smooth_l1_loss(trans["chat_probe"], fa["chat_probe"].detach())

    adjust = (
        torch.linalg.vector_norm(trans["r_target"] - trans["rb_target"], dim=-1).mean()
        / math.sqrt(trans["r_target"].shape[-1])
    )

    # Structural identity / cycle audit. With reversible_cayley these should be nearly numerical zero.
    rb = trans["rb_target"]
    qb = batch.get("qb")
    qa = batch.get("qa")
    qb_t = qb[ids, t] if qb is not None and qb.ndim == 2 else qb
    qa_t = qa[ids, t] if qa is not None and qa.ndim == 2 else qa

    rid = model.transporter(rb, batch["cb"], batch["cb"], t, qa=qb_t, qb=qb_t)
    identity = F.smooth_l1_loss(rid, rb)

    rba = trans["r_target"]
    rbab = model.transporter(rba, batch["ca"], batch["cb"], t, qa=qb_t, qb=qa_t)
    cycle = F.smooth_l1_loss(rbab, rb)

    mono_vals = []
    for a in monotonic_alphas:
        oi = model.counterfactual_from_encoded(
            fa,
            fb,
            t,
            torch.full_like(alpha, float(a)),
            qa=batch.get("qa"),
            qb=batch.get("qb"),
            transport=True,
        )
        pi = torch.sigmoid(oi["state_logits"])
        mono_vals.append(sign * (pi[ids, t] - p0[ids, t]))

    monotonic = torch.zeros((), device=t.device)
    for a, b in zip(mono_vals[:-1], mono_vals[1:]):
        monotonic += F.relu(a + float(monotonic_margin) - b).mean()
    monotonic /= max(1, len(mono_vals) - 1)

    factual_anchor, factual_diag = _pair_factual_anchor(fa, fb, batch, factual_anchor_weights)

    total = (
        weights.get("target_hinge", 1.0) * target_hinge
        + weights.get("effect_floor", 2.0) * effect_floor
        + weights.get("donor_bce", 0.15) * donor_bce
        + weights.get("endpoint_bce", 0.40) * endpoint_bce
        + weights.get("anatomy", 1.75) * anatomy
        + weights.get("coordinate", 0.35) * coordinate
        + weights.get("off_target", 1.1) * off
        + weights.get("transport_adjustment", 0.01) * adjust
        + weights.get("cycle", 0.10) * cycle
        + weights.get("identity", 0.02) * identity
        + weights.get("monotonic", 1.0) * monotonic
        + weights.get("factual_anchor", 1.5) * factual_anchor
    )

    parts = {
        "target_hinge": target_hinge.detach(),
        "effect_floor": effect_floor.detach(),
        "donor_bce": donor_bce.detach(),
        "endpoint_bce": endpoint_bce.detach(),
        "anatomy": anatomy.detach(),
        "coordinate": coordinate.detach(),
        "off_target": off.detach(),
        "transport_adjustment": adjust.detach(),
        "identity": identity.detach(),
        "cycle": cycle.detach(),
        "monotonic": monotonic.detach(),
        "factual_anchor": factual_anchor.detach(),
        "direct_signed": direct_signed.detach().mean(),
        "transport_signed": trans_signed.detach().mean(),
        **{k: v.detach() for k, v in factual_diag.items()},
    }
    return total, parts

