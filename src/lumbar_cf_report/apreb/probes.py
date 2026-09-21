#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch


def move_probe_state(state, device):
    if state is None:
        return None
    return {
        "mean": state["mean"].to(device).float(),
        "scale": state["scale"].to(device).float().clamp_min(1e-6),
        "coef": state["coef"].to(device).float(),
        "intercept": torch.tensor(float(state["intercept"]), device=device),
    }


def probe_logit(state, x):
    """Differentiable sklearn-logistic probe reconstruction."""
    return (((x.float() - state["mean"]) / state["scale"]) * state["coef"]).sum(-1) + state["intercept"]


def probe_probability(state, x):
    return torch.sigmoid(probe_logit(state, x))


def prepare_probe_bundle(raw_bundle, device):
    raw = raw_bundle.get("probe_states", raw_bundle)
    return {k: move_probe_state(v, device) for k, v in raw.items() if v is not None}

