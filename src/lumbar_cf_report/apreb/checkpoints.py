#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import torch
from .model import AnatomyPathologyResidualBottleneck
from .anatomy_probe import IndependentAnatomyProbe


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _ar_from_config(c):
    return AnatomyPathologyResidualBottleneck(
        c["segment_dim"], c["task_dim"], c["coordinate_dim"],
        c["anatomy_dim"], c["residual_dim"], c["task_embedding_dim"],
        c["coordinate_latent_dim"], c["hidden_dim"], c["dropout"]
    )


def load_ar_checkpoint(path):
    obj = safe_torch_load(path)
    if "config" not in obj or "state_dict" not in obj:
        raise RuntimeError(f"invalid A/R checkpoint: {path}")
    model = _ar_from_config(obj["config"])
    model.load_state_dict(obj["state_dict"], strict=True)
    return model, obj


def load_init_ar_checkpoint(path):
    """Load an earlier compatible A/R checkpoint into the self-contained v3 architecture.

    Only the checkpoint tensor contract is used; no earlier-version Python module is imported.
    """
    return load_ar_checkpoint(path)


def save_ar_checkpoint(out_dir, model, config, metrics):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "apreb.pt"
    torch.save({
        "config": config,
        "metrics": metrics,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, p)
    return p


def load_anatomy_probe(path):
    obj = safe_torch_load(path)
    c = obj["config"]
    model = IndependentAnatomyProbe(
        c["task_dim"], c["segment_dim"], c["coordinate_dim"],
        c["task_embedding_dim"], c["hidden_dim"], c["dropout"]
    )
    model.load_state_dict(obj["state_dict"], strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, obj

