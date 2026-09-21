#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical evidence schema for the lumbar visual branch.

The schema deliberately keeps every task/level slot separate so later report
training can replace, zero, or shuffle one slot without changing unrelated
visual evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Sequence

import torch

VERSION = "Lumbar-Visual-Evidence-Branch-v1"
SCHEMA_VERSION = "lumbar_visual_evidence_schema_v1"
COMPATIBLE_SCHEMA_VERSIONS = {SCHEMA_VERSION, "stage1_16a_visual_evidence_schema_v1"}
LEVELS: List[str] = ["L1/2", "L2/3", "L3/4", "L4/5", "L5/S1"]
TASKS: List[str] = ["disc", "stenosis", "nerve"]


@dataclass(frozen=True)
class EvidenceDimensions:
    levels: int = 5
    tasks: int = 3
    expert_feature_dim: int = 768
    task_projected_dim: int = 128
    token_dim: int = 256
    level_embedding_dim: int = 16

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


REQUIRED_CACHE_KEYS = (
    "schema_version",
    "split",
    "serials",
    "labels",
    "label_valid",
    "expert_logits",
    "task_quality",
    "task_valid",
    "expert_features",
    "node_coords_raw",
    "node_coords_canonical",
    "geometry_features",
    "geometry_feature_names",
    "lordosis_label",
    "lordosis_label_valid",
    "lordosis_anchor_logit",
    "lordosis_anchor_prediction",
    "metadata",
)


def validate_cache(cache: Mapping[str, object], *, require_nonempty: bool = True) -> Dict[str, object]:
    missing = [key for key in REQUIRED_CACHE_KEYS if key not in cache]
    if missing:
        raise KeyError(f"Evidence cache missing keys: {missing}")
    if cache["schema_version"] not in COMPATIBLE_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"Schema mismatch: {cache['schema_version']} != {SCHEMA_VERSION}"
        )
    serials = torch.as_tensor(cache["serials"])
    labels = torch.as_tensor(cache["labels"])
    logits = torch.as_tensor(cache["expert_logits"])
    quality = torch.as_tensor(cache["task_quality"])
    valid = torch.as_tensor(cache["task_valid"])
    if serials.ndim != 1:
        raise ValueError(f"serials must be [N], got {tuple(serials.shape)}")
    n = int(serials.numel())
    if require_nonempty and n == 0:
        raise ValueError("Empty evidence cache")
    expected = (n, len(LEVELS), len(TASKS))
    for name, value in (
        ("labels", labels),
        ("expert_logits", logits),
        ("task_quality", quality),
        ("task_valid", valid),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")
    feature_map = cache["expert_features"]
    if not isinstance(feature_map, Mapping):
        raise TypeError("expert_features must be a mapping")
    feature_shapes = {}
    for task in TASKS:
        if task not in feature_map:
            raise KeyError(f"expert_features missing task={task}")
        tensor = torch.as_tensor(feature_map[task])
        if tensor.ndim != 3 or tuple(tensor.shape[:2]) != (n, len(LEVELS)):
            raise ValueError(
                f"expert_features[{task}] must be [N,5,D], got {tuple(tensor.shape)}"
            )
        feature_shapes[task] = list(tensor.shape)
    node_raw = torch.as_tensor(cache["node_coords_raw"])
    node_canonical = torch.as_tensor(cache["node_coords_canonical"])
    if tuple(node_raw.shape) != (n, len(LEVELS), 2):
        raise ValueError(f"node_coords_raw shape invalid: {tuple(node_raw.shape)}")
    if tuple(node_canonical.shape) != (n, len(LEVELS), 2):
        raise ValueError(
            f"node_coords_canonical shape invalid: {tuple(node_canonical.shape)}"
        )
    geometry = torch.as_tensor(cache["geometry_features"])
    if geometry.ndim != 2 or geometry.shape[0] != n:
        raise ValueError(f"geometry_features must be [N,G], got {tuple(geometry.shape)}")
    lordosis = torch.as_tensor(cache["lordosis_label"])
    if tuple(lordosis.shape) != (n,):
        raise ValueError(f"lordosis_label must be [N], got {tuple(lordosis.shape)}")
    if torch.unique(serials).numel() != n:
        raise ValueError("serials are not unique")
    return {
        "status": "valid",
        "split": str(cache["split"]),
        "cases": n,
        "feature_shapes": feature_shapes,
        "geometry_shape": list(geometry.shape),
    }


def clone_cache_tensors(cache: Mapping[str, object]) -> Dict[str, object]:
    """Clone tensors while retaining immutable metadata."""
    out: Dict[str, object] = {}
    for key, value in cache.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, Mapping):
            out[key] = {
                subkey: (subvalue.clone() if torch.is_tensor(subvalue) else subvalue)
                for subkey, subvalue in value.items()
            }
        else:
            out[key] = value
    return out

