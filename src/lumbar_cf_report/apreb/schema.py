#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Mapping, Tuple
import torch

VERSION = "APREB-Multi-Strength-Path-Consistent-v1"
DATASET_VERSION = "APREB-Dataset-Contract-v1"
SOURCE_BANK_VERSION = "Rich-Semantic-Token-Bank-v1"
LEVELS: List[str] = ["L1/2", "L2/3", "L3/4", "L4/5", "L5/S1"]
TASKS: List[str] = ["disc", "stenosis", "nerve"]
TASK_ZH: Dict[str, str] = {
    "disc": "椎间盘异常",
    "stenosis": "椎管狭窄",
    "nerve": "神经受压",
}
SLOT_NAMES = [f"{task}:{level}" for task in TASKS for level in LEVELS]


def slot_index(level_i: int, task_i: int) -> int:
    return int(task_i) * len(LEVELS) + int(level_i)


def slot_to_level_task(index: int) -> Tuple[int, int]:
    index = int(index)
    return index % len(LEVELS), index // len(LEVELS)


def build_disease_labels(bank: Mapping[str, object]):
    y = torch.as_tensor(bank["labels"]).float()
    v = torch.as_tensor(bank.get("label_valid", torch.ones_like(y))).float()
    if y.ndim != 3 or tuple(y.shape[1:]) != (5, 3):
        raise ValueError(f"labels must be [N,5,3], got {tuple(y.shape)}")
    if v.shape != y.shape:
        raise ValueError("label_valid shape mismatch")
    return y, v


def validate_dataset(payload: Mapping[str, object]):
    required = [
        "serials", "segment_base", "task_features", "coordinate_state",
        "coordinate_state_raw", "task_quality", "task_valid", "labels", "label_valid",
        "global_source", "coordinate_feature_names",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"missing dataset keys: {missing}")
    serials = torch.as_tensor(payload["serials"]).long()
    n = int(serials.numel())
    if n == 0 or torch.unique(serials).numel() != n:
        raise ValueError("empty or duplicate serials")
    s = torch.as_tensor(payload["segment_base"])
    x = torch.as_tensor(payload["task_features"])
    c = torch.as_tensor(payload["coordinate_state"])
    cr = torch.as_tensor(payload["coordinate_state_raw"])
    q = torch.as_tensor(payload["task_quality"])
    tv = torch.as_tensor(payload["task_valid"])
    y = torch.as_tensor(payload["labels"])
    yv = torch.as_tensor(payload["label_valid"])
    g = torch.as_tensor(payload["global_source"])
    if s.ndim != 3 or tuple(s.shape[:2]) != (n, 5):
        raise ValueError(f"segment_base must be [N,5,D], got {tuple(s.shape)}")
    if x.ndim != 4 or tuple(x.shape[:3]) != (n, 5, 3):
        raise ValueError(f"task_features must be [N,5,3,D], got {tuple(x.shape)}")
    if c.ndim != 3 or tuple(c.shape[:2]) != (n, 5) or cr.shape != c.shape:
        raise ValueError("coordinate_state/raw must be [N,5,C]")
    if tuple(q.shape) != (n,5,3) or tv.shape != q.shape:
        raise ValueError("task_quality/task_valid must be [N,5,3]")
    if tuple(y.shape) != (n,5,3) or yv.shape != y.shape:
        raise ValueError("labels/label_valid must be [N,5,3]")
    if g.ndim != 2 or g.shape[0] != n:
        raise ValueError("global_source must be [N,D]")
    return {
        "cases": n,
        "segment_dim": int(s.shape[-1]),
        "task_dim": int(x.shape[-1]),
        "coordinate_dim": int(c.shape[-1]),
        "global_dim": int(g.shape[-1]),
        "serial_min": int(serials.min()),
        "serial_max": int(serials.max()),
    }

