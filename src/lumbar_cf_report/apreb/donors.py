#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from .coordinates import coordinate_distance


@dataclass(frozen=True)
class DonorCandidate:
    index: int
    coordinate_distance: float
    quality_distance: float
    score: float


def build_coordinate_candidate_cache(dataset, topk: int = 8, quality_weight: float = 0.25):
    """Precompute train-only opposite-label donor candidates.

    Key: (recipient_index, level_index, task_index)
    Value: list[DonorCandidate], same level, valid opposite target label, different patient.
    Only pre-intervention coordinates and task-quality enter the matching score.
    """
    cache: Dict[Tuple[int, int, int], List[DonorCandidate]] = {}
    n = len(dataset)
    for ri in range(n):
        for li in range(5):
            for ti in range(3):
                if float(dataset.label_valid[ri, li, ti]) <= 0.5:
                    continue
                target = float(dataset.labels[ri, li, ti])
                mask = (dataset.label_valid[:, li, ti] > 0.5) & (dataset.labels[:, li, ti] != target)
                mask[ri] = False
                idx = torch.where(mask)[0]
                if idx.numel() == 0:
                    continue
                cd = coordinate_distance(dataset.coordinate_state[ri, li], dataset.coordinate_state[idx, li])
                qd = (dataset.task_quality[idx, li, ti] - dataset.task_quality[ri, li, ti]).abs()
                score = cd + float(quality_weight) * qd
                order = torch.argsort(score)[: min(int(topk), int(idx.numel()))]
                rows = []
                for oi in order.tolist():
                    di = int(idx[oi])
                    rows.append(DonorCandidate(di, float(cd[oi]), float(qd[oi]), float(score[oi])))
                cache[(ri, li, ti)] = rows
    return cache


def sample_counterfactual_targets(batch_indices, dataset, cache, rng: random.Random, slots_per_case: int = 1):
    """Return (batch_position, recipient_index, level, task, donor_index, donor_label)."""
    pairs = []
    for bi, ri_raw in enumerate(batch_indices):
        ri = int(ri_raw)
        valid_keys = []
        for li in range(5):
            for ti in range(3):
                key = (ri, li, ti)
                if key in cache and cache[key]:
                    valid_keys.append(key)
        if not valid_keys:
            continue
        rng.shuffle(valid_keys)
        for key in valid_keys[: max(1, int(slots_per_case))]:
            _, li, ti = key
            donor = rng.choice(cache[key])
            pairs.append((bi, ri, li, ti, donor.index, float(dataset.labels[donor.index, li, ti])))
    return pairs


def build_external_coordinate_candidate_cache(recipient_dataset, donor_dataset, topk: int = 8, quality_weight: float = 0.25):
    """Counterfactual donor cache for validation recipients using train-only donors."""
    cache: Dict[Tuple[int, int, int], List[DonorCandidate]] = {}
    for ri in range(len(recipient_dataset)):
        for li in range(5):
            for ti in range(3):
                if float(recipient_dataset.label_valid[ri, li, ti]) <= 0.5:
                    continue
                target = float(recipient_dataset.labels[ri, li, ti])
                mask = (donor_dataset.label_valid[:, li, ti] > 0.5) & (donor_dataset.labels[:, li, ti] != target)
                idx = torch.where(mask)[0]
                if idx.numel() == 0:
                    continue
                cd = coordinate_distance(recipient_dataset.coordinate_state[ri, li], donor_dataset.coordinate_state[idx, li])
                qd = (donor_dataset.task_quality[idx, li, ti] - recipient_dataset.task_quality[ri, li, ti]).abs()
                score = cd + float(quality_weight) * qd
                order = torch.argsort(score)[: min(int(topk), int(idx.numel()))]
                rows = []
                for oi in order.tolist():
                    di = int(idx[oi])
                    rows.append(DonorCandidate(di, float(cd[oi]), float(qd[oi]), float(score[oi])))
                cache[(ri, li, ti)] = rows
    return cache

