#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Mapping, Tuple
import torch

LOCAL_COORDINATE_FEATURE_NAMES: List[str] = [
    "canonical_x", "canonical_y", "raw_rel_x", "raw_rel_y",
    "tangent_x", "tangent_y", "normal_x", "normal_y",
    "local_spacing", "curvature", "level_index_norm", "node_finite",
]


def _safe_normalize(v: torch.Tensor, eps: float = 1e-6):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _case_minmax(points: torch.Tensor):
    lo = points.amin(dim=1, keepdim=True)
    hi = points.amax(dim=1, keepdim=True)
    return (points - lo) / (hi - lo).clamp_min(1e-6)


def derive_local_coordinate_state(node_coords_raw, node_coords_canonical):
    raw = torch.as_tensor(node_coords_raw).float()
    canonical = torch.as_tensor(node_coords_canonical).float()
    if raw.ndim != 3 or canonical.shape != raw.shape or tuple(raw.shape[1:]) != (5,2):
        raise ValueError(f"node coordinates must be [N,5,2], got {tuple(raw.shape)}")
    finite = torch.isfinite(raw).all(-1) & torch.isfinite(canonical).all(-1)
    raw = torch.nan_to_num(raw); canonical = torch.nan_to_num(canonical)
    raw_rel = _case_minmax(raw)
    tangent = torch.zeros_like(canonical)
    tangent[:,0] = canonical[:,1]-canonical[:,0]
    tangent[:,-1] = canonical[:,-1]-canonical[:,-2]
    tangent[:,1:-1] = canonical[:,2:]-canonical[:,:-2]
    tangent = _safe_normalize(tangent)
    normal = torch.stack([-tangent[...,1], tangent[...,0]], dim=-1)
    edges = (canonical[:,1:]-canonical[:,:-1]).norm(dim=-1)
    spacing = torch.zeros(raw.shape[0],5,dtype=raw.dtype,device=raw.device)
    spacing[:,0]=edges[:,0]; spacing[:,-1]=edges[:,-1]
    spacing[:,1:-1]=0.5*(edges[:,:-1]+edges[:,1:])
    spacing = spacing / spacing.mean(1,keepdim=True).clamp_min(1e-6)
    curvature = torch.zeros_like(spacing)
    ce = (tangent[:,1:]-tangent[:,:-1]).norm(dim=-1)
    curvature[:,0]=ce[:,0]; curvature[:,-1]=ce[:,-1]
    curvature[:,1:-1]=0.5*(ce[:,:-1]+ce[:,1:])
    level = torch.linspace(-1,1,5,device=raw.device,dtype=raw.dtype)[None,:,None].expand(raw.shape[0],-1,-1)
    return torch.cat([
        canonical, raw_rel, tangent, normal, spacing[...,None], curvature[...,None],
        level, finite.float()[...,None]
    ], dim=-1)


def build_coordinate_state_from_bank(bank: Mapping[str, object]) -> Tuple[torch.Tensor,List[str]]:
    local = derive_local_coordinate_state(bank["node_coords_raw"], bank["node_coords_canonical"])
    names = list(LOCAL_COORDINATE_FEATURE_NAMES)
    geometry = torch.as_tensor(bank.get("geometry_features", torch.zeros(local.shape[0],0))).float()
    gnames = [str(x) for x in bank.get("geometry_feature_names", [])]
    if geometry.ndim != 2 or geometry.shape[0] != local.shape[0]:
        raise ValueError("geometry_features must be [N,G]")
    if len(gnames) != geometry.shape[1]:
        gnames = [f"geometry_{i}" for i in range(int(geometry.shape[1]))]
    if geometry.shape[1] > 0:
        local = torch.cat([local, geometry[:,None,:].expand(-1,5,-1)], -1)
        names.extend([f"global_geometry::{n}" for n in gnames])
    return local, names


def fit_coordinate_normalizer(state):
    x = torch.as_tensor(state).float().reshape(-1, torch.as_tensor(state).shape[-1])
    mean=x.mean(0); std=x.std(0,unbiased=False).clamp_min(1e-5)
    return mean,std


def apply_coordinate_normalizer(state, mean, std):
    return (torch.as_tensor(state).float()-mean[None,None])/std[None,None]


def coordinate_distance(query, candidates, feature_weights=None):
    q=torch.as_tensor(query).float().view(1,-1); c=torch.as_tensor(candidates).float()
    d=c-q
    if feature_weights is not None:
        d=d*torch.as_tensor(feature_weights).float().view(1,-1)
    return d.square().mean(-1).sqrt()

