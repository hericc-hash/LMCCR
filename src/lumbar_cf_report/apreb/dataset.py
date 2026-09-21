#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .schema import validate_dataset


def safe_torch_load(path: str):
    try: return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(path,map_location="cpu")


class APREBDataset(Dataset):
    def __init__(self, path: str):
        p=Path(path)
        if not p.is_file(): raise FileNotFoundError(p)
        self.path=p; self.payload=safe_torch_load(str(p)); self.report=validate_dataset(self.payload)
        for k in ["serials","segment_base","task_features","coordinate_state","coordinate_state_raw",
                  "task_quality","task_valid","labels","label_valid","global_source"]:
            setattr(self,k,torch.as_tensor(self.payload[k]).float() if k!="serials" else torch.as_tensor(self.payload[k]).long())
        self.node_coords_raw=torch.as_tensor(self.payload.get("node_coords_raw",torch.zeros(len(self.serials),5,2))).float()
        self.node_coords_canonical=torch.as_tensor(self.payload.get("node_coords_canonical",torch.zeros(len(self.serials),5,2))).float()
        self.geometry_features=torch.as_tensor(self.payload.get("geometry_features",torch.zeros(len(self.serials),0))).float()
    def __len__(self): return int(self.serials.numel())
    def __getitem__(self,i):
        i=int(i)
        return {k:getattr(self,k)[i] for k in ["serials","segment_base","task_features","coordinate_state","coordinate_state_raw","task_quality","task_valid","labels","label_valid","global_source"]}
    @property
    def segment_dim(self): return int(self.segment_base.shape[-1])
    @property
    def task_dim(self): return int(self.task_features.shape[-1])
    @property
    def coordinate_dim(self): return int(self.coordinate_state.shape[-1])
    @property
    def global_dim(self): return int(self.global_source.shape[-1])

