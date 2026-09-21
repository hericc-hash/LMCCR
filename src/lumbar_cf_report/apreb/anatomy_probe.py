#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn


class IndependentAnatomyProbe(nn.Module):
    """Independent factual probe used only for intervention audit.

    It predicts the task-agnostic segment-base token and the patient-specific coordinate
    state from a task-projected feature. It never receives coordinates as input, so a
    small counterfactual drift cannot be obtained trivially by copying C.
    """
    def __init__(self,task_dim:int,segment_dim:int,coordinate_dim:int,task_embedding_dim:int=24,hidden_dim:int=256,dropout:float=0.1):
        super().__init__();self.task_dim=int(task_dim);self.segment_dim=int(segment_dim);self.coordinate_dim=int(coordinate_dim)
        self.task_embedding=nn.Embedding(3,task_embedding_dim)
        self.trunk=nn.Sequential(nn.LayerNorm(task_dim+task_embedding_dim),nn.Linear(task_dim+task_embedding_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,hidden_dim),nn.GELU())
        self.segment_head=nn.Linear(hidden_dim,segment_dim);self.coordinate_head=nn.Linear(hidden_dim,coordinate_dim)
    def forward(self,x,task_index):
        t=torch.as_tensor(task_index,device=x.device).long()
        te=self.task_embedding(t)
        h=self.trunk(torch.cat([x.float(),te],-1))
        return {"segment_prediction":self.segment_head(h),"coordinate_prediction":self.coordinate_head(h),"embedding":h}

