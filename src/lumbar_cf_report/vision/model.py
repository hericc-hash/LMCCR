#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified evidence-level visual branch for lumbar MRI.

Frozen single-task experts are not re-trained here. Their logits and features are
cached once, then this module learns a low-capacity evidence representation.
Deployment logits are residual corrections around immutable expert logits and
can fall back task-by-task to the original expert outputs.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from .schema import EvidenceDimensions, LEVELS, TASKS

VERSION = "Lumbar-Visual-Evidence-Branch-v1"


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-8), 1.0 - 1e-8)
    return math.log(value / (1.0 - value))


class TaskProjector(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


class LumbarVisualEvidenceEncoder(nn.Module):
    """Five segment tokens + one global geometry token + one patient token."""

    def __init__(
        self,
        geometry_feature_dim: int,
        expert_feature_dim: int = 768,
        task_projected_dim: int = 128,
        token_dim: int = 256,
        level_embedding_dim: int = 16,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        dropout: float = 0.10,
        max_context_residual: float = 0.20,
        max_task_logit_residual: float = 0.15,
        max_lordosis_logit_residual: float = 0.10,
        residual_initial_fraction: float = 1e-4,
    ):
        super().__init__()
        if token_dim % transformer_heads != 0:
            raise ValueError("token_dim must be divisible by transformer_heads")
        self.geometry_feature_dim = int(geometry_feature_dim)
        self.dimensions = EvidenceDimensions(
            expert_feature_dim=int(expert_feature_dim),
            task_projected_dim=int(task_projected_dim),
            token_dim=int(token_dim),
            level_embedding_dim=int(level_embedding_dim),
        )
        self.max_context_residual = float(max_context_residual)
        self.max_task_logit_residual = float(max_task_logit_residual)
        self.max_lordosis_logit_residual = float(max_lordosis_logit_residual)

        self.task_projectors = nn.ModuleDict({
            task: TaskProjector(expert_feature_dim, task_projected_dim, dropout)
            for task in TASKS
        })
        self.level_embedding = nn.Embedding(len(LEVELS), level_embedding_dim)
        scalar_dim = len(TASKS) * 4  # logit, probability, quality, validity
        segment_input_dim = task_projected_dim * len(TASKS) + scalar_dim + 2 + level_embedding_dim
        self.segment_harmonizer = nn.Sequential(
            nn.Linear(segment_input_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

        global_input_dim = geometry_feature_dim + len(LEVELS) * 2 + 3
        self.global_projector = nn.Sequential(
            nn.Linear(global_input_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=token_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        self.sequence_position = nn.Parameter(torch.zeros(1, 1 + len(LEVELS), token_dim))
        nn.init.normal_(self.sequence_position, std=0.02)

        initial_raw = _inverse_sigmoid(residual_initial_fraction)
        self.context_residual_raw = nn.Parameter(torch.tensor(initial_raw, dtype=torch.float32))
        self.task_residual_raw = nn.Parameter(
            torch.full((len(TASKS),), initial_raw, dtype=torch.float32)
        )
        self.lordosis_residual_raw = nn.Parameter(torch.tensor(initial_raw, dtype=torch.float32))

        self.task_aux_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, token_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(token_dim // 2, 1),
            )
            for task in TASKS
        })
        self.lordosis_aux_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, 1),
        )
        self.patient_projector = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

        # Set after validation model selection. False means deploy original expert logits.
        self.register_buffer(
            "deploy_unified_task_mask",
            torch.zeros(len(TASKS), dtype=torch.bool),
        )
        self.register_buffer(
            "deploy_unified_lordosis",
            torch.tensor(False, dtype=torch.bool),
        )

    def residual_scales(self) -> Dict[str, torch.Tensor]:
        return {
            "context": self.max_context_residual * torch.sigmoid(self.context_residual_raw),
            "tasks": self.max_task_logit_residual * torch.sigmoid(self.task_residual_raw),
            "lordosis": self.max_lordosis_logit_residual * torch.sigmoid(self.lordosis_residual_raw),
        }

    def set_deployment_policy(
        self,
        task_mask: torch.Tensor | list[bool],
        use_unified_lordosis: bool,
    ) -> None:
        mask = torch.as_tensor(task_mask, dtype=torch.bool, device=self.deploy_unified_task_mask.device)
        if tuple(mask.shape) != (len(TASKS),):
            raise ValueError(f"task_mask must have shape {(len(TASKS),)}, got {tuple(mask.shape)}")
        self.deploy_unified_task_mask.copy_(mask)
        self.deploy_unified_lordosis.copy_(
            torch.tensor(bool(use_unified_lordosis), device=self.deploy_unified_lordosis.device)
        )

    def _task_features(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "disc": self.task_projectors["disc"](batch["disc_features"]),
            "stenosis": self.task_projectors["stenosis"](batch["stenosis_features"]),
            "nerve": self.task_projectors["nerve"](batch["nerve_features"]),
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        slot_keep: Optional[torch.Tensor] = None,
        geometry_keep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        anchor_logits = batch["expert_logits"].float()
        quality = batch["task_quality"].float()
        validity = batch["task_valid"].float()
        if slot_keep is not None:
            keep = slot_keep.float()
            if keep.shape != validity.shape:
                raise ValueError(f"slot_keep shape {tuple(keep.shape)} != {tuple(validity.shape)}")
            validity = validity * keep
        probabilities = torch.sigmoid(anchor_logits)
        projected = self._task_features(batch)
        feature_stack = torch.cat([projected[task] for task in TASKS], dim=-1)
        scalar_stack = torch.cat(
            [anchor_logits, probabilities, quality, validity],
            dim=-1,
        )
        node_coords = batch["node_coords_canonical"].float()
        batch_size = anchor_logits.shape[0]
        level_ids = torch.arange(len(LEVELS), device=anchor_logits.device)
        level_embedding = self.level_embedding(level_ids)[None].expand(batch_size, -1, -1)
        segment_input = torch.cat(
            [feature_stack, scalar_stack, node_coords, level_embedding],
            dim=-1,
        )
        segment_base = self.segment_harmonizer(segment_input)

        lordosis_anchor = batch["lordosis_anchor_logit"].float().view(batch_size, 1)
        lordosis_pred = batch["lordosis_anchor_prediction"].float().view(batch_size, 1)
        lordosis_valid = batch["lordosis_label_valid"].float().view(batch_size, 1)
        geometry = batch["geometry_features"].float()
        if geometry_keep is not None:
            keep = geometry_keep.float().view(batch_size, 1)
            geometry = geometry * keep
            lordosis_valid = lordosis_valid * keep
        global_input = torch.cat(
            [geometry, node_coords.reshape(batch_size, -1), lordosis_anchor, lordosis_pred, lordosis_valid],
            dim=-1,
        )
        global_base = self.global_projector(global_input)
        sequence_base = torch.cat([global_base[:, None], segment_base], dim=1)
        context = self.context_encoder(sequence_base + self.sequence_position)
        scales = self.residual_scales()
        sequence = sequence_base + scales["context"] * (context - sequence_base)
        global_token = sequence[:, 0]
        segment_tokens = sequence[:, 1:]

        aux_logits = torch.cat(
            [self.task_aux_heads[task](segment_tokens) for task in TASKS],
            dim=-1,
        )
        bounded_task_residual = 2.0 * torch.tanh(aux_logits / 2.0)
        unified_logits = anchor_logits + bounded_task_residual * scales["tasks"].view(1, 1, -1)
        deploy_mask = self.deploy_unified_task_mask.view(1, 1, -1)
        deployed_logits = torch.where(deploy_mask, unified_logits, anchor_logits)

        lordosis_aux_logit = self.lordosis_aux_head(global_token).squeeze(-1)
        bounded_lordosis_residual = 2.0 * torch.tanh(lordosis_aux_logit / 2.0)
        lordosis_unified_logit = (
            lordosis_anchor.squeeze(-1) + bounded_lordosis_residual * scales["lordosis"]
        )
        lordosis_deployed_logit = torch.where(
            self.deploy_unified_lordosis,
            lordosis_unified_logit,
            lordosis_anchor.squeeze(-1),
        )
        patient_token = self.patient_projector(sequence.mean(dim=1))
        llm_tokens = torch.cat(
            [global_token[:, None], segment_tokens, patient_token[:, None]],
            dim=1,
        )
        return {
            "expert_logits": anchor_logits,
            "aux_logits": aux_logits,
            "task_residual_logits": bounded_task_residual,
            "unified_logits": unified_logits,
            "deployed_logits": deployed_logits,
            "lordosis_anchor_logit": lordosis_anchor.squeeze(-1),
            "lordosis_aux_logit": lordosis_aux_logit,
            "lordosis_residual_logit": bounded_lordosis_residual,
            "lordosis_unified_logit": lordosis_unified_logit,
            "lordosis_deployed_logit": lordosis_deployed_logit,
            "task_projected_features": torch.stack([projected[t] for t in TASKS], dim=2),
            "segment_tokens": segment_tokens,
            "global_geometry_token": global_token,
            "patient_visual_token": patient_token,
            "llm_visual_tokens": llm_tokens,
            "context_residual_scale": scales["context"],
            "task_residual_scales": scales["tasks"],
            "lordosis_residual_scale": scales["lordosis"],
            "deploy_unified_task_mask": self.deploy_unified_task_mask,
            "deploy_unified_lordosis": self.deploy_unified_lordosis,
        }

