#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared single-channel ContextCrop ResNet18-GN plus three-slice attention for the stenosis evidence expert."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


def _gn(channels: int) -> nn.GroupNorm:
    groups = min(16, int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


class AxialContextResNet18GN(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights=None, norm_layer=_gn)
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.backbone(x.float())))


class AxialStenosisEncoder(nn.Module):
    """Encode center-1/center/center+1 independently, then aggregate by attention."""

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        crop_encoder_batch_size: int = 24,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.crop_encoder_batch_size = int(crop_encoder_batch_size)
        self.encoder = AxialContextResNet18GN(1, hidden_dim, dropout)
        self.position_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.slice_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_chunks(self, flat: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, len(flat), self.crop_encoder_batch_size):
            outputs.append(self.encoder(flat[start:start + self.crop_encoder_batch_size]))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        axial_crops: torch.Tensor,
        axial_slice_valid: torch.Tensor,
        slice_permutation: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        # axial_crops: [B,5,3,1,H,W]
        if axial_crops.ndim != 6 or axial_crops.shape[2] != 3 or axial_crops.shape[3] != 1:
            raise ValueError(f"Expected [B,5,3,1,H,W], got {tuple(axial_crops.shape)}")
        if slice_permutation is not None:
            permutation = torch.as_tensor(slice_permutation, device=axial_crops.device, dtype=torch.long)
            if tuple(permutation.shape) != (3,) or sorted(permutation.tolist()) != [0, 1, 2]:
                raise ValueError(f"Invalid slice permutation: {slice_permutation}")
            axial_crops = axial_crops.index_select(2, permutation)
            axial_slice_valid = axial_slice_valid.index_select(2, permutation)

        batch, levels, slices, channels, height, width = axial_crops.shape
        flat = axial_crops.reshape(batch * levels * slices, channels, height, width)
        features = self._encode_chunks(flat).reshape(batch, levels, slices, self.hidden_dim)
        features_with_position = features + self.position_embedding.view(1, 1, 3, self.hidden_dim)
        scores = self.slice_score(features_with_position).squeeze(-1)
        scores = scores.masked_fill(axial_slice_valid < 0.5, -1e4)
        attention = torch.softmax(scores, dim=2) * axial_slice_valid
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1e-6)
        aggregated = (attention[..., None] * features).sum(dim=2)
        axial_valid = (axial_slice_valid.sum(dim=2) > 0.5).float()
        aggregated = aggregated * axial_valid[..., None]
        logits = self.head(aggregated).squeeze(-1) * axial_valid
        return {
            "logits": logits,
            "features": aggregated,
            "slice_features": features,
            "slice_attention": attention,
            "axial_valid": axial_valid,
        }

