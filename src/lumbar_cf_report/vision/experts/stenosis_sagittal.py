#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the stenosis evidence expert manual-center sagittal branch, parameter-key compatible with the pretrained sagittal expert.

The state_dict of :class:`SagittalStenosisEncoder` intentionally
contains exactly the the pretrained sagittal expert prefixes ``encoder.*`` and ``head.*`` so that
``best_common.pt['model']`` can be loaded with strict=True.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    groups = min(16, int(ch))
    while groups > 1 and ch % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(ch))


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(torch.sin(angle), torch.cos(angle))
    angle = torch.where(angle > 0.5 * torch.pi, angle - torch.pi, angle)
    return torch.where(angle < -0.5 * torch.pi, angle + torch.pi, angle)


class StenosisEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        from torchvision.models import resnet18

        base = resnet18(weights=None, norm_layer=_gn)
        old = base.conv1
        base.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(base.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
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
        return self.proj(self.pool(self.backbone(x)))


class SagittalStenosisEncoder(nn.Module):
    """Exact the pretrained sagittal expert Global/Core shared-encoder architecture."""

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        roi_crop_size: int = 192,
        crop_encoder_batch_size: int = 12,
        use_roi_mask_channel: bool = True,
        roi_pad_value: float = 0.0,
        roi_min_half_size: float = 0.012,
        sag_canal_global_posterior_shift: float = 0.56,
        sag_canal_global_half_w: float = 0.40,
        sag_canal_global_half_h: float = 0.50,
        sag_canal_core_width_frac: float = 0.60,
        sag_canal_core_height_frac: float = 0.70,
        sag_canal_core_anterior_shift: float = 0.0,
        disc_angle_clip_deg: float = 35.0,
        l5s1_angle_gain: float = 1.70,
        l5s1_angle_offset_deg: float = -2.0,
        l5s1_angle_clip_deg: float = 70.0,
        disc_angle_smooth: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.roi_crop_size = int(roi_crop_size)
        self.crop_encoder_batch_size = int(crop_encoder_batch_size)
        self.use_roi_mask_channel = bool(use_roi_mask_channel)
        self.roi_pad_value = float(roi_pad_value)
        self.roi_min_half_size = float(roi_min_half_size)
        self.sag_canal_global_posterior_shift = float(sag_canal_global_posterior_shift)
        self.sag_canal_global_half_w = float(sag_canal_global_half_w)
        self.sag_canal_global_half_h = float(sag_canal_global_half_h)
        self.sag_canal_core_width_frac = float(sag_canal_core_width_frac)
        self.sag_canal_core_height_frac = float(sag_canal_core_height_frac)
        self.sag_canal_core_anterior_shift = float(sag_canal_core_anterior_shift)
        self.disc_angle_clip_deg = float(disc_angle_clip_deg)
        self.l5s1_angle_gain = float(l5s1_angle_gain)
        self.l5s1_angle_offset_deg = float(l5s1_angle_offset_deg)
        self.l5s1_angle_clip_deg = float(l5s1_angle_clip_deg)
        self.disc_angle_smooth = bool(disc_angle_smooth)

        self.encoder = StenosisEncoder(
            4 if self.use_roi_mask_channel else 3,
            self.hidden_dim,
            dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def _neighbors(self, coords: torch.Tensor):
        first = coords[:, 1:2] - coords[:, 0:1]
        previous = torch.cat([coords[:, 0:1] - first, coords[:, :-1]], dim=1)
        last = coords[:, -1:] - coords[:, -2:-1]
        following = torch.cat([coords[:, 1:], coords[:, -1:] + last], dim=1)
        gap = 0.5 * (
            (coords[..., 1] - previous[..., 1]).abs()
            + (following[..., 1] - coords[..., 1]).abs()
        )
        return gap.clamp_min(1.0 / 384.0), following - previous

    def _angles(self, tangent: torch.Tensor) -> torch.Tensor:
        angle = _wrap(torch.atan2(tangent[..., 1], tangent[..., 0]) - 0.5 * torch.pi)
        if self.disc_angle_smooth:
            smooth = angle.clone()
            smooth[:, 0] = 0.75 * angle[:, 0] + 0.25 * angle[:, 1]
            smooth[:, 1:-1] = 0.25 * angle[:, :-2] + 0.5 * angle[:, 1:-1] + 0.25 * angle[:, 2:]
            smooth[:, -1] = 0.25 * angle[:, -2] + 0.75 * angle[:, -1]
            angle = _wrap(smooth)
        clip = self.disc_angle_clip_deg * torch.pi / 180.0
        angle = angle.clamp(-clip, clip).clone()
        l5_clip = self.l5s1_angle_clip_deg * torch.pi / 180.0
        offset = self.l5s1_angle_offset_deg * torch.pi / 180.0
        angle[:, 4] = (angle[:, 4] * self.l5s1_angle_gain + offset).clamp(-l5_clip, l5_clip)
        return angle

    def _clip(self, centers: torch.Tensor, half: torch.Tensor, angles: torch.Tensor):
        centers = centers.clamp(0.01, 0.99)
        edge = torch.minimum(centers - 0.002, 0.998 - centers).clamp_min(self.roi_min_half_size)
        return centers, torch.minimum(half, edge).clamp_min(self.roi_min_half_size), angles

    def compute_canal_geometry(self, coords: torch.Tensor):
        gap, tangent = self._neighbors(coords)
        angle = self._angles(tangent)
        axis = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        global_center = coords + self.sag_canal_global_posterior_shift * gap[..., None] * axis
        global_half = torch.stack(
            [self.sag_canal_global_half_w * gap, self.sag_canal_global_half_h * gap],
            dim=-1,
        ).clamp_min(self.roi_min_half_size)
        core_center = global_center - self.sag_canal_core_anterior_shift * gap[..., None] * axis
        fraction = torch.tensor(
            [self.sag_canal_core_width_frac, self.sag_canal_core_height_frac],
            device=coords.device,
            dtype=coords.dtype,
        ).view(1, 1, 2)
        core_half = (global_half * fraction).clamp_min(self.roi_min_half_size)
        global_center, global_half, _ = self._clip(global_center, global_half, angle)
        core_center, core_half, _ = self._clip(core_center, core_half, angle)
        return global_center, global_half, core_center, core_half, angle

    def extract_crops(
        self,
        images: torch.Tensor,
        centers: torch.Tensor,
        half: torch.Tensor,
        angles: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = images.shape
        levels = centers.shape[1]
        size = self.roi_crop_size
        dtype, device = images.dtype, images.device
        expanded = images[:, None].expand(-1, levels, -1, -1, -1)
        line = torch.linspace(-1, 1, size, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(line, line, indexing="ij")
        base = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, size, size, 2)
        half_x = half[..., 0:1].clamp_min(self.roi_min_half_size)
        half_y = half[..., 1:2].clamp_min(self.roi_min_half_size)
        maximum = torch.maximum(half_x, half_y)
        mask = (
            (base[..., 0].abs() <= (half_x / maximum)[..., None])
            & (base[..., 1].abs() <= (half_y / maximum)[..., None])
        ).to(dtype)
        local_x = base[..., 0] * maximum[..., None]
        local_y = base[..., 1] * maximum[..., None]
        cos_a = torch.cos(angles)[..., None, None]
        sin_a = torch.sin(angles)[..., None, None]
        delta_x = local_x * cos_a - local_y * sin_a
        delta_y = local_x * sin_a + local_y * cos_a
        grid = (centers[..., None, None, :] + torch.stack([delta_x, delta_y], dim=-1)) * 2.0 - 1.0
        sampled = F.grid_sample(
            expanded.reshape(batch * levels, channels, height, width),
            grid.reshape(batch * levels, size, size, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, levels, channels, size, size)
        mask = mask[:, :, None]
        sampled = sampled * mask
        gate = valid.to(dtype)[:, :, None, None, None]
        sampled = sampled * gate
        mask = mask * gate
        return torch.cat([sampled, mask], dim=2) if self.use_roi_mask_channel else sampled

    def _encode(self, crops: torch.Tensor) -> torch.Tensor:
        batch, levels, kinds, channels, height, width = crops.shape
        flat = crops.reshape(batch * levels * kinds, channels, height, width)
        outputs = []
        for start in range(0, len(flat), self.crop_encoder_batch_size):
            outputs.append(self.encoder(flat[start:start + self.crop_encoder_batch_size]))
        return torch.cat(outputs, dim=0).reshape(batch, levels, kinds, self.hidden_dim)

    def forward(
        self,
        sagittal_images: torch.Tensor,
        sagittal_coords: torch.Tensor,
        sagittal_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        global_center, global_half, core_center, core_half, angle = self.compute_canal_geometry(sagittal_coords)
        global_crop = self.extract_crops(sagittal_images, global_center, global_half, angle, sagittal_valid)
        core_crop = self.extract_crops(sagittal_images, core_center, core_half, angle, sagittal_valid)
        tokens = self._encode(torch.stack([global_crop, core_crop], dim=2))
        global_feature = tokens[:, :, 0]
        core_feature = tokens[:, :, 1]
        raw_feature = torch.cat([global_feature, core_feature, global_feature - core_feature], dim=-1)
        batch, levels = raw_feature.shape[:2]
        hidden_feature = self.head[:-1](raw_feature.reshape(batch * levels, -1)).reshape(batch, levels, self.hidden_dim)
        logits = self.head[-1](hidden_feature).reshape(batch, levels)
        return {
            "logits": logits,
            "hidden_features": hidden_feature,
            "raw_features": raw_feature,
            "tokens": tokens,
            "global_crops_preview": global_crop.detach(),
            "core_crops_preview": core_crop.detach(),
            "global_centers": global_center,
            "global_half_sizes": global_half,
            "core_centers": core_center,
            "core_half_sizes": core_half,
            "angles": angle,
        }

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def unfreeze_head(self) -> None:
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)

    def unfreeze_last_stage(self) -> None:
        for parameter in self.encoder.backbone[7].parameters():
            parameter.requires_grad_(True)
        for parameter in self.encoder.proj.parameters():
            parameter.requires_grad_(True)
        self.unfreeze_head()



