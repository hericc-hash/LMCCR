#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Axial-biased bounded-gate dual-view nerve evidence model.

Changes relative to the preceding nerve baseline:
- the trainable axial fusion gate is bounded to [gate_min, gate_max];
- the gate is initialized at gate_initial with a zero-initialized final weight;
- training can temporarily override the gate with a fixed axial weight;
- strict sagittal-only / axial-only fallback is preserved;
- ROI geometry, encoders, side attention, noisy-OR and residual fusion are unchanged.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

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


class GNResNet18Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        try:
            from torchvision.models import resnet18
        except Exception as exc:  # pragma: no cover - environment guard
            raise ImportError("torchvision is required") from exc

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
        self.body = nn.Sequential(
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
        return self.proj(self.pool(self.body(x.float())))


class NerveEvidenceExpert(nn.Module):
    VERSION = "AxialBiased-BoundedGate-Nerve-Evidence-v1"

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        roi_crop_size: int = 192,
        crop_encoder_batch_size: int = 20,
        roi_min_half_size: float = 0.012,
        stenosis_posterior_shift: float = 0.56,
        stenosis_half_w: float = 0.40,
        stenosis_half_h: float = 0.50,
        nerve_extra_posterior_shift: float = 0.07,
        nerve_extra_caudal_shift: float = 0.06,
        nerve_scale_w: float = 1.10,
        nerve_scale_h: float = 1.12,
        disc_angle_clip_deg: float = 35.0,
        l5s1_angle_gain: float = 1.70,
        l5s1_angle_offset_deg: float = -2.0,
        l5s1_angle_clip_deg: float = 70.0,
        disc_angle_smooth: bool = True,
        residual_scale: float = 0.35,
        gate_min: float = 0.45,
        gate_max: float = 0.85,
        gate_initial: float = 0.60,
    ):
        super().__init__()
        if not (0.0 <= gate_min < gate_max <= 1.0):
            raise ValueError(f"Invalid gate bounds: [{gate_min}, {gate_max}]")
        if not (gate_min <= gate_initial <= gate_max):
            raise ValueError(
                f"gate_initial={gate_initial} must lie inside [{gate_min}, {gate_max}]"
            )

        self.hidden_dim = int(hidden_dim)
        self.roi_crop_size = int(roi_crop_size)
        self.crop_encoder_batch_size = int(crop_encoder_batch_size)
        self.roi_min_half_size = float(roi_min_half_size)
        self.stenosis_posterior_shift = float(stenosis_posterior_shift)
        self.stenosis_half_w = float(stenosis_half_w)
        self.stenosis_half_h = float(stenosis_half_h)
        self.nerve_extra_posterior_shift = float(nerve_extra_posterior_shift)
        self.nerve_extra_caudal_shift = float(nerve_extra_caudal_shift)
        self.nerve_scale_w = float(nerve_scale_w)
        self.nerve_scale_h = float(nerve_scale_h)
        self.disc_angle_clip_deg = float(disc_angle_clip_deg)
        self.l5s1_angle_gain = float(l5s1_angle_gain)
        self.l5s1_angle_offset_deg = float(l5s1_angle_offset_deg)
        self.l5s1_angle_clip_deg = float(l5s1_angle_clip_deg)
        self.disc_angle_smooth = bool(disc_angle_smooth)
        self.residual_scale = float(residual_scale)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.gate_initial = float(gate_initial)

        self.sag_encoder = GNResNet18Encoder(4, hidden_dim, dropout)
        self.ax_encoder = GNResNet18Encoder(4, hidden_dim, dropout)
        self.sag_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.side_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.side_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim + 4),
            nn.Linear(4 * hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_gate()

    def _initialize_gate(self) -> None:
        """Start every both-view sample at exactly gate_initial."""
        final = self.gate[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        relative = (self.gate_initial - self.gate_min) / (self.gate_max - self.gate_min)
        relative = min(max(relative, 1e-5), 1.0 - 1e-5)
        bias = math.log(relative / (1.0 - relative))
        nn.init.constant_(final.bias, bias)

    def set_gate_trainable(self, trainable: bool) -> None:
        for parameter in self.gate.parameters():
            parameter.requires_grad_(bool(trainable))

    def _local(self, coords: torch.Tensor):
        first = coords[:, 1:2] - coords[:, 0:1]
        previous = torch.cat([coords[:, 0:1] - first, coords[:, :-1]], dim=1)
        last = coords[:, -1:] - coords[:, -2:-1]
        following = torch.cat([coords[:, 1:], coords[:, -1:] + last], dim=1)
        gap = 0.5 * (
            (coords[..., 1] - previous[..., 1]).abs()
            + (following[..., 1] - coords[..., 1]).abs()
        ).clamp_min(1.0 / 384.0)
        tangent = following - previous
        angle = _wrap(torch.atan2(tangent[..., 1], tangent[..., 0]) - 0.5 * torch.pi)
        if self.disc_angle_smooth:
            smooth = angle.clone()
            smooth[:, 0] = 0.75 * angle[:, 0] + 0.25 * angle[:, 1]
            smooth[:, 1:-1] = (
                0.25 * angle[:, :-2] + 0.50 * angle[:, 1:-1] + 0.25 * angle[:, 2:]
            )
            smooth[:, -1] = 0.25 * angle[:, -2] + 0.75 * angle[:, -1]
            angle = _wrap(smooth)
        clip = self.disc_angle_clip_deg * torch.pi / 180.0
        angle = angle.clamp(-clip, clip).clone()
        l5_clip = self.l5s1_angle_clip_deg * torch.pi / 180.0
        offset = self.l5s1_angle_offset_deg * torch.pi / 180.0
        angle[:, 4] = (angle[:, 4] * self.l5s1_angle_gain + offset).clamp(-l5_clip, l5_clip)
        posterior = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        caudal = torch.stack([-torch.sin(angle), torch.cos(angle)], dim=-1)
        return gap, angle, posterior, caudal

    def compute_sagittal_nerve_geometry(self, coords: torch.Tensor):
        gap, angle, posterior, caudal = self._local(coords)
        stenosis_center = coords + self.stenosis_posterior_shift * gap[..., None] * posterior
        half = torch.stack(
            [self.stenosis_half_w * gap, self.stenosis_half_h * gap], dim=-1
        )
        center = (
            stenosis_center
            + self.nerve_extra_posterior_shift * gap[..., None] * posterior
            + self.nerve_extra_caudal_shift * gap[..., None] * caudal
        )
        scale = torch.tensor(
            [self.nerve_scale_w, self.nerve_scale_h],
            device=coords.device,
            dtype=coords.dtype,
        )
        half = half * scale
        center = center.clamp(0.01, 0.99)
        edge = torch.minimum(center - 0.002, 0.998 - center).clamp_min(
            self.roi_min_half_size
        )
        half = torch.minimum(half, edge).clamp_min(self.roi_min_half_size)
        return center, half, angle

    def extract_sagittal_crops(
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
        half_x, half_y = half[..., 0:1], half[..., 1:2]
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
        sampling_grid = (
            centers[..., None, None, :] + torch.stack([delta_x, delta_y], dim=-1)
        ) * 2.0 - 1.0
        sampled = F.grid_sample(
            expanded.reshape(batch * levels, channels, height, width),
            sampling_grid.reshape(batch * levels, size, size, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, levels, channels, size, size)
        validity = valid[:, :, None, None, None].to(dtype)
        sampled = sampled * mask[:, :, None] * validity
        mask = mask[:, :, None] * validity
        return torch.cat([sampled, mask], dim=2)

    def _encode_chunks(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, len(x), self.crop_encoder_batch_size):
            outputs.append(encoder(x[start : start + self.crop_encoder_batch_size]))
        return torch.cat(outputs, dim=0)

    def _bounded_gate(self, gate_input: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.gate(gate_input).squeeze(-1))
        return self.gate_min + (self.gate_max - self.gate_min) * raw

    def forward(
        self,
        sagittal_images: torch.Tensor,
        sagittal_coords: torch.Tensor,
        sagittal_valid: torch.Tensor,
        sagittal_quality: torch.Tensor,
        axial_crops: torch.Tensor,
        axial_side_valid: torch.Tensor,
        axial_quality: torch.Tensor,
        sagittal_keep: Optional[torch.Tensor] = None,
        axial_keep: Optional[torch.Tensor] = None,
        swap_left_right: bool = False,
        fixed_axial_gate: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, levels = sagittal_coords.shape[:2]
        center, half, angle = self.compute_sagittal_nerve_geometry(sagittal_coords)
        sagittal_crops = self.extract_sagittal_crops(
            sagittal_images, center, half, angle, sagittal_valid
        )
        sagittal_features = self._encode_chunks(
            self.sag_encoder,
            sagittal_crops.reshape(
                batch * levels, 4, self.roi_crop_size, self.roi_crop_size
            ),
        ).reshape(batch, levels, self.hidden_dim)
        sagittal_logits = self.sag_head(sagittal_features).squeeze(-1)

        axial = axial_crops
        side_valid = axial_side_valid
        if swap_left_right:
            axial = axial.flip(2)
            side_valid = side_valid.flip(2)
        side_features = self._encode_chunks(
            self.ax_encoder,
            axial.reshape(
                batch * levels * 2, 4, self.roi_crop_size, self.roi_crop_size
            ),
        ).reshape(batch, levels, 2, self.hidden_dim)
        side_logits = self.side_head(side_features).squeeze(-1)
        side_probabilities = torch.sigmoid(side_logits) * side_valid
        probability_none = torch.prod(1.0 - side_probabilities, dim=2)
        probability_any = (1.0 - probability_none).clamp(1e-5, 1.0 - 1e-5)
        axial_logits = torch.logit(probability_any)

        side_scores = self.side_score(side_features).squeeze(-1).masked_fill(
            side_valid < 0.5, -1e4
        )
        side_attention = torch.softmax(side_scores, dim=2) * side_valid
        side_attention = side_attention / side_attention.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        axial_features = (side_attention[..., None] * side_features).sum(dim=2)
        axial_valid = (side_valid.sum(dim=2) > 0.5).float()

        if sagittal_keep is None:
            sagittal_keep = torch.ones_like(sagittal_valid)
        if axial_keep is None:
            axial_keep = torch.ones_like(axial_valid)
        sagittal_fusion_valid = (sagittal_valid * sagittal_keep).clamp(0, 1)
        axial_fusion_valid = (axial_valid * axial_keep).clamp(0, 1)
        both_valid = sagittal_fusion_valid * axial_fusion_valid

        sagittal_quality_input = (
            sagittal_quality * sagittal_fusion_valid
        ).unsqueeze(-1)
        axial_quality_input = (axial_quality * axial_fusion_valid).unsqueeze(-1)
        gate_input = torch.cat(
            [
                sagittal_features,
                axial_features,
                sagittal_features * axial_features,
                (sagittal_features - axial_features).abs(),
                sagittal_quality_input,
                axial_quality_input,
                sagittal_fusion_valid.unsqueeze(-1),
                axial_fusion_valid.unsqueeze(-1),
            ],
            dim=-1,
        )
        learned_candidate = self._bounded_gate(gate_input)
        if fixed_axial_gate is None:
            gate_candidate = learned_candidate
        else:
            if not 0.0 <= float(fixed_axial_gate) <= 1.0:
                raise ValueError(f"fixed_axial_gate must be in [0,1], got {fixed_axial_gate}")
            gate_candidate = torch.full_like(learned_candidate, float(fixed_axial_gate))

        # Strict fallback: both views use the candidate; one-view cases use that view at 100%.
        gate_effective = torch.where(
            both_valid > 0.5,
            gate_candidate,
            torch.where(
                axial_fusion_valid > 0.5,
                torch.ones_like(gate_candidate),
                torch.zeros_like(gate_candidate),
            ),
        )

        residual_input = torch.cat(
            [
                sagittal_features,
                axial_features,
                sagittal_features * axial_features,
                (sagittal_features - axial_features).abs(),
            ],
            dim=-1,
        )
        residual = (
            self.residual(residual_input).squeeze(-1)
            * self.residual_scale
            * both_valid
        )
        base = (1.0 - gate_effective) * sagittal_logits + gate_effective * axial_logits
        fused_logits = (base + residual) * torch.maximum(
            sagittal_fusion_valid, axial_fusion_valid
        )

        return {
            "nerve_visual_logits": fused_logits,
            "nerve_sagittal_logits": sagittal_logits,
            "nerve_axial_logits": axial_logits,
            "nerve_side_logits": side_logits,
            "nerve_fusion_gate": gate_effective,
            "nerve_gate_candidate": gate_candidate,
            "nerve_learned_gate_candidate": learned_candidate,
            "nerve_fusion_both_valid": both_valid,
            "nerve_fusion_residual": residual,
            "sagittal_features": sagittal_features,
            "axial_features": axial_features,
            "side_attention": side_attention,
            "sagittal_nerve_crops_preview": sagittal_crops.detach(),
            "sagittal_nerve_centers": center,
            "sagittal_nerve_half_sizes": half,
            "sagittal_nerve_angles": angle,
            "axial_valid": axial_valid,
        }


