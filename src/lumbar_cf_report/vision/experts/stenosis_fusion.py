#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded feature-level dual-view fusion for the stenosis evidence expert."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class BoundedStenosisFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        gate_min: float = 0.30,
        gate_max: float = 0.70,
        gate_initial: float = 0.42,
        residual_scale: float = 0.20,
    ):
        super().__init__()
        if not (0.0 <= gate_min < gate_max <= 1.0):
            raise ValueError(f"Invalid gate bounds: [{gate_min}, {gate_max}]")
        if not gate_min <= gate_initial <= gate_max:
            raise ValueError("gate_initial must be inside gate bounds")
        self.hidden_dim = int(hidden_dim)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.gate_initial = float(gate_initial)
        self.residual_scale = float(residual_scale)

        self.gate = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim + 4),
            nn.Linear(4 * hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.interaction = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim + 1),
            nn.Linear(4 * hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self._initialize_gate()

    def _initialize_gate(self) -> None:
        final = self.gate[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        relative = (self.gate_initial - self.gate_min) / (self.gate_max - self.gate_min)
        relative = min(max(relative, 1e-5), 1.0 - 1e-5)
        nn.init.constant_(final.bias, math.log(relative / (1.0 - relative)))

    def initialize_classifier_from_sagittal(self, sagittal_linear: nn.Linear) -> None:
        if self.classifier.weight.shape != sagittal_linear.weight.shape:
            raise ValueError(
                f"Classifier shape mismatch: fusion={tuple(self.classifier.weight.shape)} "
                f"sagittal={tuple(sagittal_linear.weight.shape)}"
            )
        with torch.no_grad():
            self.classifier.weight.copy_(sagittal_linear.weight)
            self.classifier.bias.copy_(sagittal_linear.bias)

    def _bounded_gate(self, gate_input: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.gate(gate_input).squeeze(-1))
        return self.gate_min + (self.gate_max - self.gate_min) * raw

    def forward(
        self,
        sagittal_features: torch.Tensor,
        axial_features: torch.Tensor,
        sagittal_logits: torch.Tensor,
        axial_logits: torch.Tensor,
        sagittal_valid: torch.Tensor,
        axial_valid: torch.Tensor,
        sagittal_quality: torch.Tensor,
        axial_quality: torch.Tensor,
        sagittal_keep: Optional[torch.Tensor] = None,
        axial_keep: Optional[torch.Tensor] = None,
        fixed_axial_gate: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if sagittal_keep is None:
            sagittal_keep = torch.ones_like(sagittal_valid)
        if axial_keep is None:
            axial_keep = torch.ones_like(axial_valid)
        sag_fusion_valid = (sagittal_valid * sagittal_keep).clamp(0, 1)
        ax_fusion_valid = (axial_valid * axial_keep).clamp(0, 1)
        both_valid = sag_fusion_valid * ax_fusion_valid

        gate_input = torch.cat(
            [
                sagittal_features,
                axial_features,
                sagittal_features * axial_features,
                (sagittal_features - axial_features).abs(),
                (sagittal_quality * sag_fusion_valid).unsqueeze(-1),
                (axial_quality * ax_fusion_valid).unsqueeze(-1),
                sag_fusion_valid.unsqueeze(-1),
                ax_fusion_valid.unsqueeze(-1),
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

        # Strict fallback: candidate only applies when both views are valid.
        gate_effective = torch.where(
            both_valid > 0.5,
            gate_candidate,
            torch.where(
                ax_fusion_valid > 0.5,
                torch.ones_like(gate_candidate),
                torch.zeros_like(gate_candidate),
            ),
        )
        base_feature = (1.0 - gate_effective[..., None]) * sagittal_features + gate_effective[..., None] * axial_features
        interaction_input = torch.cat(
            [
                sagittal_features,
                axial_features,
                sagittal_features * axial_features,
                (sagittal_features - axial_features).abs(),
                axial_quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        interaction = self.interaction(interaction_input) * both_valid[..., None]
        fused_feature = self.output_norm(base_feature + self.residual_scale * interaction)
        learned_fused_logits = self.classifier(fused_feature).squeeze(-1)

        # Exact one-view fallback preserves the strong the pretrained sagittal expert predictor.
        fused_logits = torch.where(
            both_valid > 0.5,
            learned_fused_logits,
            torch.where(
                sag_fusion_valid > 0.5,
                sagittal_logits,
                torch.where(ax_fusion_valid > 0.5, axial_logits, torch.zeros_like(axial_logits)),
            ),
        )
        return {
            "fused_logits": fused_logits,
            "fused_features": fused_feature,
            "gate_candidate": gate_candidate,
            "learned_gate_candidate": learned_candidate,
            "gate_effective": gate_effective,
            "both_valid": both_valid,
            "sagittal_fusion_valid": sag_fusion_valid,
            "axial_fusion_valid": ax_fusion_valid,
            "interaction": interaction,
        }

