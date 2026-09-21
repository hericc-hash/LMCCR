#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete dual-view context-crop stenosis evidence model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .stenosis_axial import AxialStenosisEncoder
from .stenosis_fusion import BoundedStenosisFusion
from .stenosis_sagittal import (
    SagittalStenosisEncoder,
)

VERSION = "DualView-ContextCrop-Stenosis-Evidence-v1"


class StenosisEvidenceExpert(nn.Module):
    VERSION = VERSION

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        roi_crop_size: int = 192,
        sag_crop_encoder_batch_size: int = 12,
        ax_crop_encoder_batch_size: int = 24,
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
        gate_min: float = 0.30,
        gate_max: float = 0.70,
        gate_initial: float = 0.42,
        fusion_residual_scale: float = 0.20,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.sagittal_branch = SagittalStenosisEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            roi_crop_size=roi_crop_size,
            crop_encoder_batch_size=sag_crop_encoder_batch_size,
            use_roi_mask_channel=use_roi_mask_channel,
            roi_pad_value=roi_pad_value,
            roi_min_half_size=roi_min_half_size,
            sag_canal_global_posterior_shift=sag_canal_global_posterior_shift,
            sag_canal_global_half_w=sag_canal_global_half_w,
            sag_canal_global_half_h=sag_canal_global_half_h,
            sag_canal_core_width_frac=sag_canal_core_width_frac,
            sag_canal_core_height_frac=sag_canal_core_height_frac,
            sag_canal_core_anterior_shift=sag_canal_core_anterior_shift,
            disc_angle_clip_deg=disc_angle_clip_deg,
            l5s1_angle_gain=l5s1_angle_gain,
            l5s1_angle_offset_deg=l5s1_angle_offset_deg,
            l5s1_angle_clip_deg=l5s1_angle_clip_deg,
            disc_angle_smooth=disc_angle_smooth,
        )
        self.axial_branch = AxialStenosisEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            crop_encoder_batch_size=ax_crop_encoder_batch_size,
        )
        self.fusion = BoundedStenosisFusion(
            hidden_dim=hidden_dim,
            dropout=dropout,
            gate_min=gate_min,
            gate_max=gate_max,
            gate_initial=gate_initial,
            residual_scale=fusion_residual_scale,
        )
        self.initialization_report: Dict[str, Any] = {}

    def load_pretrained_sagittal_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"Checkpoint root must be mapping, got {type(checkpoint).__name__}")
        if "model" not in checkpoint:
            raise KeyError(f"pretrained sagittal checkpoint lacks 'model'; keys={list(checkpoint.keys())}")
        state = checkpoint["model"]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint['model'] is not a state_dict mapping")

        expected = self.sagittal_branch.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        shape_mismatch = []
        for key in sorted(set(expected) & set(state)):
            if tuple(expected[key].shape) != tuple(state[key].shape):
                shape_mismatch.append({
                    "key": key,
                    "expected": list(expected[key].shape),
                    "checkpoint": list(state[key].shape),
                })
        if missing or unexpected or shape_mismatch:
            raise RuntimeError(
                "pretrained sagittal checkpoint compatibility failed:\n"
                + json.dumps(
                    {
                        "missing_keys": missing,
                        "unexpected_keys": unexpected,
                        "shape_mismatch": shape_mismatch,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        self.sagittal_branch.load_state_dict(state, strict=True)
        final = self.sagittal_branch.head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Unexpected pretrained sagittal final classifier type")
        self.fusion.initialize_classifier_from_sagittal(final)
        report = {
            "checkpoint_path": str(path.resolve()),
            "checkpoint_version": checkpoint.get("version"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "state_dict_field": "model",
            "tensor_count": len(state),
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatch": [],
            "strict_load_success": True,
            "fusion_classifier_initialized_from_sagittal": True,
            "source_record": checkpoint.get("record"),
            "source_args": checkpoint.get("args"),
            "source_training_contract": checkpoint.get("training_contract"),
        }
        self.initialization_report = report
        return report

    def configure_phase(self, epoch: int) -> Dict[str, Any]:
        """Apply the stenosis-expert freezing contract for a 1-indexed epoch."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        stage: str
        fixed_gate: Optional[float]
        if epoch <= 3:
            stage = "axial_warmup"
            fixed_gate = None
            for parameter in self.axial_branch.parameters():
                parameter.requires_grad_(True)
        elif epoch <= 6:
            stage = "fixed_fusion"
            fixed_gate = 0.40
            for parameter in self.axial_branch.parameters():
                parameter.requires_grad_(True)
            for parameter in self.fusion.interaction.parameters():
                parameter.requires_grad_(True)
            for parameter in self.fusion.output_norm.parameters():
                parameter.requires_grad_(True)
            for parameter in self.fusion.classifier.parameters():
                parameter.requires_grad_(True)
        elif epoch <= 10:
            stage = "bounded_gate_head_tune"
            fixed_gate = None
            for parameter in self.axial_branch.parameters():
                parameter.requires_grad_(True)
            for parameter in self.fusion.parameters():
                parameter.requires_grad_(True)
            self.sagittal_branch.unfreeze_head()
        elif epoch <= 16:
            stage = "limited_joint"
            fixed_gate = None
            for parameter in self.axial_branch.parameters():
                parameter.requires_grad_(True)
            for parameter in self.fusion.parameters():
                parameter.requires_grad_(True)
            self.sagittal_branch.unfreeze_last_stage()
        else:
            stage = "free_joint_limited_sagittal"
            fixed_gate = None
            for parameter in self.axial_branch.parameters():
                parameter.requires_grad_(True)
            for parameter in self.fusion.parameters():
                parameter.requires_grad_(True)
            self.sagittal_branch.unfreeze_last_stage()
        return {
            "stage": stage,
            "fixed_axial_gate": fixed_gate,
            "trainable_parameters": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
            "total_parameters": int(sum(p.numel() for p in self.parameters())),
        }

    def named_optimizer_groups(self) -> Dict[str, list[nn.Parameter]]:
        sag_backbone_early = []
        sag_last = []
        sag_head = list(self.sagittal_branch.head.parameters())
        for index, block in enumerate(self.sagittal_branch.encoder.backbone):
            target = sag_last if index == 7 else sag_backbone_early
            target.extend(list(block.parameters()))
        sag_last.extend(list(self.sagittal_branch.encoder.proj.parameters()))
        return {
            "sag_early": sag_backbone_early,
            "sag_last": sag_last,
            "sag_head": sag_head,
            "axial": list(self.axial_branch.parameters()),
            "fusion_gate": list(self.fusion.gate.parameters()),
            "fusion_other": [
                *self.fusion.interaction.parameters(),
                *self.fusion.output_norm.parameters(),
                *self.fusion.classifier.parameters(),
            ],
        }

    def forward(
        self,
        sagittal_images: torch.Tensor,
        sagittal_coords: torch.Tensor,
        sagittal_valid: torch.Tensor,
        sagittal_quality: torch.Tensor,
        axial_crops: torch.Tensor,
        axial_slice_valid: torch.Tensor,
        axial_quality: torch.Tensor,
        sagittal_keep: Optional[torch.Tensor] = None,
        axial_keep: Optional[torch.Tensor] = None,
        fixed_axial_gate: Optional[float] = None,
        axial_slice_permutation: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        sag = self.sagittal_branch(sagittal_images, sagittal_coords, sagittal_valid)
        axial = self.axial_branch(
            axial_crops,
            axial_slice_valid,
            slice_permutation=axial_slice_permutation,
        )
        fused = self.fusion(
            sagittal_features=sag["hidden_features"],
            axial_features=axial["features"],
            sagittal_logits=sag["logits"],
            axial_logits=axial["logits"],
            sagittal_valid=sagittal_valid,
            axial_valid=axial["axial_valid"],
            sagittal_quality=sagittal_quality,
            axial_quality=axial_quality,
            sagittal_keep=sagittal_keep,
            axial_keep=axial_keep,
            fixed_axial_gate=fixed_axial_gate,
        )
        return {
            "stenosis_visual_logits": fused["fused_logits"],
            "stenosis_fused_logits": fused["fused_logits"],
            "stenosis_sagittal_logits": sag["logits"],
            "stenosis_axial_logits": axial["logits"],
            "stenosis_fusion_gate": fused["gate_effective"],
            "stenosis_gate_candidate": fused["gate_candidate"],
            "stenosis_learned_gate_candidate": fused["learned_gate_candidate"],
            "stenosis_fusion_both_valid": fused["both_valid"],
            "stenosis_sagittal_fusion_valid": fused["sagittal_fusion_valid"],
            "stenosis_axial_fusion_valid": fused["axial_fusion_valid"],
            "sagittal_features": sag["hidden_features"],
            "axial_features": axial["features"],
            "axial_slice_attention": axial["slice_attention"],
            "axial_slice_features": axial["slice_features"],
            "axial_valid": axial["axial_valid"],
            "sagittal_global_crops_preview": sag["global_crops_preview"],
            "sagittal_core_crops_preview": sag["core_crops_preview"],
            "sagittal_global_centers": sag["global_centers"],
            "sagittal_core_centers": sag["core_centers"],
        }


