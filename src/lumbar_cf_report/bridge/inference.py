"""Raw-E inference boundary; labels and reference reports are not model inputs."""
from __future__ import annotations

import torch


@torch.inference_mode()
def predict_clinical16(model, evidence, thresholds, lordosis_valid):
    """Apply explicitly supplied Development-frozen 16-slot thresholds.

    ``raw_E`` is required even if reconstructed evidence is also present.
    ``lordosis_valid`` is an input-availability mask, not a ground-truth label.
    """
    required = ('global_source', 'raw_E', 'coordinate_state', 'task_quality', 'task_valid')
    missing = [key for key in required if key not in evidence]
    if missing:
        raise ValueError(f'Missing factual evidence fields: {missing}; Raw-E is required')
    device = next(model.parameters()).device
    inputs = [torch.as_tensor(evidence[key], dtype=torch.float32, device=device) for key in required]
    batch = inputs[1].shape[0]
    if inputs[1].shape != (batch, 5, 3, 128):
        raise ValueError('raw_E must have shape [N,5,3,128]')
    threshold = torch.as_tensor(thresholds, dtype=torch.float32, device=device)
    if threshold.shape != (16,) or not torch.isfinite(threshold).all() or not ((threshold >= 0) & (threshold <= 1)).all():
        raise ValueError('Explicit finite Development-frozen thresholds [16] in [0,1] required')
    lord = torch.as_tensor(lordosis_valid, dtype=torch.float32, device=device)
    if lord.shape != (batch,) or not ((lord == 0) | (lord == 1)).all():
        raise ValueError('lordosis_valid must be a binary [N] input mask')
    if not all(torch.isfinite(x).all() for x in inputs):
        raise ValueError('Factual evidence must be finite')
    if not ((inputs[4] == 0) | (inputs[4] == 1)).all():
        raise ValueError('task_valid must be a binary input mask')
    if model.training:
        raise ValueError('Frozen Planner must be in eval mode')
    output = model(*inputs, return_plan_tokens=False)
    probabilities = output['main_probabilities']
    valid = torch.cat([lord[:, None], inputs[4].permute(0, 2, 1).reshape(batch, 15)], dim=1)
    return {'planner_probs': probabilities, 'core_binary': ((probabilities >= threshold) & valid.bool()).long(),
            'slot_valid': valid.long(), 'planner_thresholds': threshold}
