#!/usr/bin/env python3
"""Replay the frozen Planner, then infer Clinical16 from a private Raw-E batch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from lumbar_cf_report.bridge.stage23_exact import load_runtime, validate_saved_internal_replay, build_exact_model
from lumbar_cf_report.bridge.inference import predict_clinical16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, help='Frozen v1.7 output directory including config and replay assets')
    parser.add_argument('--input', required=True, help='Private .pt mapping with Raw-E tensors and lordosis_valid')
    parser.add_argument('--thresholds', required=True, help='JSON containing Development-frozen thresholds [16]')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg, ck, data, saved, _ = load_runtime(args.runtime)
    replay = validate_saved_internal_replay(cfg, ck, data, saved)
    model = build_exact_model(cfg, ck, args.device)
    batch = torch.load(args.input, map_location='cpu', weights_only=False)
    threshold_payload = json.loads(Path(args.thresholds).read_text(encoding='utf-8'))
    if threshold_payload.get('source_cohort') != 'Development388':
        raise ValueError('Thresholds must be frozen on Development388')
    result = predict_clinical16(model, batch, threshold_payload['thresholds'], batch['lordosis_valid'])
    result = {key: value.cpu() for key, value in result.items()}
    result['replay'] = replay
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)


if __name__ == '__main__':
    main()
