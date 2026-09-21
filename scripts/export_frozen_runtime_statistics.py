"""Export only frozen inference constants from trusted original training assets.

Run on the original machine. Never fit statistics on the test patient/cohort.
The resulting small .pt files contain no per-patient feature or label arrays.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from install_weight_bundle import digest


def export(train_cache, coordinate_reference, output):
    train_cache, coordinate_reference, output = map(Path, (train_cache, coordinate_reference, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory')
    train = torch.load(train_cache, map_location='cpu', weights_only=False)
    ref = torch.load(coordinate_reference, map_location='cpu', weights_only=False)
    rule = train['metadata']['lordosis_rule']
    fields = ('flat_feature', 'flat_threshold', 'template_threshold', 'flat_scale', 'template_scale', 'normal_template')
    selected = {k: rule[k] for k in fields}
    constants = {'metadata': {'lordosis_rule': selected},
                 'geometry_feature_names': list(train['geometry_feature_names'])}
    coordinates = {k: ref[k] for k in ('coordinate_feature_names', 'coordinate_normalizer_mean', 'coordinate_normalizer_std')}
    mean = torch.as_tensor(coordinates['coordinate_normalizer_mean'])
    std = torch.as_tensor(coordinates['coordinate_normalizer_std'])
    if (mean.numel() != 47 or mean.shape != std.shape or len(coordinates['coordinate_feature_names']) != 47
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all()):
        raise ValueError('Invalid frozen 47-D normalizer')
    if not torch.isfinite(torch.as_tensor(selected['normal_template'])).all():
        raise ValueError('Invalid frozen template')
    if float(selected['flat_scale']) <= 0 or float(selected['template_scale']) <= 0:
        raise ValueError('Invalid frozen rule scales')
    output.mkdir(parents=True, exist_ok=True)
    torch.save(constants, output/'train_visual_evidence_cache.pt')
    torch.save(coordinates, output/'external_segment_ar_dataset.pt')
    provenance = {'scope': 'frozen inference constants only; no refit and no per-patient arrays',
                  'sources': {train_cache.name: digest(train_cache), coordinate_reference.name: digest(coordinate_reference)},
                  'exports': {p.name: digest(p) for p in sorted(output.glob('*.pt'))}}
    (output/'PROVENANCE.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return provenance


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cache', required=True)
    p.add_argument('--coordinate-reference', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(export(a.train_cache, a.coordinate_reference, a.output), indent=2))
