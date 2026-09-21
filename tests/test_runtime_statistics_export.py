"""Ensure deployment exports cannot accidentally include patient tensors."""
from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_frozen_runtime_statistics import export


def sources(tmp_path, invalid=False):
    train, ref = tmp_path/'train.pt', tmp_path/'reference.pt'
    torch.save({'labels': torch.ones(3,16), 'patient_ids': ['private'],
        'geometry_feature_names': ['geometry'], 'metadata': {'lordosis_rule': {
            'flat_feature':'geometry', 'flat_threshold':0.2, 'template_threshold':0.3,
            'flat_scale':1., 'template_scale':2., 'normal_template':[[0.,0.],[1.,1.]],
            'patient_records':['private']}}}, train)
    torch.save({'coordinate_feature_names':[str(i) for i in range(47)],
        'coordinate_normalizer_mean':torch.zeros(47),
        'coordinate_normalizer_std':torch.zeros(47) if invalid else torch.ones(47),
        'patients':['private']}, ref)
    return train, ref


def test_export_keeps_only_frozen_constants(tmp_path):
    train, ref = sources(tmp_path)
    report = export(train, ref, tmp_path/'out')
    a = torch.load(tmp_path/'out/train_visual_evidence_cache.pt', weights_only=False)
    b = torch.load(tmp_path/'out/external_segment_ar_dataset.pt', weights_only=False)
    assert set(a) == {'metadata','geometry_feature_names'}
    assert 'patient_records' not in a['metadata']['lordosis_rule']
    assert set(b) == {'coordinate_feature_names','coordinate_normalizer_mean','coordinate_normalizer_std'}
    assert len(report['sources']) == len(report['exports']) == 2


def test_invalid_normalizer_writes_nothing(tmp_path):
    train, ref = sources(tmp_path, invalid=True)
    with pytest.raises(ValueError, match='normalizer'):
        export(train, ref, tmp_path/'out')
    assert not (tmp_path/'out').exists()
