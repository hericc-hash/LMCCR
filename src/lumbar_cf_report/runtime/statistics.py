"""Read minimal frozen statistics without refitting or altering source files."""
from pathlib import Path


def load_frozen_statistics(weights_root):
    import torch
    root = Path(weights_root) / 'runtime_assets'
    geometry = torch.load(root / 'lordosis_geometry_constants.pt', map_location='cpu', weights_only=False)
    coordinates = torch.load(root / 'coordinate_normalizer.pt', map_location='cpu', weights_only=False)
    # The heritage export stores lordosis_rule at top level; the old visual
    # cache reader expects metadata.lordosis_rule. Adapt in memory, not on disk.
    legacy_geometry = {
        'geometry_feature_names': geometry['geometry_feature_names'],
        'metadata': {'lordosis_rule': geometry['lordosis_rule']},
    }
    legacy_coordinates = {key: coordinates[key] for key in (
        'coordinate_feature_names', 'coordinate_normalizer_mean', 'coordinate_normalizer_std')}
    return legacy_geometry, legacy_coordinates
