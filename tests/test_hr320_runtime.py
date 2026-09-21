import numpy as np
import pytest

from lumbar_cf_report.vision.hr320_runtime import canonical_coordinates, preprocess, resolve_inside


def test_raw_image_contract_and_reject_invalid_inputs():
    image = np.arange(80, dtype=np.float32).reshape(8, 10)
    result = preprocess(image)
    assert result.shape == (320, 320)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    for bad in (np.ones((8, 8)), np.full((8, 8), np.nan), np.zeros((3, 8, 8))):
        with pytest.raises(ValueError):
            preprocess(bad)


def test_orientation_flip_and_transpose_preserve_same_landmark():
    xy = np.array([[.2, .7]])
    expected, _ = canonical_coordinates(xy, (384, 384), [0, 1, 0, 0, 0, -1])
    flipped, score = canonical_coordinates([[.8, .7]], (384, 384), [0, -1, 0, 0, 0, -1])
    transposed, _ = canonical_coordinates([[.7, .2]], (384, 384), [0, 0, -1, 0, 1, 0])
    assert np.allclose(expected, flipped) and np.allclose(expected, transposed)
    assert score == 1


def test_dataset_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        resolve_inside(tmp_path, '../different_patient.dcm')
