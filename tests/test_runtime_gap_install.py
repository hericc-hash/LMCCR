"""Runtime supplement integrity, confinement and conflict regression tests."""
import hashlib
from pathlib import Path
import sys
import zipfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from install_runtime_gaps import install, READER


def package(path, corrupt=False, extra=None):
    payload = {
        READER + 'configs/reader50_frozen.json': b'{}',
        '04_planner_v17/metadata_inventory/config_used.json': b'{"version":"test"}',
        '07_selector_v22_exact/frozen_metadata/CANONICAL_LEXICALIZER.json': b'{}',
        '05_direct_qwen_auxiliary/output_metadata/SELECTION.json': b'{}',
        '08_selector_missing_ensemble_weights/fact_seed20260932.pt': b'weight',
    }
    sums = '\n'.join(hashlib.sha256(v).hexdigest() + '  ' + k for k,v in payload.items())
    if corrupt:
        payload['08_selector_missing_ensemble_weights/fact_seed20260932.pt'] = b'tampered'
    with zipfile.ZipFile(path, 'w') as z:
        for k,v in payload.items():
            z.writestr('bundle/' + k, v)
        for name in ('COLLECTION_GATE.json','COLLECTION_SUMMARY.md','DICOM_VISUAL_CANDIDATES.csv','FILE_MANIFEST.csv'):
            z.writestr('bundle/00_manifest/' + name, b'')
        z.writestr('bundle/00_manifest/SHA256SUMS.txt', sums)
        if extra:
            z.writestr(extra, b'unlisted')


def test_runtime_install_preserves_assets_and_provenance(tmp_path):
    archive = tmp_path/'runtime.zip'
    package(archive)
    weights = tmp_path/'weights'
    receipt = install(archive, weights)
    assert receipt['verified_package_files'] == 5
    assert receipt['published'] is False
    assert (weights/'selector/fact_seed20260932.pt').read_bytes() == b'weight'
    assert install(archive, weights) == receipt
    (weights/'clinical_planner/config_used.json').write_text('existing')
    with pytest.raises(FileExistsError):
        install(archive, weights)
    assert (weights/'clinical_planner/config_used.json').read_text() == 'existing'


def test_bad_runtime_hash_installs_nothing(tmp_path):
    archive = tmp_path/'runtime.zip'
    package(archive, corrupt=True)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        install(archive, tmp_path/'weights')
    assert not (tmp_path/'weights').exists()


@pytest.mark.parametrize('extra', ['bundle/../../outside.py', 'bundle/NUL.py', 'bundle/unlisted.py'])
def test_unsafe_or_unlisted_runtime_files_install_nothing(tmp_path, extra):
    archive = tmp_path/'runtime.zip'
    package(archive, extra=extra)
    with pytest.raises(ValueError):
        install(archive, tmp_path/'weights')
    assert not (tmp_path/'weights').exists()
