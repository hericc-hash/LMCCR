"""Archive integrity and confinement tests; no model weights or downloads."""
import hashlib
import io
from pathlib import Path
import runpy
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
installer = runpy.run_path(str(ROOT / 'scripts/install_weight_bundle.py'))


def bundle(path, content=b'checkpoint', corrupt=False, unsafe=None):
    entries = {
        'coordinate_prior/HR320_v7_best.pt': content,
        'SHA256SUMS.txt': (hashlib.sha256(b'wrong' if corrupt else content).hexdigest()
                          + '  ./coordinate_prior/HR320_v7_best.pt\n').encode(),
        'FILE_MANIFEST.txt': b'local test',
    }
    with tarfile.open(path, 'w:gz') as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo('bundle/' + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if unsafe:
            info = tarfile.TarInfo(unsafe)
            info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))


def test_verified_install_is_repeatable_and_does_not_replace_different_weights(tmp_path):
    archive, weights = tmp_path / 'weights.tar.gz', tmp_path / 'weights'
    bundle(archive)
    result = installer['install'](archive, weights)
    target = weights / 'coordinate_prior/hr320_v7.pt'
    assert result['verified_payloads'] == 1
    assert target.read_bytes() == b'checkpoint'
    assert installer['install'](archive, weights) == result
    target.write_bytes(b'existing different weight')
    with pytest.raises(FileExistsError):
        installer['install'](archive, weights)
    assert target.read_bytes() == b'existing different weight'


def test_bad_hash_never_installs_payload(tmp_path):
    archive, weights = tmp_path / 'bad.tar.gz', tmp_path / 'weights'
    bundle(archive, corrupt=True)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        installer['install'](archive, weights)
    assert not (weights / 'coordinate_prior/hr320_v7.pt').exists()


@pytest.mark.parametrize('name', ['bundle/../../outside.pt', '/outside.pt', 'bundle/C:/outside.pt',
                                  'bundle/NUL.pt', 'bundle/weight.pt.'])
def test_unsafe_archive_paths_are_rejected(tmp_path, name):
    archive = tmp_path / 'bad.tar.gz'
    bundle(archive, unsafe=name)
    with pytest.raises(ValueError, match='Unsafe archive path'):
        installer['install'](archive, tmp_path / 'weights')


def test_payload_cannot_overwrite_installation_receipt(tmp_path):
    archive = tmp_path / 'bad.tar.gz'
    name = '_packages/bundle/INSTALLATION.json'
    entries = {name: b'fake receipt', 'FILE_MANIFEST.txt': b'test',
               'SHA256SUMS.txt': (hashlib.sha256(b'fake receipt').hexdigest() + '  ' + name).encode()}
    with tarfile.open(archive, 'w:gz') as tar:
        for filename, data in entries.items():
            member = tarfile.TarInfo('bundle/' + filename)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match='reserved installation metadata'):
        installer['install'](archive, tmp_path / 'weights')
    assert not (tmp_path / 'weights/_packages/bundle/INSTALLATION.json').exists()
