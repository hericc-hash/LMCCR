"""Install the trusted heritage patch with byte checks only; no model loading."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
from install_weight_bundle import safe_relative, digest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    'weights/runtime_assets/lordosis_geometry_constants.pt',
    'weights/runtime_assets/coordinate_normalizer.pt',
    'weights/direct/epoch_2/direct_prefix.pt',
    'weights/direct/epoch_2/lora/adapter_model.safetensors',
    'weights/direct/epoch_2/lora/adapter_config.json',
}


def install(archive, weights):
    weights = Path(weights).resolve()
    with zipfile.ZipFile(archive) as z:
        files, roots = {}, set()
        for item in z.infolist():
            path = safe_relative(item.filename)
            roots.add(path.parts[0])
            if item.is_dir():
                continue
            if len(path.parts) < 2 or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Invalid archive member')
            name = PurePosixPath(*path.parts[1:]).as_posix()
            if name in files:
                raise ValueError('Duplicate archive member')
            files[name] = z.read(item)
        if len(roots) != 1 or not REQUIRED.issubset(files):
            raise ValueError('Expected one heritage patch root and all five runtime assets')
        sums = {}
        for line in files['provenance/SHA256SUMS.txt'].decode().splitlines():
            sha, name = line.split(maxsplit=1)
            name = safe_relative(name).as_posix()
            if name in sums:
                raise ValueError('Duplicate checksum')
            sums[name] = sha
        excluded = {'provenance/SHA256SUMS.txt', 'provenance/FILE_MANIFEST.json', 'provenance/EXPORT_GATE.json'}
        if set(files) - set(sums) != excluded or set(sums) - set(files):
            raise ValueError('Unexpected checksum coverage')
        for name, sha in sums.items():
            if hashlib.sha256(files[name]).hexdigest() != sha:
                raise ValueError('Checksum mismatch: ' + name)
        pending, records = {}, []
        metadata = '_packages/' + next(iter(roots))
        for name, data in files.items():
            if name.startswith('weights/'):
                if name not in REQUIRED and not name.startswith('weights/direct/epoch_2/'):
                    raise ValueError('Unexpected runtime asset: ' + name)
                dest = name[len('weights/'):]
            else:
                dest = metadata + '/' + name
            target = (weights / safe_relative(dest)).resolve()
            if not target.is_relative_to(weights) or target in pending:
                raise ValueError('Unsafe or colliding destination')
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError('Refusing to overwrite different asset: ' + str(target))
            pending[target] = data
            records.append({'destination': dest, 'sha256': hashlib.sha256(data).hexdigest(),
                            'manifest_verified': name in sums})
        receipt_path = (weights / metadata / 'INSTALLATION.json').resolve()
        if not receipt_path.is_relative_to(weights) or receipt_path in pending:
            raise ValueError('Unsafe receipt path')
        for path, data in pending.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
        receipt = {'archive_sha256': digest(archive), 'files': records,
                   'verified_files': len(sums), 'runtime_assets': sorted(REQUIRED),
                   'model_loading_executed': False, 'inference_executed': False, 'published': False}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--weights', default=str(ROOT/'weights'))
    a = p.parse_args()
    r = install(a.archive, a.weights)
    print(f"Installed five runtime assets; {r['verified_files']} file hashes matched. No models loaded or tested.")
