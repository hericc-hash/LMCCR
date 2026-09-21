"""Install the trusted runtime supplement locally; never publish its contents."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
from install_weight_bundle import safe_relative, digest

ROOT = Path(__file__).resolve().parents[1]
READER = '01_reader50_frozen_runtime/directory_sources/LMCCR_READER50_FINAL_FROZEN_20260911/'
MODELS = '02_discovered_runtime_candidates/dicom_preprocess/autodl-tmp/src/models/'


def install(archive, weights, supplemental_models=None):
    weights = Path(weights).resolve()
    pending = {}
    records = []

    def stage(relative, data, source):
        target = (weights / safe_relative(relative)).resolve()
        if not target.is_relative_to(weights):
            raise ValueError('Destination escapes weights')
        if target in pending and pending[target] != data:
            raise ValueError('Conflicting destination: ' + relative)
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(target)
        pending[target] = data
        records.append({'source': source, 'destination': relative,
                        'sha256': hashlib.sha256(data).hexdigest()})

    with zipfile.ZipFile(archive) as z:
        files, roots = {}, set()
        for info in z.infolist():
            p = safe_relative(info.filename)
            roots.add(p.parts[0])
            if info.is_dir():
                continue
            if len(p.parts) < 2 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Invalid ZIP member')
            relative = PurePosixPath(*p.parts[1:]).as_posix()
            if relative in files:
                raise ValueError('Duplicate ZIP path')
            files[relative] = z.read(info)
        if len(roots) != 1:
            raise ValueError('Expected one package root')
        sums = {}
        for line in files['00_manifest/SHA256SUMS.txt'].decode().splitlines():
            sha, name = line.split(maxsplit=1)
            name = safe_relative(name).as_posix()
            if name in sums:
                raise ValueError('Duplicate checksum entry')
            sums[name] = sha
        allowed = {'00_manifest/' + n for n in ('COLLECTION_GATE.json', 'COLLECTION_SUMMARY.md',
                   'DICOM_VISUAL_CANDIDATES.csv', 'FILE_MANIFEST.csv', 'SHA256SUMS.txt')}
        if set(files) - set(sums) != allowed or set(sums) - set(files):
            raise ValueError('Incomplete checksum coverage')
        for name, sha in sums.items():
            if hashlib.sha256(files[name]).hexdigest() != sha:
                raise ValueError('Checksum mismatch: ' + name)
        for name, data in files.items():
            if name.startswith(READER):
                stage('runtime/reader50/' + name[len(READER):], data, name)
            elif name.startswith(MODELS) and name.endswith('.py'):
                stage('runtime/models/' + name[len(MODELS):], data, name)
            elif name.startswith('08_selector_missing_ensemble_weights/'):
                stage('selector/' + PurePosixPath(name).name, data, name)
        assets = {
            'clinical_planner/config_used.json': '04_planner_v17/metadata_inventory/config_used.json',
            'report_realizer/CANONICAL_LEXICALIZER.json': '07_selector_v22_exact/frozen_metadata/CANONICAL_LEXICALIZER.json',
            'direct/SELECTION.json': '05_direct_qwen_auxiliary/output_metadata/SELECTION.json',
        }
        for dest, source in assets.items():
            stage(dest, files[source], source)
        # Optional local, provenance-recorded source dependencies absent from the
        # supplement. Never override files supplied by the new frozen package.
        if supplemental_models:
            for source in sorted(Path(supplemental_models).glob('stage1_19i49*.py')):
                dest = 'runtime/models/' + source.name
                if (weights / dest).resolve() not in pending:
                    stage(dest, source.read_bytes(), str(source.resolve()))
        stage('runtime/models/__init__.py', b'', 'generated empty package marker')
        receipt = {'archive_sha256': digest(archive), 'verified_package_files': len(sums),
                   'installed_files': records, 'published': False}
        receipt_path = (weights / 'runtime/INSTALLATION.json').resolve()
        if not receipt_path.is_relative_to(weights) or receipt_path in pending:
            raise ValueError('Unsafe installation receipt')
        for target, data in pending.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--weights', default=str(ROOT / 'weights'))
    p.add_argument('--supplemental-models', help='Optional trusted historical models directory')
    a = p.parse_args()
    r = install(a.archive, a.weights, a.supplemental_models)
    print(f"Verified {r['verified_package_files']} files; installed {len(r['installed_files'])} runtime files locally.")
