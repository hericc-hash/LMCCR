"""Verify and install a local LMCCR archive. Does not stage or publish weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RENAMES = {
    'coordinate_prior/HR320_v7_best.pt': 'coordinate_prior/hr320_v7.pt',
    'visual_evidence/best_visual_evidence_branch.pt': 'visual_evidence/lumbar_visual_evidence.pt',
    'apreb/best_stage1_18bv3_counterfactual_bottleneck.pt': 'apreb/apreb.pt',
    'ccmrt/best_joint.pt': 'ccmrt/ccmrt.pt',
    'clinical_planner/stage2_3B_v1_7_selected.pt': 'clinical_planner/05_stage2_3B_selected_candidate.pt',
}
PREFIXES = {
    'report_realizer/r30_language_adapter/': 'report_realizer/language_adapter/',
    'report_realizer/r31_realizer_adapter/': 'report_realizer/dual_channel_adapter/',
    'report_realizer/v32_scaffold_adapter/': 'report_realizer/scaffold_adapter/',
    'selector_v22/': 'selector/',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                *(f'LPT{i}' for i in range(1, 10))}
    if (p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or not p.parts
            or any(part.endswith((' ', '.')) or part.split('.')[0].upper() in reserved
                   for part in p.parts)):
        raise ValueError(f'Unsafe archive path: {name}')
    return p


def destination(relative):
    if relative in RENAMES:
        return RENAMES[relative]
    for before, after in PREFIXES.items():
        if relative.startswith(before):
            return after + relative[len(before):]
    return relative


def install(archive, weights):
    archive, weights = Path(archive).resolve(), Path(weights).resolve()
    weights.mkdir(parents=True, exist_ok=True)
    # Only explicitly named regular files are read: no extractall, symlinks or
    # archive-controlled filesystem writes. Staging remains inside weights.
    with tempfile.TemporaryDirectory(prefix='.verified-', dir=weights) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(weights):
            raise ValueError('Staging escaped weights directory')
        records = []
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            files = {}
            roots = set()
            for member in members:
                p = safe_relative(member.name)
                roots.add(p.parts[0])
                if member.isdir():
                    continue
                if not member.isfile() or len(p.parts) < 2:
                    raise ValueError('Only regular files under one package root are supported')
                relative = PurePosixPath(*p.parts[1:]).as_posix()
                if relative in files:
                    raise ValueError('Duplicate archive path: ' + relative)
                files[relative] = member
            if len(roots) != 1:
                raise ValueError('Expected one archive root')
            package = next(iter(roots))
            receipt_path = (weights / '_packages' / package / 'INSTALLATION.json').resolve()
            if not receipt_path.is_relative_to(weights):
                raise ValueError('Installation receipt escapes weights directory')
            checksums = tar.extractfile(files['SHA256SUMS.txt']).read().decode('utf-8')
            expected = {}
            for line in checksums.splitlines():
                if not line.strip():
                    continue
                checksum, filename = line.split(maxsplit=1)
                relative = safe_relative(filename.removeprefix('*')).as_posix()
                if relative in expected or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
                    raise ValueError('Invalid checksum entry')
                expected[relative] = checksum
            if set(files) - set(expected) != {'SHA256SUMS.txt', 'FILE_MANIFEST.txt'} or set(expected) - set(files):
                raise ValueError('Checksum manifest does not cover every payload file')
            targets = set()
            for relative, member in files.items():
                if relative in expected and safe_relative(relative).parts[0].casefold() == '_packages':
                    raise ValueError('Payload uses reserved installation metadata directory')
                dest = destination(relative) if relative in expected else f'_packages/{package}/{relative}'
                target = (weights / safe_relative(dest)).resolve()
                if not target.is_relative_to(weights) or target in targets:
                    raise ValueError('Unsafe or colliding destination')
                targets.add(target)
                staged = staging / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, staged.open('wb') as output:
                    shutil.copyfileobj(source, output)
                actual = digest(staged)
                if relative in expected and actual != expected[relative]:
                    raise ValueError('Checksum mismatch: ' + relative)
                if target.exists() and digest(target) != actual:
                    raise FileExistsError('Refusing to replace different existing asset: ' + str(target))
                records.append({'source': relative, 'destination': dest, 'sha256': actual,
                                'bytes': member.size, 'manifest_verified': relative in expected})
        # Commit only after all archive payloads and existing destinations pass.
        for record in records:
            target = weights / record['destination']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(staging / record['source'], target)
        receipt = {'archive_sha256': digest(archive), 'package': package,
                   'verified_payloads': len(expected), 'files': records, 'published': False}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--weights', default=str(ROOT / 'weights'))
    args = parser.parse_args()
    result = install(args.archive, args.weights)
    print(f"Installed {result['verified_payloads']} SHA256-verified payloads locally; nothing published.")


if __name__ == '__main__':
    main()
