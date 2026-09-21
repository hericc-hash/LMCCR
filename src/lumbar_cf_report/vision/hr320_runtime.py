"""HR320-v7 inference on the de-identified dataset's annotated center slices.

Annotations select the image only; landmark coordinates are read after inference
for diagnostics. This is not automatic series/slice selection or full reporting.
Preprocessing follows Stage1.13h2 manual-center HR320: per-image 1/99 percentiles,
uint8 PIL bilinear square resize, then raw-frame normalized coordinate prediction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .coordinate_prior import ResAttentionSagittalPriorNet

LEVELS6 = ('T12/L1', 'L1/2', 'L2/3', 'L3/4', 'L4/5', 'L5/S1')
HR320_SHA256 = '3c7158a9b5d96f2e7d3e7c7eeaea2158285fe9313ea1401eedfb472872f8718d'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Dataset path escapes dataset root')
    return path


def preprocess(image, size=320):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError('Expected a finite two-dimensional DICOM image')
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        raise ValueError('Constant image cannot provide localization evidence')
    normalized = (np.clip(image, low, high) - low) / max(float(high - low), 1e-6)
    quantized = (normalized * 255).astype(np.uint8)
    resized = Image.fromarray(quantized).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def load_hr320(checkpoint, device='cpu'):
    digest = sha256(checkpoint)
    if digest != HR320_SHA256:
        raise ValueError('Checkpoint hash differs from the supplied frozen HR320-v7 best.pt')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args = payload['args']
    expected = {'arch': 'resatt_unet', 'out_hw': 320, 'base_ch': 32,
                'softargmax_beta': 30.0, 'dropout': 0.05}
    if any(args.get(k) != v for k, v in expected.items()):
        raise ValueError('HR320-v7 architecture/preprocessing metadata mismatch')
    model = ResAttentionSagittalPriorNet(base_ch=args['base_ch'],
        softargmax_beta=args['softargmax_beta'], dropout=args['dropout'])
    # Unlike the historical >=90% loader, reproduction requires every key.
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, {'sha256': digest, 'state_keys': len(model.state_dict()),
                   'strict': True, 'model_args': expected}


def canonical_coordinates(coords, shape, iop, visual_hw=384):
    """Exact transpose/flip/padding convention of Stage1.13h2 canonical_point."""
    iop = np.asarray(iop, dtype=np.float64)
    if iop.shape != (6,) or not np.isfinite(iop).all():
        raise ValueError('Missing or invalid DICOM ImageOrientationPatient')
    col, row = iop[:3], iop[3:]
    if min(np.linalg.norm(col), np.linalg.norm(row)) < 1e-8:
        raise ValueError('Degenerate DICOM orientation')
    col, row = col / np.linalg.norm(col), row / np.linalg.norm(row)
    tc, tr = np.array([0., 1., 0.]), np.array([0., 0., -1.])
    h, w = shape
    xy = np.asarray(coords, dtype=np.float64).copy() * [w - 1, h - 1]
    if abs(row @ tc) + abs(col @ tr) > abs(col @ tc) + abs(row @ tr):
        h, w = w, h
        xy = xy[:, ::-1].copy()
        col, row = row, col
    if col @ tc < 0:
        xy[:, 0] = w - 1 - xy[:, 0]
        col = -col
    if row @ tr < 0:
        xy[:, 1] = h - 1 - xy[:, 1]
        row = -row
    scale = min(visual_hw / w, visual_hw / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    xy = (xy * scale + [(visual_hw - nw) // 2, (visual_hw - nh) // 2]) / (visual_hw - 1)
    return np.clip(xy, 0, 1), float(min(abs(col @ tc), abs(row @ tr)))


def run_prior_smoke(dataset_root, checkpoint, output, cohort='Development388', limit=8, device='cpu'):
    import pydicom
    if limit < 1:
        raise ValueError('limit must be positive')
    if cohort not in ('Development388', 'Internal100', 'Independent49'):
        raise ValueError('Unknown cohort')
    root, output = Path(dataset_root).resolve(), Path(output)
    cases = root / cohort / 'cases'
    if not cases.is_dir():
        raise FileNotFoundError(cases)
    output.mkdir(parents=True, exist_ok=True)
    model, load_report = load_hr320(checkpoint, device)
    predictions, errors, skipped = [], [], 0
    selected = []
    for case in sorted(cases.iterdir()):
        annotations = sorted((case / 'annotations').glob('*.json'))
        if not annotations:
            skipped += 1
            continue
        # Multiple annotations require an explicit choice rather than a silent fallback.
        if len(annotations) != 1:
            errors.append({'case_id': case.name, 'error': 'ambiguous center annotations'})
            continue
        selected.append((case, annotations[0]))
        if len(selected) >= limit:
            break
    for case, annotation_path in selected:
        try:
            annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
            path = resolve_inside(root, annotation['dicom_path'])
            if not path.is_relative_to(case.resolve()) or annotation['case_id'] != case.name:
                raise ValueError('Annotation points to a different patient')
            ds = pydicom.dcmread(path)
            if str(ds.SOPInstanceUID) != annotation['slice_sop_uid']:
                raise ValueError('Annotation/DICOM SOP mismatch')
            image = np.asarray(ds.pixel_array, dtype=np.float32)
            image = image * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
            inp = torch.from_numpy(preprocess(image))[None, None].to(device)
            with torch.inference_mode():
                result = model(inp)
            coords = result['coords_norm'][0].cpu().numpy()
            if coords.shape != (6, 2) or not np.isfinite(coords).all() or not ((coords >= 0) & (coords <= 1)).all():
                raise ValueError('Invalid six-point model output')
            canonical, score = canonical_coordinates(coords[1:], image.shape, ds.ImageOrientationPatient)
            item = {'case_id': case.name, 'cohort': cohort,
                    'dicom_path': path.relative_to(root).as_posix(),
                    'coords_norm_raw': coords.tolist(), 'coords_norm_canonical_5': canonical.tolist(),
                    'orientation_score': score, 'image_shape': list(image.shape),
                    'finite_heatmaps': bool(torch.isfinite(result['heatmap_logits']).all())}
            if not item['finite_heatmaps']:
                raise ValueError('Nonfinite heatmaps')
            # Ground-truth coordinates do not enter preprocess/model/canonicalization.
            gt = np.asarray([annotation['keypoints'][level] for level in LEVELS6], dtype=float)
            if gt.shape != (6, 2) or not np.isfinite(gt).all():
                raise ValueError('Invalid diagnostic landmark annotations')
            delta = coords * [image.shape[1] - 1, image.shape[0] - 1] - gt
            spacing = np.asarray(ds.PixelSpacing, dtype=float)
            item['landmark_distance_mm'] = np.linalg.norm(delta * spacing[::-1], axis=1).tolist()
            predictions.append(item)
            print(f'HR320 progress: {len(predictions)}/{len(selected)} images', flush=True)
        except Exception as exc:
            errors.append({'case_id': case.name, 'error': str(exc)})
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    summary = {'status': 'PASS' if predictions and not errors and len(selected) == limit else 'FAIL',
               'scope': 'annotated-center HR320 execution smoke; NOT end-to-end reporting or held-out accuracy',
               'cohort': cohort, 'requested': limit, 'passed': len(predictions),
               'skipped_without_annotations_before_limit': skipped, 'errors': errors,
               'checkpoint': load_report, 'torch': torch.__version__, 'pydicom': pydicom.__version__,
               'device': str(device), 'diagnostic_mean_landmark_distance_mm':
                   float(np.mean([p['landmark_distance_mm'] for p in predictions])) if predictions else None}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
