"""Attempt a single real DICOM case with frozen runtime sources, on CPU.

Stops at missing scientific assets instead of inventing normalization or drafts.
No annotations or reference reports are read. All outputs remain local.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def invoke(m, args):
    old = sys.argv
    try:
        sys.argv = [m.__file__, *map(str, args)]
        m.main()
    finally:
        sys.argv = old


def run(dataset, case_id, weights, output, threads=4):
    import torch
    import pandas as pd
    from lumbar_cf_report.vision.hr320_runtime import load_hr320
    from install_weight_bundle import digest
    torch.set_num_threads(threads)
    runtime = weights / 'runtime'
    reader = runtime / 'reader50'
    sys.path[:0] = [str(runtime), str(reader)]
    case = (dataset / 'Development388/cases' / case_id).resolve()
    if not re.fullmatch(r'LMR_\d+', case_id) or not case.is_relative_to(dataset.resolve()):
        raise ValueError('Invalid case identifier')
    if not (case / 'dicom').is_dir():
        raise FileNotFoundError(case / 'dicom')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory to preserve prior runs')
    output.mkdir(parents=True, exist_ok=True)
    report = {'case_id': case_id, 'cohort': 'Development388', 'device': 'cpu',
              'full_pipeline_pass': False, 'annotations_or_reports_read': False,
              'stages': {}, 'blockers': []}

    def save():
        (output / 'single_case_result.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def stage(name, fn):
        print('Running ' + name, flush=True)
        start = time.monotonic()
        try:
            with (output / (name + '.log')).open('w', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = fn()
            report['stages'][name] = {'status': 'PASS', 'seconds': round(time.monotonic()-start, 2),
                                     'details': result}
            save()
            print(name + ': PASS', flush=True)
            return True
        except Exception as exc:
            report['stages'][name] = {'status': 'FAIL', 'error': str(exc),
                                     'traceback': traceback.format_exc()}
            save()
            print(name + ': FAIL: ' + str(exc), flush=True)
            return False
        finally:
            gc.collect()

    def integrity():
        receipt = json.loads((runtime / 'INSTALLATION.json').read_text(encoding='utf-8'))
        for r in receipt['installed_files']:
            path = (weights / r['destination']).resolve()
            if not path.is_relative_to(weights) or digest(path) != r['sha256']:
                raise ValueError('Runtime source integrity failed: ' + r['destination'])
        return {'files_verified': len(receipt['installed_files'])}

    if not stage('00_runtime_integrity', integrity):
        return report
    cfg = json.loads((reader / 'configs/reader50_frozen.json').read_text(encoding='utf-8'))
    cfg.update(reader_root=str(case.parent), output_root=str(output), expected_cases=1,
               device='cpu', num_workers=0, project_root=str(runtime))
    cfg_path = output / 'runtime_config.json'
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    serial = int(case_id.split('_')[-1])

    def select():
        m = module(reader / 'scripts/01_prepare_case_contract.py', '_single_center')
        original_headers = m.read_headers
        m.reader_cases = lambda root: [(serial, case)]
        # The frozen selector scans only images here, never case.json/annotations.
        m.read_headers = lambda path: original_headers(path / 'dicom')
        invoke(m, ['--config', cfg_path])
        return json.loads((output / '01_case_contract/SUMMARY.json').read_text())

    if not stage('01_auto_center', select):
        return report
    visual = reader / 'vendor/visual'

    def prior():
        m = module(visual / 'prepare_stage1_19i49_node_mapping.py', '_single_prior')
        def strict(model, checkpoint, device):
            loaded, info = load_hr320(checkpoint, device)
            model.load_state_dict(loaded.state_dict(), strict=True)
            return {'checkpoint_args': info['model_args'], 'strict': True, 'sha256': info['sha256']}
        m.load_prior_checkpoint = strict
        # Avoid scanning unrelated patients when resolving one known DICOM.
        original_index = m.build_selected_basename_index
        m.build_selected_basename_index = lambda root, rows: original_index(case / 'dicom', rows)
        invoke(m, ['--mri_root', case.parent, '--manual_selection_json', output/'01_case_contract/reader50_center_selection.json',
                   '--prior_ckpt_path', weights/'coordinate_prior/hr320_v7.pt', '--out_dir', output/'02_node_mapping',
                   '--source_dataset', 'reader50', '--expected_cases', 1, '--source_serials_file',
                   output/'01_case_contract/source_serials.txt', '--serial_offset', 0, '--save_previews', 1])
        # The vendor exporter infers run identity from the checkpoint directory
        # name. Our verified installer relocates it, so restore the frozen run
        # identifier after strict hash validation; keep the actual local path.
        mapping_path = output/'02_node_mapping/manual_center_node_mapping.csv'
        frame = pd.read_csv(mapping_path)
        frame['prior_checkpoint_run'] = cfg['visual']['prior_run']
        frame.to_csv(mapping_path, index=False, encoding='utf-8-sig')
        return json.loads((output/'02_node_mapping/summary.json').read_text(encoding='utf-8'))

    if not stage('02_HR320', prior):
        return report

    def case_root(center, root, *unused):
        if not Path(center).resolve().is_relative_to(case / 'dicom'):
            raise ValueError('Center escaped selected case DICOM directory')
        return case / 'dicom'

    for kind, folder in [('stenosis', '03_stenosis_mapping'), ('nerve', '04_nerve_mapping')]:
        def mapping(kind=kind, folder=folder):
            m = module(visual/f'prepare_stage1_19i49_{kind}_mapping.py', '_single_'+kind)
            m.reader50_case_root = case_root
            args = ['--mri_root', case.parent, '--node_mapping_csv', output/'02_node_mapping/manual_center_node_mapping.csv',
                    '--out_dir', output/folder, '--expected_cases', 1]
            if kind == 'stenosis':
                args += ['--coverage_gate_mode', 'warn']
            invoke(m, args)
            return {'executed': True, 'output': folder}
        if not stage(folder, mapping):
            return report

    def experts():
        m = module(visual/'prepare_stage1_19i49_external_evidence_cache.py', '_single_experts')
        for dsmod in (m._disc_ds_mod, m._nerve_ds_mod, m._sten_ds_mod):
            if hasattr(dsmod, 'find_case_root'):
                dsmod.find_case_root = case_root
        # The legacy Dataset API needs label columns; these generated placeholders
        # never enter the forward-call whitelist and are stripped from saved tensors.
        cc = output/'01_case_contract'
        node = output/'02_node_mapping/manual_center_node_mapping.csv'
        cache = output/'05_expert_cache'
        cache.mkdir(exist_ok=True)
        results = {}
        specs = [('disc', m.Stage116aDiscExpert, 'best_common.pt'),
                 ('stenosis', m.Stage116aStenosisExpert, 'best_fused_common.pt'),
                 ('nerve', m.Stage116aNerveExpert, 'best_fused_common.pt')]
        for kind, cls, filename in specs:
            model, ck, provenance = m.load_expert(cls, weights/f'visual_experts/{kind}/{filename}',
                                                ['model', 'model_state_dict'], torch.device('cpu'))
            a = ck.get('args', {})
            if kind == 'disc':
                ds = m.Stage116aDiscExpertDataset(cc/'manifest.csv', node, cc/'dummy_labels.csv', case.parent,
                          int(a.get('visual_hw',384)), str(cache/'disc'), 1)
                result = m.infer_disc(model, ds, torch.device('cpu'), 1, 0)
            elif kind == 'stenosis':
                ds = m.Stage116aStenosisExpertDataset(cc/'manifest.csv', node,
                    output/'03_stenosis_mapping/dualview_stenosis_context_mapping.csv', case.parent,
                    visual_hw=int(a.get('visual_hw',384)), roi_crop_size=int(a.get('roi_crop_size',192)),
                    sagittal_cache_dir=str(cache/'sten_sag'), axial_cache_dir=str(cache/'sten_ax'),
                    training=False, augment_intensity=False, posterior_shift_mm=float(a.get('posterior_shift_mm',21)),
                    context_width_mm=float(a.get('context_width_mm',46)), context_height_mm=float(a.get('context_height_mm',36)))
                result = m.infer_stenosis(model, ds, torch.device('cpu'), 1, 0)
            else:
                ds = m.Stage116aNerveExpertDataset(cc/'manifest.csv', output/'04_nerve_mapping/dualview_nerve_mapping.csv',
                    cc/'dummy_labels.csv', case.parent, int(a.get('visual_hw',384)), int(a.get('roi_crop_size',192)),
                    str(cache/'nerve'), 1, training=False, augment_intensity=False,
                    **{k:float(a.get(k,v)) for k,v in dict(lateral_shift_mm=17,posterior_shift_mm=21,
                        primary_width_mm=24,primary_height_mm=22,context_width_mm=38,context_height_mm=32).items()})
                result = m.infer_nerve(model, ds, torch.device('cpu'), 1, 0)
            if set(result) != {serial}:
                raise ValueError('Expert case identity mismatch')
            clean = {k:v for k,v in result[serial].items() if k not in ('label','label_valid')}
            if any(not torch.isfinite(v).all() for v in clean.values() if torch.is_tensor(v)):
                raise ValueError('Non-finite expert output')
            torch.save(clean, output/f'05_{kind}_features.pt')
            results[kind] = {'features': list(clean['features'].shape), 'valid': clean['valid'].tolist(),
                             'checkpoint_sha256': provenance['sha256']}
            del model, ck, ds, result
            gc.collect()
        return results

    if not stage('05_visual_experts', experts):
        return report

    required = {
        'visual_geometry_training_metadata': weights/'runtime_assets/lordosis_geometry_constants.pt',
        'coordinate_normalizer': weights/'runtime_assets/coordinate_normalizer.pt',
        'direct_prefix': weights/'direct/epoch_2/direct_prefix.pt',
        'direct_lora': weights/'direct/epoch_2/lora/adapter_model.safetensors',
        'direct_lora_config': weights/'direct/epoch_2/lora/adapter_config.json',
    }
    for name, path in required.items():
        if not path.is_file():
            report['blockers'].append({'asset': name, 'expected_local_path': str(path)})
    report['stages']['06_visual_evidence_and_normalization'] = {
        'status': 'BLOCKED' if report['blockers'] else 'NOT_EXECUTED',
        'reason': ('Missing frozen runtime assets; no substitutes used.' if report['blockers'] else
                   'Assets are installed; downstream execution is not implemented in this diagnostic entry point. No inference claim.')}
    for name in ('07_planner', '08_direct_draft', '09_qwen_candidates', '10_embeddings_selector', '11_final_report'):
        report['stages'][name] = {'status': 'NOT_EXECUTED', 'reason': 'Upstream factual runtime incomplete'}
    save()
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/local_assets.json'))
    p.add_argument('--case-id', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text(encoding='utf-8'))
    weights = Path(c['weights_root'])
    if not weights.is_absolute():
        weights = ROOT/weights
    r = run(Path(c['dataset_root']).resolve(), a.case_id, weights.resolve(), Path(a.output).resolve(), a.threads)
    print(json.dumps({'full_pipeline_pass': r['full_pipeline_pass'], 'stages':
          {k:v['status'] for k,v in r['stages'].items()}, 'blockers': r['blockers']}, indent=2))
    raise SystemExit(0 if r['full_pipeline_pass'] else 2)
