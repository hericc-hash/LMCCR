"""Audit installed real checkpoints without equating loading with E2E success.

All paths in the config are relative to the repository, or explicit absolute
local paths. Results contain local asset metadata and stay under outputs/.
"""
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from install_weight_bundle import digest


def asset_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def strict_load(model, state):
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return {'status': 'PASS', 'state_tensors': len(state),
            'parameters': sum(p.numel() for p in model.parameters()),
            'scope': 'strict state loading only; no patient accuracy or replay claim'}


def run(config):
    import torch
    from lumbar_cf_report.vision.hr320_runtime import load_hr320
    from lumbar_cf_report.vision.experts.disc import DiscEvidenceExpert
    from lumbar_cf_report.vision.experts.stenosis import StenosisEvidenceExpert
    from lumbar_cf_report.vision.experts.nerve import NerveEvidenceExpert
    from lumbar_cf_report.vision.model import LumbarVisualEvidenceEncoder
    from lumbar_cf_report.apreb.checkpoints import load_ar_checkpoint
    from lumbar_cf_report.ccmrt.model import CCMRTModel
    from lumbar_cf_report.bridge.stage23_exact import EXPECTED_MODEL, EXPECTED_VERSION, build_exact_model
    from lumbar_cf_report.clinical.checkpoints import load_planner_checkpoint
    from lumbar_cf_report.selection.reranker import FactNet, TextListwiseNet

    torch.set_num_threads(4)
    weights = asset_path(config['weights_root'])
    receipt = json.loads(asset_path(config['installation_receipt']).read_text(encoding='utf-8'))
    checks, errors, blockers = {}, [], []
    for record in receipt['files']:
        p = weights / record['destination']
        if not p.is_file() or digest(p) != record['sha256']:
            errors.append('Installed file integrity failed: ' + record['destination'])
    supplemental_receipt = weights / 'runtime/INSTALLATION.json'
    if supplemental_receipt.is_file():
        supplement = json.loads(supplemental_receipt.read_text(encoding='utf-8'))
        for record in supplement['installed_files']:
            p = (weights / record['destination']).resolve()
            if not p.is_relative_to(weights.resolve()) or not p.is_file() or digest(p) != record['sha256']:
                errors.append('Runtime supplement integrity failed: ' + record['destination'])
    if errors:
        return {'status': 'FAIL', 'errors': errors, 'full_pipeline_pass': False}

    def check(name, fn):
        print('Checking ' + name, flush=True)
        try:
            checks[name] = fn()
        except Exception as exc:
            checks[name] = {'status': 'FAIL', 'error': str(exc)}
            errors.append(name + ': ' + str(exc))
        gc.collect()

    def load(relative):
        return torch.load(weights / relative, map_location='cpu', weights_only=False)

    def dataset_inventory():
        root = asset_path(config['dataset_root'])
        with (root / 'metadata/case_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        counts = {}
        seen = set()
        for row in rows:
            cohort, case = row['cohort'], row['case_id']
            if cohort not in ('Development388', 'Internal100', 'Independent49') or Path(case).name != case:
                raise ValueError('Unexpected cohort or case identifier')
            if case in seen or not (root / cohort / 'cases' / case).is_dir():
                raise ValueError('Duplicate or missing case directory')
            seen.add(case)
            counts[cohort] = counts.get(cohort, 0) + 1
        if counts != {'Development388': 388, 'Internal100': 100, 'Independent49': 49}:
            raise ValueError('Dataset cohort counts differ from the expected contract')
        return {'status': 'PASS', 'cohort_counts': counts,
                'scope': 'metadata and case directory inventory; no labels/reports used as model input'}

    check('dataset_inventory', dataset_inventory)

    def prior():
        _, info = load_hr320(weights / 'coordinate_prior/hr320_v7.pt')
        return {'status': 'PASS', **info}

    check('HR320', prior)
    for name, cls, field, filename in [
        ('disc', DiscEvidenceExpert, 'model', 'best_common.pt'),
        ('stenosis', StenosisEvidenceExpert, 'model_state_dict', 'best_fused_common.pt'),
        ('nerve', NerveEvidenceExpert, 'model_state_dict', 'best_fused_common.pt'),
    ]:
        def expert(name=name, cls=cls, field=field, filename=filename):
            ck = load(f'visual_experts/{name}/{filename}')
            params = inspect.signature(cls).parameters
            args = {k: v for k, v in ck['args'].items() if k in params}
            info = strict_load(cls(**args), ck[field])
            info.update(version=ck.get('version'), constructor_args_from_checkpoint=args,
                        constructor_defaults=[k for k in params if k not in args])
            return info
        check(name, expert)

    def visual():
        ck = load('visual_evidence/lumbar_visual_evidence.pt')
        return strict_load(LumbarVisualEvidenceEncoder(**ck['model_config']), ck['model_state_dict'])
    check('visual_evidence', visual)

    def apreb():
        model, ck = load_ar_checkpoint(weights / 'apreb/apreb.pt')
        return strict_load(model, ck['state_dict'])
    check('APREB', apreb)

    def ccmrt():
        ck = load('ccmrt/ccmrt.pt')
        cfg = dict(ck['config']['model'])
        cfg['a_dim'] = cfg.pop('anatomy_dim')
        cfg['r_dim'] = cfg.pop('residual_dim')
        if 'coordinate_latent_dim' in cfg:
            cfg['coord_latent_dim'] = cfg.pop('coordinate_latent_dim')
        return strict_load(CCMRTModel(x_dim=256, c_dim=47, task_dim=128, **cfg), ck['model_state'])
    check('CCMRT', ccmrt)

    def planner():
        ck = load('clinical_planner/05_stage2_3B_selected_candidate.pt')
        # Architecture is the repository's frozen contract. Never fabricate an
        # original config_used.json or claim saved probability replay.
        config_path = weights / 'clinical_planner/config_used.json'
        cfg = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {
            'version': EXPECTED_VERSION, 'model': EXPECTED_MODEL}
        model = build_exact_model(cfg, ck)
        info = strict_load(model, ck['state_dict'])
        info['config_source'] = ('installed original config_used.json' if config_path.is_file()
                                 else 'repository EXPECTED_MODEL; original runtime config not supplied')
        info['threshold_shape'] = list(torch.as_tensor(ck['thresholds']).shape)
        frozen_runtime = weights / 'runtime/reader50/configs/reader50_frozen.json'
        if frozen_runtime.is_file():
            frozen = json.loads(frozen_runtime.read_text(encoding='utf-8'))['stage23']
            if not torch.equal(torch.as_tensor(ck['thresholds']), torch.tensor(frozen['task_thresholds'])):
                raise ValueError('Planner threshold mismatch against frozen runtime')
            info['thresholds_16'] = [float(frozen['lordosis_threshold'])] + [float(v) for v in ck['thresholds'] for _ in range(5)]
        return info
    check('Planner_v1_7', planner)

    def auxiliary():
        model, ck = load_planner_checkpoint(weights / 'direct_auxiliary_planner/best_stage1_18c_explicit_clinical_mediator.pt')
        return strict_load(model, ck['state_dict'])
    check('direct_auxiliary_planner', auxiliary)

    for kind, key in [('fact', 'fact_models'), ('text_listwise', 'text_models')]:
        for filename in config[key]:
            path = asset_path(filename)
            if not path.is_file():
                label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
                blockers.append('Missing ensemble member: ' + str(label))
                continue
            def selector(path=path, kind=kind):
                ck = torch.load(path, map_location='cpu', weights_only=False)
                if ck['kind'] != kind:
                    raise ValueError('Selector kind mismatch')
                model = FactNet(ck['in_dim'], ck.get('hidden', 160)) if kind == 'fact' else TextListwiseNet(
                    ck['emb_dim'], ck['tab_dim'], ck['emb_proj'], ck['tab_proj'], ck['hidden'])
                info = strict_load(model, ck['state_dict'])
                info['seed'] = ck['seed']
                return info
            check(path.stem, selector)

    def language_assets():
        from safetensors import safe_open
        from transformers import AutoTokenizer
        from peft import LoraConfig
        base = asset_path(config['base_model'])
        cfg = json.loads((base / 'config.json').read_text(encoding='utf-8'))
        if cfg.get('model_type') != 'qwen3' or cfg.get('hidden_size') != 2560:
            raise ValueError('Expected Qwen3-4B architecture with hidden size 2560')
        index = json.loads((base / 'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
        base_shapes = {}
        for shard in set(index.values()):
            p = (base / shard).resolve()
            if not p.is_relative_to(base.resolve()):
                raise ValueError('Qwen shard path escapes base model')
            with safe_open(p, framework='pt', device='cpu') as f:
                for key in f.keys():
                    if index.get(key) != shard:
                        raise ValueError('Qwen shard/index mismatch')
                    base_shapes[key] = f.get_slice(key).get_shape()
        if set(base_shapes) != set(index):
            raise ValueError('Incomplete Qwen shards')
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        if not tokenizer('所见：腰椎。')['input_ids']:
            raise ValueError('Tokenizer failed')
        counts = {}
        for key in ('language_adapter', 'dual_channel_adapter', 'scaffold_adapter'):
            adapter = asset_path(config[key])
            lora = LoraConfig.from_pretrained(adapter, local_files_only=True)
            if lora.r != 16:
                raise ValueError('Unexpected adapter rank')
            with safe_open(adapter / 'adapter_model.safetensors', framework='pt', device='cpu') as f:
                names = set(f.keys())
                for name in names:
                    base_key = name.removeprefix('base_model.model.').replace('.lora_A.weight', '.weight').replace('.lora_B.weight', '.weight')
                    if base_key not in base_shapes:
                        raise ValueError('Adapter target absent from base: ' + name)
                    shape = f.get_slice(name).get_shape()
                    original = base_shapes[base_key]
                    expected = [lora.r, original[1]] if '.lora_A.' in name else [original[0], lora.r]
                    if shape != expected:
                        raise ValueError('Adapter/base shape mismatch: ' + name)
                    counterpart = name.replace('.lora_A.', '.lora_B.') if '.lora_A.' in name else name.replace('.lora_B.', '.lora_A.')
                    if counterpart == name or counterpart not in names:
                        raise ValueError('Incomplete LoRA A/B pair')
                expected_count = 2 * cfg['num_hidden_layers'] * len(lora.target_modules)
                if len(names) != expected_count:
                    raise ValueError('Incomplete layer/module adapter coverage')
                counts[key] = len(names)
        return {'status': 'PASS', 'base_tensors': len(base_shapes), 'adapter_tensors': counts,
                'scope': 'offline tokenizer and complete safetensor shape checks; generation not executed'}
    check('Qwen3_and_adapters', language_assets)

    profile = json.loads(asset_path(config['selector_profile']).read_text(encoding='utf-8'))
    expected_profile = json.loads((ROOT / 'configs/selector/profile.json').read_text(encoding='utf-8'))
    if any(profile.get(k) != v for k, v in expected_profile.items()):
        errors.append('Selector profile differs from frozen repository profile')
    replay_blockers = []
    for file in ('config_used.json', '01_stage23b_data.pt', '07_internal_factual_probs.pt'):
        if not (asset_path(config['planner_runtime']) / file).is_file():
            target = blockers if file == 'config_used.json' else replay_blockers
            target.append('Missing Planner runtime/replay asset: ' + file)
    for relative, purpose in [
        ('runtime_assets/lordosis_geometry_constants.pt', 'Frozen lordosis template/scales and geometry feature metadata'),
        ('runtime_assets/coordinate_normalizer.pt', 'Frozen coordinate feature names, normalization mean and std'),
        ('direct/epoch_2/direct_prefix.pt', 'Selected Direct prefix checkpoint'),
        ('direct/epoch_2/lora/adapter_model.safetensors', 'Selected Direct LoRA weights'),
        ('direct/epoch_2/lora/adapter_config.json', 'Selected Direct LoRA configuration'),
        ('report_realizer/CANONICAL_LEXICALIZER.json', 'Frozen realizer lexicalizer'),
    ]:
        if not (weights / relative).is_file():
            blockers.append(purpose + ': ' + relative)
    blockers.append('Full factual execution remains unverified: inspect the single-case run; do not replace missing frozen statistics.')
    return {'status': 'FAIL' if errors else 'WEIGHTS_VALIDATED_PIPELINE_INCOMPLETE',
            'full_pipeline_pass': False, 'integrity_verified_files': len(receipt['files']),
            'manifest_verified_payloads': receipt['verified_payloads'], 'checks': checks,
            'errors': errors, 'full_pipeline_blockers': blockers,
            'historical_replay_blockers': replay_blockers,
            'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()},
            'weight_publication': 'UNDECIDED; local-only integration'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local_assets.json')
    parser.add_argument('--require-full-pipeline', action='store_true')
    args = parser.parse_args()
    config = json.loads(asset_path(args.config).read_text(encoding='utf-8'))
    result = run(config)
    output = asset_path(config['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'weight_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result['status'], flush=True)
    for error in result['errors']:
        print('ERROR: ' + error)
    for blocker in result.get('full_pipeline_blockers', []):
        print('BLOCKED: ' + blocker)
    return 1 if result['errors'] else (2 if args.require_full_pipeline and not result['full_pipeline_pass'] else 0)


if __name__ == '__main__':
    raise SystemExit(main())
