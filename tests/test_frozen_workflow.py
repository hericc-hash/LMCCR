"""CPU regressions for merged contracts; no patient data or model downloads."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import ast

import numpy as np
import pytest
import torch

from lumbar_cf_report.planner import Stage23UnifiedClinicalPlanner
from lumbar_cf_report.bridge.stage23_exact import EXPECTED_MODEL, EXPECTED_VERSION, build_exact_model, validate_saved_internal_replay
from lumbar_cf_report.bridge.inference import predict_clinical16
from lumbar_cf_report.evaluation import report_metrics as parser
from lumbar_cf_report.generation.eval_bridge import normalize, parse
from lumbar_cf_report.generation.scaffold import build_scaffold, build_similarity_scaffold
from lumbar_cf_report.generation.precision_patch import precision_finalize
from lumbar_cf_report.generation.prompting import messages
from lumbar_cf_report.generation.canonical import SLOT_NAMES
from lumbar_cf_report.p3.ops import blend_direction, adaptive_select, freeze_model_parameters
from lumbar_cf_report.selection import inference_features, require_final_selector

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen_model():
    torch.manual_seed(42)
    cfg = {'version': EXPECTED_VERSION, 'model': deepcopy(EXPECTED_MODEL)}
    model = Stage23UnifiedClinicalPlanner(cfg).eval()
    ck = {'version': EXPECTED_VERSION, 'anchor_weight': model.anchor_w.detach().clone(),
          'anchor_bias': model.anchor_b.detach().clone(), 'state_dict': model.state_dict()}
    return cfg, ck, model


def evidence(n=2):
    return {'global_source': torch.randn(n, 256), 'raw_E': torch.randn(n, 5, 3, 128),
            'coordinate_state': torch.randn(n, 5, 47), 'task_quality': torch.ones(n, 5, 3),
            'task_valid': torch.ones(n, 5, 3)}


def test_exact_planner_roundtrip_and_raw_e_firewall(frozen_model):
    cfg, ck, _ = frozen_model
    model = build_exact_model(cfg, ck)
    batch = evidence()
    batch['task_valid'][0, 2, 1] = 0
    first = predict_clinical16(model, batch, [.5]*16, [1, 0])
    batch.update(Ehat=torch.full_like(batch['raw_E'], float('nan')), labels='NEVER_READ', reference='NEVER_READ')
    second = predict_clinical16(model, batch, [.5]*16, [1, 0])
    assert torch.equal(first['planner_probs'], second['planner_probs'])
    assert first['core_binary'].shape == (2, 16)
    assert first['slot_valid'][0, 8] == 0
    assert first['slot_valid'][1, 0] == 0
    del batch['raw_E']
    with pytest.raises(ValueError, match='Raw-E'):
        predict_clinical16(model, batch, [.5]*16, [1, 0])


def test_exact_planner_rejects_semantic_drift_and_wrong_checkpoint(frozen_model):
    cfg, ck, _ = frozen_model
    bad = deepcopy(cfg)
    bad['model']['max_cross_slot_gate'] = .2
    with pytest.raises(ValueError, match='config drift'):
        build_exact_model(bad, ck)
    bad_ck = deepcopy(ck)
    del bad_ck['state_dict']['cross_gate_raw']
    with pytest.raises(RuntimeError):
        build_exact_model(cfg, bad_ck)


def test_saved_replay_pass_and_fail_closed(frozen_model):
    cfg, ck, model = frozen_model
    batch = evidence()
    batch['serials'] = [1, 2]
    with torch.inference_mode():
        pred = model(batch['global_source'], batch['raw_E'], batch['coordinate_state'], batch['task_quality'], batch['task_valid'])['disease_probabilities']
    assert validate_saved_internal_replay(cfg, ck, {'internal': batch}, {'probs': pred})['pass']
    with pytest.raises(RuntimeError, match='replay mismatch'):
        validate_saved_internal_replay(cfg, ck, {'internal': batch}, {'probs': pred + .1})


def test_canonical_sources_are_unchanged():
    manifest = json.loads((ROOT/'docs/CURATED_MERGE_MANIFEST.json').read_text(encoding='utf-8'))
    for suffix in ['planner/model.py', 'evaluation/report_metrics.py']:
        rec = next(x for x in manifest['files'] if x['destination'].endswith(suffix))
        assert hashlib.sha256((ROOT/rec['destination']).read_bytes()).hexdigest() == rec['source_sha256']


def test_clinical16_order_and_none_lordosis():
    result = parser.parse_focused_semantics('L4/5椎间盘膨出。L3/4水平椎管狭窄。')
    vec = normalize(result)
    assert vec.shape == (16,)
    assert result['lordosis'] is None and vec[0] == 0
    assert vec[4] == 1 and vec[8] == 1
    # Mapping order must not control Clinical16 order.
    assert np.array_equal(vec, normalize(dict(reversed(list(result.items())))))
    metric = parser.compute_report_slot_metrics([result], vec[None], np.ones((1, 16)))
    assert metric['micro_f1'] == 1


def test_scaffold_is_parser_zero_and_preserves_safe_language():
    text = '所见：L4/5椎间盘膨出，T2WI信号减低，相应硬膜囊前缘受压。\n结论：L4/5椎间盘膨出。'
    scaffold, meta = build_scaffold(text, parser, parse, normalize, {})
    assert meta['parser_zero'] and not normalize(parse(parser, scaffold)).any()
    assert 'T2WI信号减低' in scaffold and '硬膜囊前缘受压' in scaffold
    assert '<CORE>' in scaffold
    prompt = str(messages([0]*16, [1]*16, scaffold))
    assert 'CORE-REDACTED LINGUISTIC SCAFFOLD' in prompt
    assert 'L4/5椎间盘膨出' not in prompt


def test_precision_patch_preserves_protected_level_without_fallback():
    def fake_parse(mod, text):
        # Exact production logic is separately tested above. This isolates the
        # patch algorithm's handling of compact multi-level statements.
        vec = [0]*16
        for i, lv in enumerate(['L1/2', 'L2/3', 'L3/4', 'L4/5', 'L5/S1']):
            if lv in text and '椎管狭窄' in text:
                vec[6+i] = 1
        return vec
    core = [0]*16
    core[8] = 1
    lex = {key: '安全' for key in SLOT_NAMES}
    lex['STENOSIS_L3/4'] = 'L3/4水平椎管狭窄'
    raw = '所见：L3/4、L4/5水平椎管狭窄，T2WI信号减低。\n结论：L3/4、L4/5水平椎管狭窄。'
    text, meta = precision_finalize(raw, core, [1]*16, lex, None, fake_parse, normalize, {})
    assert fake_parse(None, text)[8:10] == [1, 0]
    assert 'T2WI信号减低' in text and meta['fallback'] is False


def test_target_similarity_projection_does_not_relax_source_safety():
    def rare(mod, text):
        return [0, int('RARE_CORE_PHRASE' in text)] + [0]*14
    lex = {key: 'UNUSED' for key in SLOT_NAMES}
    lex['DISC_L1/2'] = 'RARE_CORE_PHRASE'
    text, meta = build_similarity_scaffold('所见：RARE_CORE_PHRASE，T2WI信号减低。', lex, None, rare, normalize, {})
    assert meta['parser_zero'] and 'RARE_CORE_PHRASE' not in text


def test_v32_joint_language_gate_and_factual_negative_control():
    from lumbar_cf_report.generation.data import scaffold_data_gate
    summary = {'scaffold_parser_zero_rate': 1., 'scaffold_nonempty_fraction': 1.,
               'scaffold_hard_fallback_fraction': 0., 'mean_scaffold_payload_chars': 90.,
               'mean_scaffold_safe_payload_chars': 43., 'mean_preserved_safe_fraction': .51}
    # Exercise supplied source defaults, not an invented frozen experiment config.
    assert scaffold_data_gate(summary, {})['pass']
    assert not scaffold_data_gate(dict(summary, scaffold_parser_zero_rate=.99), {})['pass']
    assert not scaffold_data_gate(dict(summary, mean_scaffold_safe_payload_chars=8.), {})['pass']


def test_v17_nerve_recovery_leaves_other_parameters_exact():
    import runpy
    helper = runpy.run_path(str(ROOT/'scripts/planner/05b_nerve_head_recovery.py'))['_head_rollback_state']
    context = {'residual_heads.2.weight': torch.ones(2), 'residual_heads.0.weight': torch.ones(2),
               'global_proj.weight': torch.ones(2)}
    phase_a = {key: torch.zeros_like(value) for key, value in context.items()}
    result, changed = helper(context, phase_a, .1, ['residual_heads.2.'])
    assert changed == ['residual_heads.2.weight']
    assert torch.allclose(result[changed[0]], torch.full((2,), .9))
    for key in ('residual_heads.0.weight', 'global_proj.weight'):
        assert torch.equal(result[key], context[key])


def test_p3_finite_step_selection_and_frozen_parameters():
    spec = torch.zeros(3, 3, 128); spec[:, 0, 0] = 1
    stage = torch.zeros_like(spec); stage[:, 0, 1] = 1
    assert torch.equal(blend_direction(spec, stage, 0, 1e-8), spec)
    assert torch.equal(blend_direction(spec, stage, 1, 1e-8), stage)
    with pytest.raises(ValueError):
        blend_direction(spec, stage, 1.1, 1e-8)
    dirs = torch.stack([spec, stage, stage])
    mds = torch.tensor([[.01, 0, -.01], [.032, .018, .005], [.04, .02, -.005]])
    drift = torch.tensor([[.002, .001, .001], [.01, .006, .004], [.02, .015, .02]])
    _, diag = adaptive_select(dirs, mds, drift, [0, .5, 1], mds[-1], .9, 1e-8)
    assert diag['selected_lambda'].tolist() == [1, .5, .5]
    model = freeze_model_parameters(torch.nn.Linear(2, 2))
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def test_p3_source_has_no_training_or_autograd():
    for path in (ROOT/'src/lumbar_cf_report/p3').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'backward', 'step', 'zero_grad', 'train', 'grad'}


def test_shared_selector_features_ignore_labels_and_references():
    case = {'core_binary': [0]*16, 'slot_valid': [1]*16, 'planner_probs': [.2]*16,
            'planner_thresholds': [.5]*16, 'direct_draft': '正常形态。',
            'linguistic_scaffold': '所见：<CORE>。', 'adaptive_meta': {'score': 1},
            'candidates': [{'tag': 'greedy', 'text': '正常形态。', 'parser_vec': [0]*16},
                           {'tag': 'sample0', 'text': '所见：形态尚可。', 'parser_vec': [0]*16}]}
    original, names = inference_features(case)
    altered = deepcopy(case)
    altered.update(reference_raw='SECRET', slot_labels=[1]*16, gt='SECRET')
    for cand in altered['candidates']:
        cand.update(rouge=999, bleu=999, target_fact_f1=999, reference='SECRET')
    changed, changed_names = inference_features(altered)
    assert original.shape == (2, 266)
    assert names == changed_names and np.array_equal(original, changed)
    assert not any(any(k in n for k in ['reference', 'rouge', 'bleu', 'gt_', 'logprob']) for n in names)
    assert require_final_selector().__name__ == 'FrozenSelector'


@pytest.mark.parametrize('module', ['p3.pipeline', 'planner.training', 'planner.train_cf', 'planner.original_p3_replay',
                                   'bridge.stage23_exact', 'generation.generate', 'generation.train', 'generation.data', 'selection'])
def test_integrated_module_imports(module):
    importlib.import_module('lumbar_cf_report.' + module)
