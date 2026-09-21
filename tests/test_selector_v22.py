from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch
from lumbar_cf_report.selection import FrozenSelector
from lumbar_cf_report.selection.reranker import FactNet, TextListwiseNet, choose
from lumbar_cf_report.selection.text_embedder import encode_pool, cache_to_case_embeddings, input_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_exact_source_hashes_and_imports():
    manifest = json.loads((ROOT/'docs/SELECTOR_V22_MANIFEST.json').read_text())
    for entry in manifest['files']:
        if '/r32s4v22/' in entry['source']:
            assert hashlib.sha256((ROOT/entry['destination']).read_bytes()).hexdigest() == entry['source_sha256']
            importlib.import_module('lumbar_cf_report.selection.'+Path(entry['destination']).stem)
    for script in (ROOT/'scripts/selector').glob('*.py'):
        runpy.run_path(str(script))


def test_original_selector_selftests():
    checks = runpy.run_path(str(ROOT/'tests/selector_v22_selftest.py'))
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for name in ('planner_smoke', 'feature_smoke', 'encoder_smoke', 'reranker_smoke'):
            checks[name]()
    finally:
        torch.set_num_threads(old_threads)


def make_case():
    return {'serial':'test-only', 'core_binary':[0]*16, 'slot_valid':[1]*16,
            'planner_probs':[.2]*16, 'planner_thresholds':[.5]*16,
            'direct_draft':'Direct test text', 'linguistic_scaffold':'<CORE> test scaffold',
            'candidates':[{'tag':'greedy','text':'Candidate A','parser_vec':[0]*16},
                          {'tag':'sample0','text':'Candidate B','parser_vec':[0]*16}]}


def test_actual_neural_selection_firewall_and_cache_alignment(tmp_path):
    helpers = runpy.run_path(str(ROOT/'tests/selector_v22_selftest.py'))
    case = make_case()
    cfg={'text_encoder':{'batch_size':2,'prefix_max_tokens':80,'candidate_max_tokens':40}}
    cache=encode_pool(helpers['_TinyTok'](),helpers['_TinyModel'](),[case],cfg,'cpu')
    torch.manual_seed(3)
    fact={'kind':'fact','state_dict':FactNet(185).state_dict(),'in_dim':185,'mean':np.zeros(185),'std':np.ones(185)}
    text={'kind':'text_listwise','state_dict':TextListwiseNet(16,266).state_dict(),'emb_dim':16,'tab_dim':266,
          'tab_mean':np.zeros(266),'tab_std':np.ones(266)}
    fp=tmp_path/'fact.pt';tp=tmp_path/'text.pt'
    torch.save(fact,fp);torch.save(text,tp)
    profile=json.loads((ROOT/'configs/selector/profile.json').read_text())
    selector=FrozenSelector.from_paths([fp],[tp],profile)
    original=selector.select([case],cache)
    changed=deepcopy(case)
    changed.update(reference_raw='SECRET',slot_labels=[1]*16,GT='SECRET')
    for c in changed['candidates']:c.update(rouge=999,bleu=999,target_fact_f1=999,reference='SECRET')
    assert selector.select([changed],cache)==original
    assert input_fingerprint(changed,changed['candidates'][0])==input_fingerprint(case,case['candidates'][0])
    assert original[0]['selected_index'] in original[0]['factual_safe_candidate_indices']
    changed['candidates'][0]['text']='Different candidate'
    with pytest.raises(RuntimeError,match='cache mismatch'):
        selector.select([changed],cache)
    with pytest.raises(ValueError,match='checkpoints'):
        FrozenSelector([fact],[dict(text,kind='language')],profile)


def test_language_cannot_select_outside_fact_safe_set():
    profile={'alpha_fact':1.,'fact_margin':.01,'text_model_weight':1.}
    index,_,_,eligible=choose(np.array([[.95,.95],[.1,.1]]),[.9,.1],[0.,100.],[0.,1.],profile)
    assert index==0 and eligible==[0]


def test_portable_config_preserves_frozen_scientific_parameters():
    frozen=json.loads((ROOT/'configs/selector/frozen_source.json').read_text())
    portable=json.loads((ROOT/'configs/selector/default.json').read_text())
    for key in ('reranker','generation','scaffold_channel','precision_patch','candidate_protocol','text_encoder','selection_gate','holdout_gate'):
        assert portable[key]==frozen[key]
    assert portable['parser']['expected_sha256']==hashlib.sha256((ROOT/'src/lumbar_cf_report/evaluation/report_metrics.py').read_bytes()).hexdigest()
