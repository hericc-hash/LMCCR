from __future__ import annotations
from pathlib import Path
import json, torch
from .io import load_pt, sha256
from ..planner.model import Stage23UnifiedClinicalPlanner

EXPECTED_VERSION = 'LMCCR-Stage2.3B-CF-Supervision-v1.7'
EXPECTED_MODEL = {
    'task_dim':128,'global_dim':256,'coordinate_dim':47,
    'slot_hidden':96,'global_hidden':64,'coordinate_hidden':32,'embedding_dim':16,
    'dropout':0.1,'max_context_logit_residual':0.75,'max_cross_slot_gate':0.15,
    'initial_cross_slot_gate':0.02,'plan_state_dim':128,'plan_token_dim':4096,
    'produce_plan_tokens':False,
}
MODEL_SOURCE_SHA256 = '937619ece8b8b4406f671783b7a9f543de7d8562f975a41af0bd297b9c716e5a'


def load_runtime(stage23_dir, candidate_name='05_stage2_3B_selected_candidate.pt', data_name='01_stage23b_data.pt', config_name='config_used.json', saved_probs_name='07_internal_factual_probs.pt'):
    root=Path(stage23_dir)
    paths={
        'candidate':root/candidate_name,
        'data':root/data_name,
        'config':root/config_name,
        'saved_probs':root/saved_probs_name,
    }
    missing=[str(p) for p in paths.values() if not p.exists()]
    if missing: raise FileNotFoundError('Stage2.3-v1.7 frozen runtime assets missing:\n'+'\n'.join(missing))
    ck=load_pt(paths['candidate']); data=load_pt(paths['data']); cfg=json.loads(paths['config'].read_text(encoding='utf-8')); saved=load_pt(paths['saved_probs'])
    if str(ck.get('version')) != EXPECTED_VERSION:
        raise RuntimeError(f'Unexpected candidate version: {ck.get("version")}')
    if str(cfg.get('version')) != EXPECTED_VERSION:
        raise RuntimeError(f'Unexpected config_used version: {cfg.get("version")}')
    m=cfg.get('model',{})
    mismatches={k:{'expected':v,'got':m.get(k)} for k,v in EXPECTED_MODEL.items() if m.get(k)!=v}
    if mismatches: raise RuntimeError(f'Stage2.3-v1.7 model config drift: {mismatches}')
    if 'internal' not in data: raise RuntimeError('stage23b data missing internal split')
    if not isinstance(saved,dict) or 'probs' not in saved: raise RuntimeError('07_internal_factual_probs.pt missing probs')
    return cfg,ck,data,saved,paths


def build_exact_model(cfg,ck,device='cpu'):
    if cfg.get('version') != EXPECTED_VERSION or ck.get('version') != EXPECTED_VERSION:
        raise ValueError('Expected frozen Stage2.3-B v1.7 config and checkpoint')
    mismatches = {k: cfg.get('model', {}).get(k) for k, v in EXPECTED_MODEL.items()
                  if cfg.get('model', {}).get(k) != v}
    if mismatches:
        raise ValueError(f'Stage2.3-v1.7 model config drift: {mismatches}')
    model=Stage23UnifiedClinicalPlanner(cfg=cfg,anchor_w=ck['anchor_weight'],anchor_b=ck['anchor_bias'])
    model.load_state_dict(ck['state_dict'],strict=True)
    model.to(device).eval()
    return model


def replay_internal_disease_probs_cpu(cfg,ck,data,batch_size=32):
    d=data['internal']; model=build_exact_model(cfg,ck,'cpu'); outs=[]
    with torch.no_grad():
        for st in range(0,len(d['serials']),batch_size):
            sl=slice(st,min(st+batch_size,len(d['serials'])))
            o=model(d['global_source'][sl],d['raw_E'][sl],d['coordinate_state'][sl],d['task_quality'][sl],d['task_valid'][sl])
            outs.append(o['disease_probabilities'].detach().cpu())
    return torch.cat(outs,0)


def validate_saved_internal_replay(cfg,ck,data,saved,max_abs_tol=2e-6):
    got=replay_internal_disease_probs_cpu(cfg,ck,data)
    ref=torch.as_tensor(saved['probs']).float().cpu()
    if tuple(got.shape)!=tuple(ref.shape):
        raise RuntimeError(f'Saved Internal factual probability shape mismatch {tuple(got.shape)} vs {tuple(ref.shape)}')
    diff=(got-ref).abs()
    rep={'shape':list(got.shape),'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'tolerance':float(max_abs_tol),'pass':bool(float(diff.max())<=max_abs_tol)}
    if not rep['pass']: raise RuntimeError(f'Stage2.3-v1.7 exact replay mismatch: {rep}')
    return rep
