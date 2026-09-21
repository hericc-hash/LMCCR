from __future__ import annotations
import importlib, json, sys, traceback
from pathlib import Path
import torch

from .io import safe_load, sha256_file


def normalize_full(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x=torch.as_tensor(x).float()
    if x.ndim!=3 or tuple(x.shape[1:])!=(3,128):
        raise ValueError(f'expected [M,3,128], got {tuple(x.shape)}')
    f=x.reshape(len(x),-1)
    return (f/f.norm(dim=-1,keepdim=True).clamp_min(float(eps))).reshape_as(x)


def ensemble_delta(seed_directions, local_scale, beta: float, rule: str='normalized_mean_direction', eps: float=1e-8):
    """Frozen 3-seed P3 ensemble -> finite-step delta.

    Historical Stage2.1/2.2 stores one ensemble P3 direction after the three
    Stage-A seeds.  The canonical rule is the normalized mean direction, then
    the exact Stage1.21 finite step beta*local_scale*u.
    """
    ds=torch.stack([normalize_full(d,eps) for d in seed_directions],0)  # [S,M,3,128]
    scale=torch.as_tensor(local_scale).float().reshape(-1)
    if ds.shape[1]!=len(scale):raise ValueError('seed direction/local_scale row mismatch')
    if rule=='normalized_mean_direction':
        u=normalize_full(ds.mean(0),eps)
        delta=float(beta)*scale[:,None,None]*u
    elif rule=='mean_finite_step':
        delta=(float(beta)*scale[None,:,None,None]*ds).mean(0)
        u=normalize_full(ds.mean(0),eps)
    else:raise ValueError(f'unknown ensemble rule: {rule}')
    return delta,u


def _import_original_modules(source_root: Path):
    # The semantically merged P3 package supplies the exact replay operations.
    backend=importlib.import_module('lumbar_cf_report.p3.backend')
    ops=importlib.import_module('lumbar_cf_report.p3.ops')
    evaluator=importlib.import_module('lumbar_cf_report.p3.evaluator')
    return backend,ops,evaluator


def _resolve_original_config(cfg):
    explicit=cfg.get('p3_original_config') or cfg.get('replay',{}).get('p3_original_config')
    if explicit:
        p=Path(explicit)
        if p.is_file():return p
    root=Path(cfg['p3_original_source_root'])
    preferred=root/'configs'/'stage1_21_P3_config.json'
    if preferred.is_file():return preferred
    candidates=[]
    for p in sorted(root.rglob('*.json')):
        try:
            x=json.loads(p.read_text(encoding='utf-8'))
        except Exception:continue
        if not isinstance(x,dict):continue
        if all(k in x for k in ('paths','model','specificity_anchor','functional_search')):
            paths=x.get('paths',{})
            if 'pair_bank' in paths and 'frozen_old_factual_evidence' in paths:
                candidates.append(p)
    if len(candidates)==1:return candidates[0]
    raise RuntimeError(f'could not uniquely resolve original P3 config; candidates={[str(x) for x in candidates[:20]]}')


def replay_original_internal(cfg, device='cuda'):
    """Replay frozen Stage1.21-P3 on its original Internal1179 pairs.

    This is a fallback only when no frozen Internal production tensor exists.
    It uses the original source package, original X/E/C feature bank, original
    protected Jacobian, original Stage-A seed checkpoints and the frozen global
    alpha/lambda/beta.  No Stage2.3 global_source is ever passed to Stage1.20.
    """
    p3cfg_path=_resolve_original_config(cfg)
    orig=json.loads(p3cfg_path.read_text(encoding='utf-8'))
    fixed=cfg['p3_fixed_contract']
    # Bind all critical assets/settings before running anything.
    if Path(orig['paths']['pair_bank']).resolve()!=Path(cfg['pair_bank']).resolve():
        raise RuntimeError('original P3 config pair_bank does not match Stage2.3-B frozen pair_bank')
    if Path(orig['paths']['frozen_old_factual_evidence']).resolve()!=Path(cfg['frozen_old_factual_evidence']).resolve():
        raise RuntimeError('original P3 config frozen E0 path mismatch')
    alpha=float(orig['specificity_anchor']['alpha'])
    beta=float(orig['functional_search']['selection_beta'])
    if abs(alpha-float(fixed['alpha']))>1e-12 or abs(beta-float(fixed['beta']))>1e-12:
        raise RuntimeError(f'original P3 alpha/beta mismatch: alpha={alpha} beta={beta} fixed={fixed}')
    lam=float(fixed['lambda'])
    eps=float(orig['functional_search'].get('eps',1e-8))

    backend,ops,evaluator=_import_original_modules(Path(cfg['p3_original_source_root']))
    pb,E0=ops.load_pair_assets(orig)
    rows=pb['internal']
    if len(rows)!=1179:
        raise RuntimeError(f'original internal P3 rows expected 1179, got {len(rows)}')
    if any(str(r.get('direction','')).lower() not in ('majority_to_minority','primary','forward','') for r in rows):
        raise RuntimeError('original internal P3 split unexpectedly contains reverse/non-primary rows')
    J=ops.load_protected_jacobian_cache(orig)['J']
    if tuple(J.shape)!=(len(rows),2,384):
        raise RuntimeError(f'original Internal protected J shape mismatch: {tuple(J.shape)}')
    dev=torch.device(device if not (str(device).startswith('cuda') and not torch.cuda.is_available()) else 'cpu')
    bank=ops.move_bank_to_device(backend.load_bank(orig['paths']['feature_bank']),dev)
    base,_=backend.load_model(orig,dev);base=ops.freeze_model_parameters(base)
    checkpoints=evaluator._load_stageA_checkpoints(orig)
    got_seeds=[int(x['seed']) for x in checkpoints]
    if got_seeds!=sorted(int(x) for x in fixed['ensemble_seeds']):
        raise RuntimeError(f'Stage-A seed mismatch: got={got_seeds} expected={fixed["ensemble_seeds"]}')

    # Exact factual anchor check using the original base model and X/E/C interface.
    with torch.no_grad():
        *_,cr,cd,qr,qd,fa,fb,factual,_=ops.encode_pair(base,bank,rows,dev,transport=True)
    factual_e0=torch.stack([E0[int(r['recipient_index']),int(r['level'])] for r in rows],0).to(dev)
    factual_maxdiff=float((factual-factual_e0).abs().max())
    if factual_maxdiff>1e-5:
        raise RuntimeError(f'original P3 factual/E0 mismatch: {factual_maxdiff}')

    seed_dirs=[];seed_info=[]
    for info in checkpoints:
        model,_=backend.load_model(orig,dev,checkpoint_override=str(info['path']));model=ops.freeze_model_parameters(model)
        with torch.no_grad():
            *_,factualA,endpoint=ops.encode_pair(model,bank,rows,dev,transport=True)
        fd=float((factualA-factual).abs().max())
        if fd>1e-5:raise RuntimeError(f'Stage-A factual changed seed={info["seed"]}: {fd}')
        stage_dir=endpoint-factualA
        spec_dir=ops.ordinary_project_direction(stage_dir,J.to(dev),float(fixed['alpha']),eps)
        p3_dir=ops.blend_direction(spec_dir,stage_dir,lam,eps)
        seed_dirs.append(p3_dir.detach().cpu())
        seed_info.append({'seed':int(info['seed']),'checkpoint':str(info['path']),'factual_maxdiff':fd})
        del model
        if dev.type=='cuda':torch.cuda.empty_cache()

    scale=torch.tensor([float(r['local_scale']) for r in rows],dtype=torch.float32)
    rule=str(cfg.get('replay',{}).get('internal_source_ensemble_rule','normalized_mean_direction'))
    delta,u=ensemble_delta(seed_dirs,scale,float(fixed['beta']),rule,eps)
    sign=torch.tensor([1.0 if int(r['minority_state'])==1 else -1.0 for r in rows],dtype=torch.float32)
    meta={
      'recipient_serial':[r['recipient_serial'] for r in rows],
      'level':torch.tensor([int(r['level']) for r in rows],dtype=torch.long),
      'task':torch.tensor([int(r['task']) for r in rows],dtype=torch.long),
      'sign':sign,
      'local_scale':scale,
      'beta':float(fixed['beta']),'p3_lambda':lam,'specificity_alpha':float(fixed['alpha']),
      'ensemble_seeds':[int(x) for x in fixed['ensemble_seeds']],
    }
    provenance={
      'status':'RESOLVED_BY_ORIGINAL_STAGE1_21_SOURCE_REPLAY',
      'original_config':str(p3cfg_path),
      'original_config_sha256':sha256_file(p3cfg_path),
      'source_root':str(Path(cfg['p3_original_source_root']).resolve()),
      'rows':len(rows),'J_shape':list(J.shape),'E0_shape':list(E0.shape),
      'factual_maxdiff_vs_E0':factual_maxdiff,'seed_info':seed_info,
      'ensemble_rule':rule,'alpha':float(fixed['alpha']),'lambda':lam,'beta':float(fixed['beta']),
      'direction_shape':list(u.shape),'delta_shape':list(delta.shape),
      'contract':'original X/E/C interface; E*=E0+beta*local_scale*u; no sign multiplier; full [3,128] level response'
    }
    return delta.cpu(),meta,provenance
