#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json,sys,traceback,torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import safe_load,dumpj
from lumbar_cf_report.planner.p3_replay import (
    normalize_pair_split,match_teacher_to_pairs,pair_primary_mask,
    scan_historical_artifacts,candidate_deltas_from_artifacts,
    load_explicit_user_delta,snapshot_sources,import_and_inventory,_schema,
    load_frozen_e0,align_e0_to_serials
)
from lumbar_cf_report.planner.planner_legacy import load_frozen_planner
from lumbar_cf_report.planner.training import resolve_device

def _tolist(v):
    if torch.is_tensor(v):return v.detach().cpu().reshape(-1).tolist()
    if isinstance(v,np.ndarray):return v.reshape(-1).tolist()
    return list(v)

def _s(v):
    if torch.is_tensor(v) and v.numel()==1:v=v.item()
    if isinstance(v,np.generic):v=v.item()
    return str(v)

def disease_logits(main):return torch.stack([main[:,1:6],main[:,6:11],main[:,11:16]],dim=2)

def _teacher_rows(teacher):
    if 'rows' in teacher:return int(teacher['rows'])
    return len(_tolist(teacher['recipient_serial']))

def validate_delta(model,d,e0_split,teacher,delta,device,batch=96,row_ids=None):
    """Validate full level-wise P3 transition against the frozen historical Planner teacher."""
    M=_teacher_rows(teacher);dd=torch.as_tensor(delta).detach().cpu().float()
    if dd.shape!=(M,3,128):raise ValueError(f'exact P3 candidate must be [M,3,128], got {tuple(dd.shape)}')
    rs=[_s(v) for v in _tolist(teacher['recipient_serial'])];smap={str(s):i for i,s in enumerate(d['serials'])}
    levels=torch.as_tensor(_tolist(teacher['level'])).long();target=torch.as_tensor(teacher['cf_base_logits_selected']).float()
    if target.shape[0]!=M:raise ValueError(f'teacher target rows {target.shape[0]} != {M}')
    if target.ndim!=2 or target.shape[1]!=3:raise ValueError(f'cf_base_logits_selected expected [M,3], got {tuple(target.shape)}')
    ids=np.arange(M,dtype=int) if row_ids is None else np.asarray(row_ids,dtype=int)
    pred=[];tgt=[];model.eval()
    with torch.inference_mode():
        for st in range(0,len(ids),batch):
            di=torch.tensor(ids[st:st+batch],dtype=torch.long)
            miss=[(int(r),rs[int(r)]) for r in di.tolist() if rs[int(r)] not in smap]
            if miss:raise RuntimeError(f'teacher recipients missing from Development388: {miss[:10]}')
            ii=torch.tensor([smap[rs[int(r)]] for r in di.tolist()],dtype=torch.long)
            ev=e0_split[ii].clone();li=levels[di]
            for j,r in enumerate(di.tolist()):ev[j,int(li[j])]+=dd[r]
            o=model(d['global_source'][ii].to(device),ev.to(device),d['coordinate_state'][ii].to(device),d['task_quality'][ii].to(device),d['task_valid'][ii].to(device))
            dl=disease_logits(o['main_logits'].detach().cpu());pred.append(torch.stack([dl[j,int(li[j])] for j in range(len(di))]));tgt.append(target[di])
    pred=torch.cat(pred);tgt=torch.cat(tgt);err=(pred-tgt).abs()
    return {'rows':len(ids),'mae':float(err.mean()),'p99_abs':float(torch.quantile(err.flatten(),.99)),'max_abs':float(err.max()),'rmse':float(torch.sqrt((err*err).mean()))}

def inspect_csv(path):
    p=Path(path)
    if not p.exists():return {'exists':False,'path':str(p)}
    try:
        with open(p,newline='') as f:
            r=csv.reader(f);header=next(r,[]);sample=[]
            for _ in range(3):
                try:sample.append(next(r))
                except StopIteration:break
        return {'exists':True,'path':str(p),'columns':header,'sample_rows':sample}
    except Exception as e:return {'exists':True,'path':str(p),'read_error':repr(e)}

def _fixed_contract(teacher,cfg):
    fixed=cfg['p3_fixed_contract']
    got={
      'teacher_p3_lambda':float(teacher.get('p3_lambda',float('nan'))),
      'teacher_alpha':float(teacher.get('specificity_alpha',float('nan'))),
      'teacher_beta':float(teacher.get('beta',float('nan'))),
      'teacher_seeds':[int(x) for x in teacher.get('ensemble_seeds',[])],
      'expected':fixed,
    }
    ok=(abs(got['teacher_p3_lambda']-fixed['lambda'])<1e-9 and abs(got['teacher_alpha']-fixed['alpha'])<1e-9 and abs(got['teacher_beta']-fixed['beta'])<1e-9 and got['teacher_seeds']==fixed['ensemble_seeds'])
    return got,ok

def _candidate_summary(cands,prov):
    return [{'name':k,'shape':list(v.shape),'provenance':prov.get(k,{})} for k,v in cands.items()]

def main_impl(cfg,out):
    device=resolve_device(cfg['device']);data=safe_load(out/'01_stage23b_data.pt');d=data['development']
    teacher=safe_load(cfg['teacher_cache']);M=_teacher_rows(teacher);pair_bank=safe_load(cfg['pair_bank'])
    dumpj({'pair_bank_schema':_schema(pair_bank)},out/'02_PAIR_BANK_SCHEMA.json')
    dumpj({'teacher_schema':_schema(teacher)},out/'02_TEACHER_CACHE_SCHEMA.json')
    pairs=normalize_pair_split(pair_bank,'train');pm=pair_primary_mask(pairs)
    labels=pairs.get('direction_label');label_counts={}
    if labels is not None:
        for x in labels:label_counts[str(x)]=label_counts.get(str(x),0)+1
    dumpj({
      'schema_kind':pairs.get('schema_kind'),'source_split_key':pairs.get('source_split_key'),'rows':pairs['rows'],
      'primary_rows':int(pm.sum()),'direction_label_counts':label_counts,
      'contract':'pair direction is categorical orientation metadata; it is never parsed as a latent vector',
      'recipient_type':type(pairs['recipient']).__name__,'level_type':type(pairs['level']).__name__,'task_type':type(pairs['task']).__name__,
    },out/'02_PAIR_BANK_NORMALIZED.json')
    fixed,contract_ok=_fixed_contract(teacher,cfg)
    if cfg['replay']['require_exact_fixed_contract'] and not contract_ok:
        raise RuntimeError(f'P3 fixed contract mismatch: {fixed}')
    matched,amb=match_teacher_to_pairs(teacher,pairs)
    if amb or int((matched>=0).sum())!=M:
        raise RuntimeError(f'teacher-to-pair mapping incomplete: matched={int((matched>=0).sum())}/{M}, examples={amb[:8]}')

    # Use the exact frozen historical factual anchor from Stage1.21.
    # IMPORTANT: do NOT reconstruct Stage1.20 from Stage2.3 global_source.
    E0,e0_serials=load_frozen_e0(cfg['frozen_old_factual_evidence'])
    e0_dev=align_e0_to_serials(E0,e0_serials,d['serials'])
    dumpj({'E0_shape':list(E0.shape),'E0_serials':len(e0_serials),'development_aligned_shape':list(e0_dev.shape),
           'contract':'historical P3 factual anchor is frozen E0; Stage1.20 forward reconstruction is intentionally not called'},
          out/'02_FROZEN_E0_ALIGNMENT.json')
    legacy,_=load_frozen_planner(cfg['legacy_planner_checkpoint'],device)

    # Artifact-first recovery: Stage2.1/2.2 already produced the exact production P3 tensors.
    artifacts,inventory=scan_historical_artifacts(cfg,M)
    dumpj({'rows_expected':M,'artifacts':inventory},out/'02_HISTORICAL_P3_ARTIFACT_INVENTORY.json')
    candidate_map,provenance=candidate_deltas_from_artifacts(artifacts,teacher,E0,e0_serials,M,cfg)

    # Optional explicit user asset. It must still pass exact historical teacher replay.
    user=load_explicit_user_delta(cfg)
    if user:
        for c in user.get('candidates',[]):
            dd=torch.as_tensor(c['delta']).float()
            if dd.shape==(M,3,128):
                name=f"user::{Path(user['file']).name}::{c['path']}";candidate_map[name]=dd;provenance[name]={'semantic':'explicit_user_full_level_delta','alignment':'USER_SUPPLIED_ORDER_REQUIRES_TEACHER_VALIDATION'}
    dumpj({'candidate_count':len(candidate_map),'candidates':_candidate_summary(candidate_map,provenance)},out/'02_P3_CANDIDATE_INVENTORY.json')

    # Cheap screening on a deterministic spread of teacher rows, then exact full replay on the best few.
    if M>192:screen_ids=np.unique(np.linspace(0,M-1,192,dtype=int))
    else:screen_ids=np.arange(M,dtype=int)
    screen=[]
    for name,delta in candidate_map.items():
        try:
            m=validate_delta(legacy,d,e0_dev,teacher,delta,device,row_ids=screen_ids)
            screen.append((m['mae'],m['p99_abs'],name,m))
        except Exception as e:
            screen.append((float('inf'),float('inf'),name,{'error':repr(e)}))
    screen.sort(key=lambda x:(x[0],x[1]))
    full_results={};best=None
    for _,_,name,sm in screen[:min(12,len(screen))]:
        if 'error' in sm:
            full_results[name]={'screen':sm,'teacher_exact_pass':False};continue
        try:
            m=validate_delta(legacy,d,e0_dev,teacher,candidate_map[name],device)
            exact=(m['mae']<=cfg['replay']['max_teacher_logit_mae'] and m['p99_abs']<=cfg['replay']['max_teacher_logit_p99_abs'] and 'TARGET_ONLY_DIAGNOSTIC' not in name)
            full_results[name]={'screen':sm,'full':m,'teacher_exact_pass':exact,'provenance':provenance.get(name,{})}
            if exact and (best is None or m['mae']<best[0]):best=(m['mae'],name,candidate_map[name],m)
        except Exception as e:full_results[name]={'screen':sm,'error':repr(e),'teacher_exact_pass':False,'provenance':provenance.get(name,{})}
    # Preserve screen diagnostics for candidates not fully evaluated.
    for _,_,name,sm in screen:
        if name not in full_results:full_results[name]={'screen':sm,'full_validation':'NOT_RUN_TOP12_ONLY','teacher_exact_pass':False,'provenance':provenance.get(name,{})}

    exact_names=[n for n,r in full_results.items() if r.get('teacher_exact_pass') is True]
    # Preserve every exact formula, not only the numerically best one. This matters for Internal100:
    # the Development cache may store cf_evidence_level while an Internal cache stores p3_direction.
    rep={
      'status':'RESOLVED_EXACT_P3_DEVELOPMENT' if best else 'UNRESOLVED_EXACT_P3',
      'fixed_contract':fixed,'pair_rows':pairs['rows'],'primary_pair_rows':int(pm.sum()),'teacher_rows':M,
      'matched_rows':int((matched>=0).sum()),'ambiguous_or_missing_matches':amb[:100],
      'production_transition_contract':'full level-wise [rows,3,128] P3 response anchored at frozen Stage1.21 E0; pair_bank.direction is a string label only',
      'candidate_validation':full_results,'exact_replay_candidates':exact_names,'p3_metrics_csv_inventory':inspect_csv(cfg['p3_metrics_csv']),
      'note':'Only a [rows,3,128] candidate that reproduces historical Stage2.1 teacher CF logits inside locked tolerances is accepted.'
    }
    if best:
        _,name,delta,m=best
        meta={k:teacher[k] for k in ('recipient_dev_pos','recipient_serial','level','task','sign','local_scale','selected_disease_indices','target_position_within_level') if k in teacher}
        exact_specs=[{'name':n,'suffix':n.split('::')[-1],'provenance':provenance.get(n,{})} for n in exact_names]
        payload={
          'status':'EXACT_TEACHER_VALIDATED','source_formula':name,'source_provenance':provenance.get(name,{}),
          'exact_replay_candidates':exact_specs,
          'delta':delta.cpu(),'meta':meta,'p3_lambda':float(teacher['p3_lambda']),'specificity_alpha':float(teacher['specificity_alpha']),
          'beta':float(teacher['beta']),'ensemble_seeds':list(teacher['ensemble_seeds']),'teacher_validation':m,
          'transition_shape_contract':'[rows,3,128] full level-wise response'
        }
        torch.save(payload,out/'02_exact_p3_development.pt');rep['selected_formula']=name;rep['selected_provenance']=provenance.get(name,{});rep['selected_validation']=m
    else:
        # Full forensics are written on failure; no traceback-only dead end.
        rep['source_snapshot']=snapshot_sources(cfg,out)
        _,backend=import_and_inventory(cfg.get('stage22c_p3_backend',''));rep['stage22c_backend_inventory']=backend
    dumpj(rep,out/'02_P3_RECOVERY_DECISION.json');print(json.dumps(rep,indent=2,default=str))
    return bool(best)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out)
    try:ok=main_impl(cfg,out)
    except Exception as e:
        rep={'status':'UNRESOLVED_EXACT_P3','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'note':'Recovery failed closed before CF training. See schema/artifact/source inventories in this directory.'}
        try:
            rep['source_snapshot']=snapshot_sources(cfg,out);_,backend=import_and_inventory(cfg.get('stage22c_p3_backend',''));rep['stage22c_backend_inventory']=backend
        except Exception as e2:rep['forensics_error']=repr(e2)
        dumpj(rep,out/'02_P3_RECOVERY_DECISION.json');print(json.dumps(rep,indent=2));ok=False
    if not ok and cfg.get('replay',{}).get('fail_closed',True):raise SystemExit(42)
if __name__=='__main__':main()
