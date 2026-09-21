#!/usr/bin/env python3
from pathlib import Path
import argparse,copy,json,sys,torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import safe_load,dumpj,set_seed
from lumbar_cf_report.planner.model import Stage23UnifiedClinicalPlanner
from lumbar_cf_report.planner.training import ensure_model_device,evaluate
from lumbar_cf_report.planner.train_cf import train_mode,make_drift_scales,context_rollback_state,state_group_distance,normalized_pair_drift
from lumbar_cf_report.planner.cf import row_to_patient_indices,evaluate_pairs


def _build(cfg,ck):
    m=Stage23UnifiedClinicalPlanner(cfg,ck['anchor_weight'],ck['anchor_bias']);m.load_state_dict(ck['state_dict'],strict=True);return m


def _build_from_state(cfg,ck,state):
    m=Stage23UnifiedClinicalPlanner(cfg,ck['anchor_weight'],ck['anchor_bias']);m.load_state_dict(state,strict=True);return m


def _save_candidate(out,name,cfg,ck,data,p3,state,epoch,factual,pair,extra):
    torch.save({
      'version':cfg['version'],'phase':'B_CF_SUPERVISION','mode':name,'state_dict':state,
      'anchor_weight':ck['anchor_weight'],'anchor_bias':ck['anchor_bias'],'thresholds':ck['thresholds'],
      'phaseA_candidate_path':data['phaseA_candidate_path'],'phaseA_candidate_sha256':data['phaseA_candidate_sha256'],
      'p3_source_formula':p3['source_formula'],'best_epoch':epoch,'best_factual':factual,'best_pair':pair,
      'best_normalized_pair':extra.get('normalized_pair'),'best_bounded_pair':None,**extra
    },out/f'04_{name}_candidate.pt')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out);set_seed(cfg['seed']+2300)
    data=safe_load(out/'01_stage23b_data.pt');d=data['development'];p3=safe_load(out/'02_exact_p3_development.pt');split=safe_load(out/'03_cf_split.pt');baseline=json.load(open(out/'03_PHASEA_CF_BASELINE.json'))
    tr=split['train_rows'].numpy();va=split['val_rows'].numpy();phaseA_scales=baseline['drift_normalization_scales'];delta=p3['delta'];meta=p3['meta'];ck=safe_load(data['phaseA_candidate_path']);th=ck['thresholds'].numpy();pi=row_to_patient_indices(meta,d['serials']);val_pat=np.asarray(sorted(set(pi[va].tolist())),dtype=int)
    base=_build(cfg,ck);dev=ensure_model_device(base,cfg['device']);phaseA_fact,_=evaluate(base,d,val_pat,dev,th)

    # 1) Target-only remains exactly the successful v1.5 training stage.
    model=_build(cfg,ck);phaseA=copy.deepcopy(model)
    best,hist=train_mode(model,phaseA,d,delta,meta,tr,va,dev,cfg['phaseB'],th,phaseA_fact,'target_only',cfg['seed'],drift_scales=phaseA_scales,profile={'name':'target_only','disease_weight':0.0,'structure_weight':0.0},target_reference_pair=None)
    model.load_state_dict(best['state']);model.cpu();phaseA.cpu()
    _save_candidate(out,'target_only',cfg,ck,data,p3,best['state'],best['epoch'],best['factual'],best['pair'],{
        'drift_normalization_scales':phaseA_scales,'profile':{'name':'target_only'},'initialization':'PHASEA_FULL_PASS','normalized_pair':best.get('normalized_pair')})

    # 2) Freeze Target-only reference and its Development CF-train drift scales.
    target_ref=_build_from_state(cfg,ck,best['state']);ensure_model_device(target_ref,dev);target_ref.eval()
    target_train_pair=evaluate_pairs(target_ref,d,delta,meta,tr,dev,cfg['phaseB']['target_margin_logit'])
    floor=float(cfg['phaseB']['normalized_drift']['normalization_floor']);lord=float(cfg['phaseB']['normalized_drift']['lordosis_weight'])
    refine_scales=make_drift_scales(target_train_pair,floor,lord,source='TARGET_ONLY_DEVELOPMENT_CF_TRAIN_ROWS_ONLY')

    # 3) v1.6 deterministic excess-drift margin refinement.
    #    Shared context is gently rolled back toward Phase-A; disease residual heads stay bit-exact Target-only.
    rcfg=cfg['phaseB']['excess_drift_margin_refinement'];groups=tuple(rcfg['rollback_groups']);preserve=tuple(rcfg['preserve_exact_target_only_groups']);profiles={}
    phasea_state={k:v.detach().cpu().clone() for k,v in ck['state_dict'].items()};target_state={k:v.detach().cpu().clone() for k,v in best['state'].items()}
    base_distance=state_group_distance(target_state,phasea_state,groups)
    for alpha in rcfg['alpha_grid']:
        alpha=float(alpha)
        if alpha<=0:continue
        if alpha>float(rcfg['max_alpha'])+1e-12:raise RuntimeError(f'alpha {alpha} exceeds max_alpha {rcfg["max_alpha"]}')
        state,contract=context_rollback_state(phasea_state,target_state,alpha,groups,preserve)
        # Enforce exact preservation of Target-only disease residual heads.
        for k in state:
            if any(g in k for g in preserve) and not torch.equal(state[k],target_state[k]):
                raise RuntimeError(f'v1.6 preserve-exact contract violated for {k}')
        cand=_build_from_state(cfg,ck,state);ensure_model_device(cand,dev);cand.eval()
        factual,_=evaluate(cand,d,val_pat,dev,th);pair=evaluate_pairs(cand,d,delta,meta,va,dev,cfg['phaseB']['target_margin_logit']);norm=normalized_pair_drift(pair,refine_scales)
        name=f'excess_margin_rollback_a{int(round(alpha*1000)):03d}'
        rollback_distance=state_group_distance(state,phasea_state,groups);target_distance=state_group_distance(state,target_state,groups)
        extra={
          'drift_normalization_scales':refine_scales,'profile':{'name':name,'alpha':alpha},'initialization':'TARGET_ONLY_BEST_PARAMETER_ROLLBACK',
          'rollback_contract':contract,'rollback_groups':list(groups),'preserve_exact_target_only_groups':list(preserve),
          'target_only_reference_pair':best['pair'],'target_only_reference_factual':best['factual'],'target_only_best_epoch':best['epoch'],
          'normalized_pair':norm,'target_to_phaseA_group_distance':base_distance,'candidate_to_phaseA_group_distance':rollback_distance,'candidate_to_target_group_distance':target_distance,
          'optimization':'DETERMINISTIC_NO_GRADIENT'
        }
        _save_candidate(out,name,cfg,ck,data,p3,state,0,factual,pair,extra)
        profiles[name]={'candidate_name':name,'alpha':alpha,'profile':extra['profile'],'best_epoch':0,'best_factual':factual,'best_pair':pair,'best_normalized_pair':norm,'rollback_contract':contract,'target_to_phaseA_group_distance':base_distance,'candidate_to_phaseA_group_distance':rollback_distance,'candidate_to_target_group_distance':target_distance}
        cand.cpu()
    target_ref.cpu()

    rep={
      'status':'PASS','phaseA_drift_scales':phaseA_scales,'target_only_train_pair':target_train_pair,'excess_drift_margin_scales':refine_scales,
      'modes':{'target_only':{'best_epoch':best['epoch'],'score':best['score'],'best_factual':best['factual'],'best_pair':best['pair'],'best_normalized_pair':best.get('normalized_pair'),'history':hist},'excess_drift_margin_profiles':profiles},
      'training_contract':{
        'target_only_initialization':'PHASEA_FULL_PASS','refinement_initialization':'TARGET_ONLY_BEST','refinement_optimization':'DETERMINISTIC_PARAMETER_INTERPOLATION_NO_GRADIENT',
        'rollback_groups':list(groups),'preserve_exact_target_only_groups':list(preserve),'alpha_grid':rcfg['alpha_grid'],
        'rationale':'Target-only target response is preserved in residual heads; only shared context is partially rolled back toward Phase-A to remove excess protected drift.',
        'internal100_used_for_selection':False,
      },
      'profile_selection_scope':'Predeclared alpha grid; Development388 CF-validation only; Internal100 never used.'
    }
    dumpj(rep,out/'04_CF_MODE_TRAINING.json')
    short={'target_only':best['pair'],'excess_drift_margin_profiles':{k:{'alpha':v['alpha'],'pair':v['best_pair'],'factual_auprc':{t:v['best_factual'][t]['auprc'] for t in ('disc','stenosis','nerve')}} for k,v in profiles.items()}}
    print(json.dumps(short,indent=2))
if __name__=='__main__':main()
