#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, shutil, copy
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import safe_load,dumpj,sha256_file
from lumbar_cf_report.planner.model import Stage23UnifiedClinicalPlanner
from lumbar_cf_report.planner.training import ensure_model_device,evaluate
from lumbar_cf_report.planner.cf import row_to_patient_indices,evaluate_pairs
from lumbar_cf_report.planner.train_cf import normalized_pair_drift


def _build(cfg,ck,state,device):
    m=Stage23UnifiedClinicalPlanner(cfg,ck['anchor_weight'],ck['anchor_bias']);m.load_state_dict(state,strict=True);ensure_model_device(m,device);m.eval();return m


def _head_rollback_state(context_state,phasea_state,eta,prefixes):
    out={k:v.detach().cpu().clone() for k,v in context_state.items()}
    changed=[]
    for k in out:
        if any(k.startswith(pref) for pref in prefixes):
            if k not in phasea_state: raise KeyError(f'Phase-A state missing {k}')
            out[k]=(1.0-float(eta))*context_state[k].detach().cpu()+float(eta)*phasea_state[k].detach().cpu()
            changed.append(k)
    if not changed: raise RuntimeError('No Nerve head keys matched configured prefixes')
    # Everything outside Nerve head must remain bit-exact context candidate.
    for k,v in out.items():
        if k not in changed and not torch.equal(v,context_state[k].detach().cpu()):
            raise RuntimeError(f'non-Nerve parameter changed unexpectedly: {k}')
    return out,changed


def _l2_distance(a,b,keys):
    num=0.0
    for k in keys:
        d=(a[k].float()-b[k].float()).reshape(-1);num+=float((d*d).sum())
    return num**0.5


def _check(full,context,to,full_fact,context_fact,phasea_fact,rcfg,scales):
    nf=normalized_pair_drift(full,scales);nt=normalized_pair_drift(to,scales);nc=normalized_pair_drift(context,scales)
    retained=(nt['normalized_combined_drift']-nf['normalized_combined_drift'])/max(nt['normalized_combined_drift'],1e-12)
    rel=float(rcfg['relative_drift_tolerance_vs_context'])
    checks={
      'nerve_auprc_preserved_vs_context':full_fact['nerve']['auprc']>=context_fact['nerve']['auprc']-float(rcfg['max_nerve_auprc_drop_vs_context']),
      'nerve_auc_preserved_vs_context':full_fact['nerve']['auroc']>=context_fact['nerve']['auroc']-float(rcfg['max_nerve_auc_drop_vs_context']),
      'nerve_f1_preserved_vs_context':full_fact['nerve']['f1']>=context_fact['nerve']['f1']-float(rcfg['max_nerve_f1_drop_vs_context']),
      'nerve_auprc_preserved_vs_phaseA':full_fact['nerve']['auprc']>=phasea_fact['nerve']['auprc']-float(rcfg['max_nerve_auprc_drop_vs_phaseA']),
      'nerve_auc_preserved_vs_phaseA':full_fact['nerve']['auroc']>=phasea_fact['nerve']['auroc']-float(rcfg['max_nerve_auc_drop_vs_phaseA']),
      'nerve_f1_preserved_vs_phaseA':full_fact['nerve']['f1']>=phasea_fact['nerve']['f1']-float(rcfg['max_nerve_f1_drop_vs_phaseA']),
      'target_success_preserved_vs_context':full['target_success_rate']>=context['target_success_rate']-float(rcfg['target_success_tolerance_vs_context']),
      'target_margin_preserved_vs_context':full['target_margin_success_rate']>=context['target_margin_success_rate']-float(rcfg['target_margin_success_tolerance_vs_context']),
      'mean_shift_preserved_vs_context':full['mean_signed_target_logit_shift']>=context['mean_signed_target_logit_shift']-float(rcfg['mean_signed_shift_tolerance_vs_context']),
      'disease_drift_not_worse_than_context':full['protected_disease_drift']<=context['protected_disease_drift']*(1.0+rel)+1e-12,
      'structure_drift_not_worse_than_context':full['protected_structure_drift']<=context['protected_structure_drift']*(1.0+rel)+1e-12,
      'retains_v16_drift_reduction_vs_target_only':retained>=float(rcfg['minimum_retained_v16_combined_drift_reduction_vs_target_only']),
    }
    return checks,nf,nc,nt,retained


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );ap.add_argument('--out',required=True);a=ap.parse_args();cfg=json.load(open(a.config));out=Path(a.out);device=cfg['device'];rcfg=cfg['phaseB']['nerve_specific_head_recovery']
    csel=json.load(open(out/'05_CONTEXT_ROLLBACK_SELECTION.json'))
    if not csel.get('selected_mode'):
        rep={'verdict':'STAGE2_3B_V17_BLOCKED_NO_V16_CONTEXT_CANDIDATE','selected_mode':None,'selected_eta':None};dumpj(rep,out/'05_CF_SELECTION_DECISION.json');print(json.dumps(rep,indent=2));return
    data=safe_load(out/'01_stage23b_data.pt');d=data['development'];p3=safe_load(out/'02_exact_p3_development.pt');split=safe_load(out/'03_cf_split.pt');va=split['val_rows'].numpy();meta=p3['meta'];delta=p3['delta'];pi=row_to_patient_indices(meta,d['serials']);val_pat=np.asarray(sorted(set(pi[va].tolist())),dtype=int)
    phasea=safe_load(data['phaseA_candidate_path']);to=safe_load(out/'04_target_only_candidate.pt');context=safe_load(out/'05_context_rollback_candidate.pt');th=context['thresholds'].numpy();training=json.load(open(out/'04_CF_MODE_TRAINING.json'));scales=training['excess_drift_margin_scales']
    pm=_build(cfg,phasea,phasea['state_dict'],device);tm=_build(cfg,to,to['state_dict'],device);cm=_build(cfg,context,context['state_dict'],device)
    phasea_fact,_=evaluate(pm,d,val_pat,device,th);context_fact,_=evaluate(cm,d,val_pat,device,th);to_pair=evaluate_pairs(tm,d,delta,meta,va,device,cfg['phaseB']['target_margin_logit']);context_pair=evaluate_pairs(cm,d,delta,meta,va,device,cfg['phaseB']['target_margin_logit'])
    phasea_state={k:v.detach().cpu() for k,v in phasea['state_dict'].items()};context_state={k:v.detach().cpu() for k,v in context['state_dict'].items()};prefixes=tuple(rcfg['head_key_prefixes'])
    audits={};passing=[]
    for eta in rcfg['eta_grid']:
        eta=float(eta)
        if eta<0 or eta>float(rcfg['max_eta'])+1e-12: raise ValueError(f'eta out of range: {eta}')
        state,keys=_head_rollback_state(context_state,phasea_state,eta,prefixes)
        m=_build(cfg,context,state,device);fact,_=evaluate(m,d,val_pat,device,th);pair=evaluate_pairs(m,d,delta,meta,va,device,cfg['phaseB']['target_margin_logit']);checks,nf,nc,nt,retained=_check(pair,context_pair,to_pair,fact,context_fact,phasea_fact,rcfg,scales);ok=all(checks.values())
        d_ctx=_l2_distance(state,context_state,keys);d_pa=_l2_distance(state,phasea_state,keys);base_pa=_l2_distance(context_state,phasea_state,keys)
        name=f'nerve_head_recovery_e{int(round(eta*1000)):03d}'
        audits[name]={'eta':eta,'pass':ok,'checks':checks,'factual':fact,'pair':pair,'normalized_pair':nf,'retained_v16_combined_drift_reduction_fraction':retained,'nerve_head_keys':keys,'nerve_head_distance_to_context':d_ctx,'nerve_head_distance_to_phaseA':d_pa,'context_head_distance_to_phaseA':base_pa}
        torch.save({'version':cfg['version'],'phase':'B_CF_SUPERVISION_V17','mode':name,'state_dict':state,'anchor_weight':context['anchor_weight'],'anchor_bias':context['anchor_bias'],'thresholds':context['thresholds'],'phaseA_candidate_path':data['phaseA_candidate_path'],'phaseA_candidate_sha256':data['phaseA_candidate_sha256'],'p3_source_formula':p3['source_formula'],'profile':{'name':name,'eta':eta,'context_alpha':csel.get('selected_alpha')},'initialization':'V1_6_CONTEXT_ROLLBACK_CANDIDATE','selected_context_profile':csel.get('selected_profile'),'selected_context_alpha':csel.get('selected_alpha'),'nerve_head_recovery_eta':eta,'nerve_head_keys':keys,'development_factual':fact,'development_pair':pair,'development_checks':checks,'drift_normalization_scales':scales,'optimization':'DETERMINISTIC_NERVE_HEAD_INTERPOLATION_NO_GRADIENT'},out/f'05b_{name}_candidate.pt')
        if ok and (eta>0 or not rcfg.get('require_nonzero_eta_for_recovery_claim',True)):passing.append((eta,name))
        m.cpu()
    pm.cpu();tm.cpu();cm.cpu()
    selected_name=None;selected_eta=None
    if passing:
        # This is a generalization-recovery hypothesis test: choose the largest SAFE nonzero rollback.
        passing.sort(reverse=True);selected_eta,selected_name=passing[0]
        src=out/f'05b_{selected_name}_candidate.pt';dst=out/'05_stage2_3B_selected_candidate.pt';shutil.copy2(src,dst)
        verdict='STAGE2_3B_DEVELOPMENT_NERVE_HEAD_RECOVERY_SAFE_NONZERO_PASS';mode='nerve_specific_factual_head_recovery'
    else:
        # Fail closed to the v1.6 context candidate; no recovery claim.
        shutil.copy2(out/'05_context_rollback_candidate.pt',out/'05_stage2_3B_selected_candidate.pt')
        selected_eta=0.0;selected_name='context_only_eta000';verdict='STAGE2_3B_DEVELOPMENT_NERVE_HEAD_RECOVERY_NO_SAFE_NONZERO';mode='excess_drift_margin_refinement'
    rep={'verdict':verdict,'selected_mode':mode,'selected_profile':selected_name,'selected_alpha':csel.get('selected_alpha'),'selected_eta':selected_eta,'context_selection':csel,'context_factual':context_fact,'context_pair':context_pair,'target_only_pair':to_pair,'phaseA_factual':phasea_fact,'drift_normalization_scales':scales,'head_recovery_profiles':audits,'selection_contract':'First freeze v1.6 Development-selected context rollback alpha. Then, without changing any non-Nerve parameter, choose the largest predeclared nonzero Nerve residual-head rollback eta that passes Development factual, CF-target, and protected-drift hard gates. Internal100 is never used for eta selection.','selection_scope':'Development388 only. Internal100 is frozen hypothesis confirmation; Independent49 remains untouched.','selected_checkpoint_sha256':sha256_file(out/'05_stage2_3B_selected_candidate.pt')}
    dumpj(rep,out/'05_CF_SELECTION_DECISION.json');print(json.dumps({'verdict':verdict,'selected_mode':mode,'selected_alpha':rep['selected_alpha'],'selected_eta':selected_eta,'selected_profile':selected_name},indent=2))
if __name__=='__main__':main()
