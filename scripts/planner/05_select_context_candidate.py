#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import dumpj,sha256_file
from lumbar_cf_report.planner.train_cf import normalized_pair_drift
from lumbar_cf_report.planner.training import preservation_pass


def _check_candidate(full,full_fact,to,to_fact,base,base_fact,g,rcfg,scales):
    nf=normalized_pair_drift(full,scales);nt=normalized_pair_drift(to,scales)
    reduction=(nt['normalized_combined_drift']-nf['normalized_combined_drift'])/max(nt['normalized_combined_drift'],1e-12)
    checks={
      'factual_preserved_vs_target_only':preservation_pass(full_fact,to_fact,rcfg['development_preservation_vs_target_only']),
      'factual_preserved_vs_phaseA':preservation_pass(full_fact,base_fact,rcfg['development_preservation_vs_phaseA']),
      'full_min_target_success':full['target_success_rate']>=g['min_target_success_rate'],
      'full_min_margin_success':full['target_margin_success_rate']>=g['min_target_margin_success_rate'],
      'full_max_disease_drift':full['protected_disease_drift']<=g['max_protected_disease_drift'],
      'full_max_structure_drift':full['protected_structure_drift']<=g['max_protected_structure_drift'],
      'target_success_noninferior_to_target_only':full['target_success_rate']>=to['target_success_rate']-float(rcfg['target_success_tolerance']),
      'target_margin_noninferior_to_target_only':full['target_margin_success_rate']>=to['target_margin_success_rate']-float(rcfg['target_margin_success_tolerance']),
      'mean_signed_shift_noninferior_to_target_only':full['mean_signed_target_logit_shift']>=to['mean_signed_target_logit_shift']-float(rcfg['mean_signed_shift_tolerance']),
      'disease_drift_not_worse_than_target_only':full['protected_disease_drift']<=to['protected_disease_drift']+1e-12,
      'structure_drift_not_worse_than_target_only':full['protected_structure_drift']<=to['protected_structure_drift']+1e-12,
      'combined_normalized_drift_reduction_margin':reduction>=float(rcfg['minimum_combined_normalized_drift_reduction']),
    }
    return checks,nf,nt,reduction


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out)
    b=json.load(open(out/'03_PHASEA_CF_BASELINE.json'));base=b['phaseA_cf_pair_baseline_validation'];base_fact=b['phaseA_factual_on_cf_validation_patients']
    tr=json.load(open(out/'04_CF_MODE_TRAINING.json'));modes=tr['modes'];to=modes['target_only']['best_pair'];to_fact=modes['target_only']['best_factual'];scales=tr['excess_drift_margin_scales'];profiles=modes['excess_drift_margin_profiles'];g=cfg['phaseB']['mechanism_gate'];rcfg=cfg['phaseB']['excess_drift_margin_refinement']
    audits={};passing=[]
    for name,z in profiles.items():
        full=z['best_pair'];full_fact=z['best_factual'];checks,nf,nt,reduction=_check_candidate(full,full_fact,to,to_fact,base,base_fact,g,rcfg,scales);ok=all(checks.values())
        audits[name]={
          'alpha':z['alpha'],'profile':z['profile'],'best_pair':full,'best_factual':full_fact,'normalized_pair':nf,'target_only_normalized_pair':nt,
          'combined_normalized_drift_reduction_fraction':reduction,'checks':checks,'pass':ok,'candidate_name':z['candidate_name'],
          'rollback_contract':z['rollback_contract'],'target_to_phaseA_group_distance':z['target_to_phaseA_group_distance'],'candidate_to_phaseA_group_distance':z['candidate_to_phaseA_group_distance'],'candidate_to_target_group_distance':z['candidate_to_target_group_distance']
        }
        if ok:passing.append((reduction,float(z['alpha']),name,z))
    selected=None;selected_name=None
    if passing:
        # Primary objective: largest protected-drift reduction after all factual/target hard gates.
        best_reduction=max(x[0] for x in passing);tie=float(rcfg.get('conservative_tie_tolerance',0.002))
        near=[x for x in passing if x[0]>=best_reduction-tie]
        # If reductions are essentially tied, choose the smaller rollback to stay closer to Target-only.
        near.sort(key=lambda x:(x[1],-x[0]))
        reduction,alpha,selected_name,z=near[0];selected=z['candidate_name'];verdict='STAGE2_3B_DEVELOPMENT_EXCESS_DRIFT_MARGIN_COPRESERVATION_PASS'
    elif to['target_success_rate']>=g['min_target_success_rate']:
        verdict='STAGE2_3B_DEVELOPMENT_TARGET_ONLY_PARTIAL_NO_DRIFT_CLAIM'
    else:verdict='STAGE2_3B_DEVELOPMENT_CF_NO_GO'
    rep={
      'verdict':verdict,'selected_mode':'excess_drift_margin_refinement' if selected else None,'selected_profile':selected_name,'selected_alpha':audits[selected_name]['alpha'] if selected_name else None,
      'phaseA_baseline':base,'phaseA_factual':base_fact,'target_only':to,'target_only_factual':to_fact,'drift_normalization_scales':scales,'profiles':audits,
      'mechanism_effect':{'target_only_minus_phaseA_target_success':to['target_success_rate']-base['target_success_rate']},
      'selection_contract':'After factual and CF-target hard gates, select the predeclared Target-only→Phase-A shared-context rollback with maximal normalized protected-drift reduction. Residual heads remain exactly Target-only.',
      'selection_scope':'Development388 CF-validation patients only; Internal100 is not used. Alpha is frozen before Internal100.'
    }
    if selected:
        src=out/f'04_{selected}_candidate.pt';dst=out/'05_context_rollback_candidate.pt';shutil.copy2(src,dst);rep['context_checkpoint_sha256']=sha256_file(dst);rep['context_candidate_name']=selected;rep['context_candidate']=audits[selected_name]
    dumpj(rep,out/'05_CONTEXT_ROLLBACK_SELECTION.json');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
