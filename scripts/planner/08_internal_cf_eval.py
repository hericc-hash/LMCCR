#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import safe_load,dumpj
from lumbar_cf_report.planner.model import Stage23UnifiedClinicalPlanner
from lumbar_cf_report.planner.training import ensure_model_device
from lumbar_cf_report.planner.cf import evaluate_pairs
from lumbar_cf_report.planner.train_cf import normalized_pair_drift,bounded_pair_drift


def load_model(cfg,ck,device):
    m=Stage23UnifiedClinicalPlanner(cfg,ck['anchor_weight'],ck['anchor_bias']);m.load_state_dict(ck['state_dict'],strict=True);ensure_model_device(m,device);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out);ip=out/'06_exact_p3_internal.pt'
    if not ip.exists():dumpj({'status':'SKIPPED_INTERNAL_P3_UNRESOLVED'},out/'08_INTERNAL_CF_DECISION.json');return
    sel=json.load(open(out/'05_CF_SELECTION_DECISION.json'))
    if not sel.get('selected_mode'):
        dumpj({'status':'SKIPPED_NO_DEVELOPMENT_FULL_CF_CANDIDATE'},out/'08_INTERNAL_CF_DECISION.json');return
    data=safe_load(out/'01_stage23b_data.pt');d=data['internal'];p3=safe_load(ip);delta=p3['delta'];meta=p3['meta'];rows=np.arange(len(delta));device=cfg['device'];pa=safe_load(data['phaseA_candidate_path']);to=safe_load(out/'04_target_only_candidate.pt');full=safe_load(out/'05_stage2_3B_selected_candidate.pt')
    models={'phaseA_no_cf':load_model(cfg,pa,device),'target_only':load_model(cfg,to,device),'nerve_specific_factual_head_recovery':load_model(cfg,full,device)};res={k:evaluate_pairs(m,d,delta,meta,rows,device,cfg['phaseB']['target_margin_logit']) for k,m in models.items()}
    scales=sel['drift_normalization_scales'];cap=cfg['phaseB']['target_first_refinement_legacy_v15'].get('log1p_cap',8.0);norm={k:normalized_pair_drift(v,scales) for k,v in res.items()};bounded={k:bounded_pair_drift(v,scales,cap) for k,v in res.items()};g=cfg['phaseB']['mechanism_gate'];f=res['nerve_specific_factual_head_recovery'];t=res['target_only'];b=res['phaseA_no_cf'];nf=norm['nerve_specific_factual_head_recovery'];nt=norm['target_only'];rel=float(g.get('internal_relative_drift_tolerance',0.05));combined_reduction=(nt['normalized_combined_drift']-nf['normalized_combined_drift'])/max(nt['normalized_combined_drift'],1e-12)
    checks={
      'min_target_success':f['target_success_rate']>=g['min_target_success_rate'],
      'min_margin_success':f['target_margin_success_rate']>=g['min_target_margin_success_rate'],
      'max_disease_drift':f['protected_disease_drift']<=g['max_protected_disease_drift'],
      'max_structure_drift':f['protected_structure_drift']<=g['max_protected_structure_drift'],
      'target_noninferior_to_target_only':f['target_success_rate']>=t['target_success_rate']-g['full_vs_target_only_target_success_tolerance'],
      'margin_noninferior_to_target_only':f['target_margin_success_rate']>=t['target_margin_success_rate']-g.get('full_vs_target_only_target_margin_tolerance',0.02),
      'mean_shift_noninferior_to_target_only':f['mean_signed_target_logit_shift']>=t['mean_signed_target_logit_shift']-g.get('full_vs_target_only_mean_signed_shift_tolerance',0.01),
      'disease_drift_not_worse_than_target_only':f['protected_disease_drift']<=t['protected_disease_drift']*(1.0+rel)+1e-12,
      'structure_drift_not_worse_than_target_only':f['protected_structure_drift']<=t['protected_structure_drift']*(1.0+rel)+1e-12,
      'combined_normalized_drift_not_worse_than_target_only':combined_reduction>=float(g.get('internal_min_combined_normalized_drift_reduction',0.0)),
    }
    ok=all(checks.values())
    rep={'status':'PASS' if ok else 'FAIL','internal_policy':'FROZEN_MECHANISM_EVALUATION_ONLY_NO_SELECTION','selected_profile':sel.get('selected_profile'),'selected_alpha':sel.get('selected_alpha'),'selected_eta':sel.get('selected_eta'),'drift_normalization_scales_from_development_target_only_train':scales,'metrics':res,'normalized_metrics':norm,'bounded_metrics':bounded,'checks':checks,'combined_normalized_drift_reduction_fraction':combined_reduction,'mechanism_effect':{'target_only_minus_phaseA_target_success':t['target_success_rate']-b['target_success_rate'],'refinement_minus_target_only_target_success':f['target_success_rate']-t['target_success_rate'],'refinement_minus_target_only_target_margin_success':f['target_margin_success_rate']-t['target_margin_success_rate'],'refinement_minus_target_only_mean_signed_shift':f['mean_signed_target_logit_shift']-t['mean_signed_target_logit_shift'],'refinement_minus_target_only_protected_disease_drift':f['protected_disease_drift']-t['protected_disease_drift'],'refinement_minus_target_only_protected_structure_drift':f['protected_structure_drift']-t['protected_structure_drift'],'refinement_minus_phaseA_protected_disease_drift':f['protected_disease_drift']-b['protected_disease_drift']}}
    dumpj(rep,out/'08_INTERNAL_CF_DECISION.json');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
