#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import safe_load,dumpj
from lumbar_cf_report.planner.model import Stage23UnifiedClinicalPlanner
from lumbar_cf_report.planner.training import ensure_model_device,evaluate
from lumbar_cf_report.planner.cf import split_rows,evaluate_pairs
from lumbar_cf_report.planner.train_cf import make_drift_scales,normalized_pair_drift


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out)
    data=safe_load(out/'01_stage23b_data.pt');d=data['development'];p3=safe_load(out/'02_exact_p3_development.pt');delta=p3['delta'];meta=p3['meta'];tr,va,pi=split_rows(meta,d['serials'],cfg['phaseB']['validation_patient_modulus'],cfg['phaseB']['validation_patient_remainder'])
    ck=safe_load(data['phaseA_candidate_path']);m=Stage23UnifiedClinicalPlanner(cfg,ck['anchor_weight'],ck['anchor_bias']);m.load_state_dict(ck['state_dict'],strict=True);dev=ensure_model_device(m,cfg['device']);th=ck['thresholds'].numpy();val_pat=np.asarray(sorted(set(pi[va].tolist())),dtype=int);fm,_=evaluate(m,d,val_pat,dev,th)
    train_pm=evaluate_pairs(m,d,delta,meta,tr,dev,cfg['phaseB']['target_margin_logit']);val_pm=evaluate_pairs(m,d,delta,meta,va,dev,cfg['phaseB']['target_margin_logit'])
    ndcfg=cfg['phaseB']['normalized_drift'];scales=make_drift_scales(train_pm,ndcfg['normalization_floor'],ndcfg.get('lordosis_weight',0.25))
    rep={
      'status':'PASS','validation_rows':len(va),'training_rows':len(tr),'validation_patients':len(val_pat),
      'phaseA_factual_on_cf_validation_patients':fm,
      'phaseA_cf_pair_baseline':val_pm,
      'phaseA_cf_pair_baseline_validation':val_pm,
      'phaseA_cf_pair_baseline_training':train_pm,
      'drift_normalization_scales':scales,
      'normalized_training_baseline':normalized_pair_drift(train_pm,scales),
      'normalized_validation_baseline':normalized_pair_drift(val_pm,scales),
      'normalization_policy':'Frozen from Phase-A Development CF TRAIN rows only. Internal100 never contributes.',
      'split_policy':'recipient-patient hash; no Internal100 selection'
    }
    dumpj(rep,out/'03_PHASEA_CF_BASELINE.json');torch.save({'train_rows':torch.tensor(tr),'val_rows':torch.tensor(va),'row_patient_idx':pi,'drift_normalization_scales':scales},out/'03_cf_split.pt');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
