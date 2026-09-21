#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import safe_load,dumpj
from lumbar_cf_report.planner.model import Stage23UnifiedClinicalPlanner
from lumbar_cf_report.planner.training import evaluate,ensure_model_device,preservation_pass


def _build(cfg,ck,device):
    m=Stage23UnifiedClinicalPlanner(cfg,ck['anchor_weight'],ck['anchor_bias']);m.load_state_dict(ck['state_dict'],strict=True);ensure_model_device(m,device);return m


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out);sel=json.load(open(out/'05_CF_SELECTION_DECISION.json'))
 if not sel.get('selected_mode'):
  dumpj({'status':'SKIPPED_NO_FULL_CF_CANDIDATE'},out/'07_INTERNAL_FACTUAL_DECISION.json');return
 data=safe_load(out/'01_stage23b_data.pt');d=data['internal'];pa=safe_load(data['phaseA_candidate_path']);to=safe_load(out/'04_target_only_candidate.pt');ck=safe_load(out/'05_stage2_3B_selected_candidate.pt');th=ck['thresholds'].numpy();idx=np.arange(len(d['serials']));device=cfg['device']
 base=_build(cfg,pa,device);bm,_=evaluate(base,d,idx,device,th)
 tom=_build(cfg,to,device);tm,_=evaluate(tom,d,idx,device,th)
 m=_build(cfg,ck,device);cm,p=evaluate(m,d,idx,device,th)
 g=cfg['phaseB']['internal_acceptance_gate'];gt=cfg['phaseB'].get('internal_target_only_acceptance_gate',g)
 base_gate={'max_auprc_drop_per_task':g['max_auprc_drop_per_task'],'max_auc_drop_per_task':g['max_auc_drop_per_task'],'max_f1_drop_per_task':g['max_f1_drop_per_task'],'max_hard_task_mean_auprc_drop':g['max_hard_task_mean_auprc_drop']}
 to_gate={'max_auprc_drop_per_task':gt['max_auprc_drop_per_task'],'max_auc_drop_per_task':gt['max_auc_drop_per_task'],'max_f1_drop_per_task':gt['max_f1_drop_per_task'],'max_hard_task_mean_auprc_drop':gt['max_hard_task_mean_auprc_drop']}
 ok_phaseA=preservation_pass(cm,bm,base_gate);ok_target=preservation_pass(cm,tm,to_gate);ok=bool(ok_phaseA and ok_target)
 delta_pa={t:{k:cm[t][k]-bm[t][k] for k in ('auroc','auprc','f1')} for t in cm};delta_to={t:{k:cm[t][k]-tm[t][k] for k in ('auroc','auprc','f1')} for t in cm}
 rep={'status':'PASS' if ok else 'FAIL','internal_policy':'FROZEN_EVALUATION_ONLY_NO_SELECTION','phaseA_factual':bm,'target_only_factual':tm,'stage2_3B_factual':cm,'delta_stage2_3B_minus_phaseA':delta_pa,'delta_stage2_3B_minus_target_only':delta_to,'factual_preservation_vs_phaseA':ok_phaseA,'factual_preservation_vs_target_only':ok_target,'factual_copreservation_pass':ok,'selected_alpha':sel.get('selected_alpha'),'selected_eta':sel.get('selected_eta'),'nerve_recovery_delta_vs_target_only':{k:cm['nerve'][k]-tm['nerve'][k] for k in ('auroc','auprc','f1')},'nerve_recovery_delta_vs_phaseA':{k:cm['nerve'][k]-bm['nerve'][k] for k in ('auroc','auprc','f1')}};dumpj(rep,out/'07_INTERNAL_FACTUAL_DECISION.json');torch.save({'probs':torch.tensor(p)},out/'07_internal_factual_probs.pt');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
