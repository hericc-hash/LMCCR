from __future__ import annotations
from .metrics import language_value

def _m(m):return float(m['report_vs_gt']['micro_f1']),float(m['report_vs_core']['micro_f1']),language_value(m['language'],'rouge'),language_value(m['language'],'bleu')
def selection_gate(raw,final,proc,cfg):
    rc,rf,rr,rb=_m(raw);fc,ff,fr,fb=_m(final)
    checks={
      'parse_rate':raw.get('n',0)==50 and final.get('n',0)==50,
      'raw_fidelity':rf>=float(cfg['minimum_raw_report_vs_core_f1']),
      'raw_clinical':rc>=float(cfg['minimum_raw_clinical_f1']),
      'raw_rouge':rr is not None and rr>=float(cfg['minimum_raw_rouge_l']),
      'raw_bleu':rb is not None and rb>=float(cfg['minimum_raw_bleu4']),
      'final_fidelity':ff>=float(cfg['minimum_final_report_vs_core_f1']),
      'final_clinical':fc>=float(cfg['minimum_final_clinical_f1']),
      'final_rouge':fr is not None and fr>=float(cfg['minimum_final_rouge_l']),
      'final_bleu':fb is not None and fb>=float(cfg['minimum_final_bleu4']),
      'raw_final_similarity':proc['mean_similarity_raw_final']>=float(cfg['minimum_mean_similarity_raw_final']),
      'edit_fraction':proc['mean_edit_char_fraction']<=float(cfg['maximum_mean_edit_char_fraction']),
      'unresolved_slots':proc['mean_unresolved_slots']<=float(cfg['maximum_mean_unresolved_slots']),
      'targeted_patch_success':proc['targeted_patch_success_rate']>=float(cfg['minimum_targeted_patch_success_rate']),
      'no_fallback':proc['fallback_rate']==float(cfg['fallback_rate_must_equal']),
      'rouge_preserved_by_patch':rr is not None and fr is not None and (rr-fr)<=float(cfg['maximum_rouge_drop_after_patch']),
    }
    return {'pass':all(checks.values()),'checks':checks,'raw_clinical':rc,'raw_fidelity':rf,'raw_rouge':rr,'raw_bleu':rb,'final_clinical':fc,'final_fidelity':ff,'final_rouge':fr,'final_bleu':fb,'process':proc}
def holdout_gate(base,raw,final,proc,cfg):
    bc=float(base['report_vs_gt']['micro_f1']);rc,rf,rr,rb=_m(raw);fc,ff,fr,fb=_m(final)
    checks={
      'parse_rate':raw.get('n',0)==50 and final.get('n',0)==50,
      'final_fidelity':ff>=float(cfg['minimum_final_report_vs_core_f1']),
      'clinical_gain_vs_stage3a':fc-bc>=float(cfg['minimum_final_clinical_gain_vs_stage3a']),
      'final_rouge':fr is not None and fr>=float(cfg['minimum_final_rouge_l']),
      'final_bleu':fb is not None and fb>=float(cfg['minimum_final_bleu4']),
      'raw_final_similarity':proc['mean_similarity_raw_final']>=float(cfg['minimum_mean_similarity_raw_final']),
      'edit_fraction':proc['mean_edit_char_fraction']<=float(cfg['maximum_mean_edit_char_fraction']),
      'unresolved_slots':proc['mean_unresolved_slots']<=float(cfg['maximum_mean_unresolved_slots']),
      'targeted_patch_success':proc['targeted_patch_success_rate']>=float(cfg['minimum_targeted_patch_success_rate']),
      'no_fallback':proc['fallback_rate']==float(cfg['fallback_rate_must_equal']),
      'rouge_preserved_by_patch':rr is not None and fr is not None and (rr-fr)<=float(cfg['maximum_rouge_drop_after_patch']),
    }
    return {'pass':all(checks.values()),'checks':checks,'stage3a_clinical':bc,'raw_clinical':rc,'raw_fidelity':rf,'raw_rouge':rr,'raw_bleu':rb,'final_clinical':fc,'final_fidelity':ff,'final_rouge':fr,'final_bleu':fb,'process':proc}
