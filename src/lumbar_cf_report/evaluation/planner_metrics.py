from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from ..clinical.schema import MAIN_SLOT_NAMES,related_structure_indices,disease_slot_index


def _macro_slot_auc(probs,labels,valid,min_pos=5,min_neg=5):
    p=np.asarray(probs,float); y=np.asarray(labels,int); v=np.asarray(valid)>0.5; vals=[]; aps=[]; rows=[]
    for j,n in enumerate(MAIN_SLOT_NAMES):
        m=v[:,j]; yy=y[m,j]; pp=p[m,j]; pos=int((yy==1).sum()); neg=int((yy==0).sum()); auc=ap=None
        if pos>0 and neg>0: auc=float(roc_auc_score(yy,pp)); ap=float(average_precision_score(yy,pp))
        eligible=pos>=min_pos and neg>=min_neg
        if eligible and auc is not None: vals.append(auc); aps.append(ap)
        rows.append({'slot':n,'n':int(m.sum()),'positive':pos,'negative':neg,'auc':auc,'auprc':ap,'eligible':eligible})
    return {'eligible_macro_auc':float(np.mean(vals)) if vals else None,'eligible_macro_auprc':float(np.mean(aps)) if aps else None,'eligible_slots':len(vals),'per_slot':rows}


def _stack(ds, serials, key):
    return torch.stack([torch.as_tensor(ds.item_by_serial(int(s))[key]).float() for s in serials],0)


def run_full_planner_audit(ds,handoff,planner,device,output_dir: Path,batch_size=16):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); planner=planner.to(device).eval(); bs=max(1,int(batch_size))
    serial_order=[int(x) for x in ds.serials.tolist()]; factual={}; fp_all=[]; labels=[]; valids=[]
    with torch.inference_mode():
        for st in range(0,len(serial_order),bs):
            ss=serial_order[st:st+bs]; g=_stack(ds,ss,'global_source').to(device); e=torch.stack([handoff.factual_e(s) for s in ss],0).to(device); c=_stack(ds,ss,'coordinate_state').to(device); q=_stack(ds,ss,'task_quality').to(device); v=_stack(ds,ss,'task_valid').to(device); o=planner(g,e,c,q,v)
            for bi,s in enumerate(ss):
                mp=o['main_probabilities'][bi].detach().cpu().numpy(); sp=o['structure_probabilities'][bi].detach().cpu().numpy(); factual[s]={'main_probabilities':mp,'structure_probabilities':sp}; fp_all.append(mp); x=ds.item_by_serial(s); labels.append(x['slot_labels'].numpy()); valids.append(x['slot_valid'].numpy())
    rows=[]; signed=[]; direction_ok=[]; donor_cross=[]; protected=[]; unstruct=[]; keys=sorted(handoff.records)
    with torch.inference_mode():
        for st in range(0,len(keys),bs):
            kk=keys[st:st+bs]; ss=[k[0] for k in kk]; g=_stack(ds,ss,'global_source').to(device); e=torch.stack([handoff.counterfactual_e(*k) for k in kk],0).to(device); c=_stack(ds,ss,'coordinate_state').to(device); q=_stack(ds,ss,'task_quality').to(device); v=_stack(ds,ss,'task_valid').to(device); pc=planner(g,e,c,q,v)
            for bi,(rid,li,ti) in enumerate(kk):
                r=handoff.record(rid,li,ti); p0=factual[rid]['main_probabilities']; p1=pc['main_probabilities'][bi].detach().cpu().numpy(); sidx=disease_slot_index(li,ti); donor=int(round(float(r['donor_target_label']))); direction=2*donor-1; ssigned=float(direction*(p1[sidx]-p0[sidx])); signed.append(ssigned); direction_ok.append(float(ssigned>0)); donor_cross.append(float(int(p1[sidx]>=.5)==donor)); mask=np.ones(len(MAIN_SLOT_NAMES),bool); mask[0]=False; mask[sidx]=False; pd=float(np.mean(np.abs(p1[mask]-p0[mask]))); protected.append(pd); sp0=factual[rid]['structure_probabilities']; sp1=pc['structure_probabilities'][bi].detach().cpu().numpy(); allowed=set(related_structure_indices(ti)); uu=[k for k in range(len(sp0)) if k not in allowed]; ud=float(np.mean(np.abs(sp1[uu]-sp0[uu]))) if uu else 0.0; unstruct.append(ud)
                rows.append({'recipient_serial':rid,'donor_serial':int(r['donor_id']),'level_index':li,'task_index':ti,'slot_index':sidx,'slot_name':MAIN_SLOT_NAMES[sidx],'recipient_label':float(r['recipient_target_label']),'donor_label':float(r['donor_target_label']),'coordinate_distance':float(r.get('coordinate_distance',float('nan'))),'target_signed_shift':ssigned,'target_direction_correct':int(ssigned>0),'target_crosses_to_donor':int(int(p1[sidx]>=.5)==donor),'protected_main_probability_drift':pd,'unrelated_structure_probability_drift':ud,'factual_main_probabilities':p0.tolist(),'transport_cf_main_probabilities':p1.tolist()})
            done=min(st+len(kk),len(keys))
            if done%150==0 or done==len(keys): print(f'[planner audit] {done}/1500')
    jp=output_dir/'planner_full_1500.jsonl'
    with jp.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary={'status':'finished','factual_cases':100,'interventions':1500,'factual_planner':_macro_slot_auc(fp_all,labels,valids),'transport_to_planner':{'mean_target_signed_shift':float(np.mean(signed)),'target_direction_accuracy':float(np.mean(direction_ok)),'target_crosses_to_donor_fraction':float(np.mean(donor_cross)),'mean_protected_main_probability_drift':float(np.mean(protected)),'mean_unrelated_structure_probability_drift':float(np.mean(unstruct))},'jsonl':str(jp.resolve()),'batch_size':bs}
    (output_dir/'planner_full_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); return summary

