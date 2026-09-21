from __future__ import annotations
import hashlib,numpy as np,torch

def serial_hash(s):return int(hashlib.sha256(str(s).encode()).hexdigest()[:8],16)

def _tolist(v):
    if torch.is_tensor(v):return v.detach().cpu().reshape(-1).tolist()
    if isinstance(v,np.ndarray):return v.reshape(-1).tolist()
    return list(v)

def row_to_patient_indices(meta,serials):
    smap={str(s):i for i,s in enumerate(serials)};rs=[str(v.item() if hasattr(v,'item') else v) for v in _tolist(meta['recipient_serial'])]
    idx=[];missing=[]
    for i,s in enumerate(rs):
        if s not in smap:missing.append((i,s));idx.append(-1)
        else:idx.append(smap[s])
    if missing:raise RuntimeError(f'P3 recipient mapping failed: {missing[:10]}')
    return torch.tensor(idx,dtype=torch.long)

def split_rows(meta,serials,modulus=5,remainder=0):
    pi=row_to_patient_indices(meta,serials);unique=sorted(set(pi.tolist()))
    val_pat={i for i in unique if serial_hash(serials[i])%modulus==remainder}
    tr=[r for r,p in enumerate(pi.tolist()) if p not in val_pat];va=[r for r,p in enumerate(pi.tolist()) if p in val_pat]
    return np.asarray(tr,dtype=int),np.asarray(va,dtype=int),pi

def make_cf_e(raw_E,patient_idx,row_idx,delta,meta):
    """Apply frozen P3 response at one lumbar level.

    Production P3 delta is [rows,3,128]: the complete level-wise response across
    disc/stenosis/nerve. A legacy [rows,128] target-only tensor is supported only
    for synthetic backward-compatibility tests and is not considered exact P3.
    """
    pi=torch.as_tensor(patient_idx,dtype=torch.long);ri=torch.as_tensor(row_idx,dtype=torch.long)
    E=raw_E[pi].clone();li=torch.as_tensor(_tolist(meta['level'])).long()[ri];ti=torch.as_tensor(_tolist(meta['task'])).long()[ri]
    dd=torch.as_tensor(delta)
    for j in range(len(ri)):
        r=int(ri[j]);l=int(li[j]);t=int(ti[j])
        if dd.ndim==3 and tuple(dd.shape[1:])==(3,128):E[j,l]+=dd[r]
        elif dd.ndim==2 and dd.shape[1]==128:E[j,l,t]+=dd[r]
        else:raise ValueError(f'P3 delta must be [M,3,128] (production) or [M,128] (legacy test), got {tuple(dd.shape)}')
    return E,li,ti

def pair_metrics(fact,cf,li,ti,sign,margin=0.15):
    B=len(li);shift=[];margin_ok=[];disease=[];struct=[];lord=[]
    for j in range(B):
        l=int(li[j]);t=int(ti[j]);sg=float(sign[j]);ds=float((cf['disease_logits'][j,l,t]-fact['disease_logits'][j,l,t]).detach().cpu());shift.append(sg*ds);margin_ok.append(sg*ds>=margin)
        mask=torch.ones((5,3),dtype=torch.bool,device=cf['disease_probabilities'].device);mask[l,t]=False
        disease.append(float((cf['disease_probabilities'][j][mask]-fact['disease_probabilities'][j][mask]).abs().mean().detach().cpu()))
        struct.append(float((cf['structure_probabilities'][j]-fact['structure_probabilities'][j]).abs().mean().detach().cpu()))
        lord.append(float((torch.sigmoid(cf['main_logits'][j,0])-torch.sigmoid(fact['main_logits'][j,0])).abs().detach().cpu()))
    a=np.asarray(shift,float);return {'n':B,'mean_signed_target_logit_shift':float(a.mean()) if B else float('nan'),'median_signed_target_logit_shift':float(np.median(a)) if B else float('nan'),'target_success_rate':float((a>0).mean()) if B else float('nan'),'target_margin_success_rate':float(np.mean(margin_ok)) if B else float('nan'),'protected_disease_drift':float(np.mean(disease)) if B else float('nan'),'protected_disease_drift_p95':float(np.quantile(disease,.95)) if B else float('nan'),'protected_structure_drift':float(np.mean(struct)) if B else float('nan'),'protected_lordosis_drift':float(np.mean(lord)) if B else float('nan')}

def evaluate_pairs(model,d,delta,meta,rows,device,margin=0.15,batch=128):
    pi=row_to_patient_indices(meta,d['serials']);sign=torch.as_tensor(_tolist(meta['sign'])).float();vals={k:[] for k in ('signed','success','margin','disease','structure','lordosis')}
    model.eval()
    with torch.inference_mode():
        for st in range(0,len(rows),batch):
            rr=np.asarray(rows,dtype=int)[st:st+batch];pat=pi[rr];cfE,li,ti=make_cf_e(d['raw_E'],pat,rr,delta,meta)
            fact=model(d['global_source'][pat].to(device),d['raw_E'][pat].to(device),d['coordinate_state'][pat].to(device),d['task_quality'][pat].to(device),d['task_valid'][pat].to(device),False)
            cf=model(d['global_source'][pat].to(device),cfE.to(device),d['coordinate_state'][pat].to(device),d['task_quality'][pat].to(device),d['task_valid'][pat].to(device),False)
            for j,r in enumerate(rr):
                l=int(li[j]);t=int(ti[j]);sg=float(sign[r]);ss=sg*float((cf['disease_logits'][j,l,t]-fact['disease_logits'][j,l,t]).detach().cpu());vals['signed'].append(ss);vals['success'].append(ss>0);vals['margin'].append(ss>=margin)
                mask=torch.ones((5,3),dtype=torch.bool,device=cf['disease_probabilities'].device);mask[l,t]=False
                vals['disease'].append(float((cf['disease_probabilities'][j][mask]-fact['disease_probabilities'][j][mask]).abs().mean().detach().cpu()))
                vals['structure'].append(float((cf['structure_probabilities'][j]-fact['structure_probabilities'][j]).abs().mean().detach().cpu()))
                vals['lordosis'].append(float((torch.sigmoid(cf['main_logits'][j,0])-torch.sigmoid(fact['main_logits'][j,0])).abs().detach().cpu()))
    n=len(vals['signed']);a=np.asarray(vals['signed'],float);return {'n':n,'mean_signed_target_logit_shift':float(a.mean()) if n else float('nan'),'median_signed_target_logit_shift':float(np.median(a)) if n else float('nan'),'target_success_rate':float(np.mean(vals['success'])) if n else float('nan'),'target_margin_success_rate':float(np.mean(vals['margin'])) if n else float('nan'),'protected_disease_drift':float(np.mean(vals['disease'])) if n else float('nan'),'protected_disease_drift_p95':float(np.quantile(vals['disease'],.95)) if n else float('nan'),'protected_structure_drift':float(np.mean(vals['structure'])) if n else float('nan'),'protected_structure_drift_p95':float(np.quantile(vals['structure'],.95)) if n else float('nan'),'protected_lordosis_drift':float(np.mean(vals['lordosis'])) if n else float('nan')}
