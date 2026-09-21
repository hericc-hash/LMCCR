from __future__ import annotations
import copy, math, numpy as np, torch
import torch.nn.functional as F
from .training import ensure_model_device,evaluate,masked_bce,pos_weight,preservation_pass
from .cf import row_to_patient_indices,make_cf_e,evaluate_pairs


def _trainable(model, scope='broad_context'):
    """Return the explicitly allowed trainable parameter set.

    Stage2.3-B target-only keeps the historical broad bounded-context update.
    v1.5 legacy protected-drift refinement was intentionally narrower: only the
    cross-slot gate and disease residual heads can move. The raw-E anchor,
    slot encoder, cross projection, structure head and all factual evidence
    inputs remain frozen.
    """
    if scope=='broad_context':
        names=('slot_mlp','cross_proj','cross_gate_raw','residual_heads')
    elif scope=='gate_and_heads_only':
        names=('cross_gate_raw','residual_heads')
    else:
        raise ValueError(f'unknown trainable scope: {scope}')
    for n,p in model.named_parameters():
        p.requires_grad=any(x in n for x in names)
    pars=[p for p in model.parameters() if p.requires_grad]
    if not pars: raise RuntimeError(f'no trainable parameters for scope={scope}')
    return pars


def trainable_parameter_names(model):
    return [n for n,p in model.named_parameters() if p.requires_grad]


def _param_anchor(model,ref_state,device):
    z=torch.tensor(0.,device=device);n=0
    for name,p in model.named_parameters():
        if p.requires_grad and name in ref_state:
            q=ref_state[name].to(device);z=z+(p-q).pow(2).mean();n+=1
    return z/max(n,1)


def make_drift_scales(pair_metrics, floor=1e-5, lordosis_weight=0.25, source='PHASEA_DEVELOPMENT_CF_TRAIN_ROWS_ONLY'):
    disease=max(float(pair_metrics['protected_disease_drift']),float(floor))
    structure_raw=float(pair_metrics['protected_structure_drift'])+float(lordosis_weight)*float(pair_metrics.get('protected_lordosis_drift',0.0))
    structure=max(structure_raw,float(floor))
    return {
        'disease_scale':disease,
        'structure_scale':structure,
        'floor':float(floor),
        'lordosis_weight':float(lordosis_weight),
        'source':str(source),
    }




def context_rollback_state(phasea_state,target_only_state,alpha,rollback_groups,preserve_exact_groups=('residual_heads',)):
    """Deterministically move only shared context parameters toward Phase-A.

    v1.6 uses a parameter-space rollback rather than another gradient-based drift
    optimization. Target-only residual heads remain bit-exact, preserving the
    disease-specific CF response learned in the successful Target-only phase.
    """
    a=float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError(f'rollback alpha must be in [0,1], got {a}')
    if set(phasea_state.keys()) != set(target_only_state.keys()):
        miss_a=sorted(set(target_only_state)-set(phasea_state))[:10]
        miss_t=sorted(set(phasea_state)-set(target_only_state))[:10]
        raise RuntimeError(f'Phase-A/Target-only state key mismatch: missing_phaseA={miss_a} missing_target={miss_t}')
    out={}
    blended=[];preserved=[]
    for k,tv in target_only_state.items():
        pv=phasea_state[k]
        if tuple(tv.shape)!=tuple(pv.shape):
            raise RuntimeError(f'state shape mismatch for {k}: target={tuple(tv.shape)} phaseA={tuple(pv.shape)}')
        do_blend=any(g in k for g in rollback_groups)
        must_preserve=any(g in k for g in preserve_exact_groups)
        if do_blend and must_preserve:
            raise RuntimeError(f'parameter {k} matches both rollback and preserve groups')
        if do_blend:
            if not (torch.is_floating_point(tv) or torch.is_complex(tv)):
                raise TypeError(f'cannot interpolate non-floating parameter {k}: {tv.dtype}')
            out[k]=(1.0-a)*tv + a*pv
            blended.append(k)
        else:
            out[k]=tv.clone()
            if must_preserve:preserved.append(k)
    if not blended:
        raise RuntimeError(f'rollback groups matched no parameters: {rollback_groups}')
    if preserve_exact_groups and not preserved:
        raise RuntimeError(f'preserve groups matched no parameters: {preserve_exact_groups}')
    return out,{'alpha':a,'blended_parameters':blended,'preserved_exact_parameters':preserved}


def state_group_distance(state_a,state_b,groups):
    vals=[]
    for k in state_a:
        if any(g in k for g in groups):
            d=(state_a[k].detach().float()-state_b[k].detach().float()).reshape(-1)
            vals.append(d.pow(2).mean())
    if not vals:return float('nan')
    return float(torch.stack(vals).mean().sqrt().item())

def normalized_pair_drift(pair_metrics, scales):
    sd=max(float(scales['disease_scale']),1e-12);ss=max(float(scales['structure_scale']),1e-12)
    nd=float(pair_metrics['protected_disease_drift'])/sd
    ns=(float(pair_metrics['protected_structure_drift'])+float(scales.get('lordosis_weight',0.25))*float(pair_metrics.get('protected_lordosis_drift',0.0)))/ss
    return {'normalized_disease_drift':nd,'normalized_structure_drift':ns,'normalized_combined_drift':nd+0.5*ns}


def bounded_relative_penalty(raw, scale, cap=8.0):
    """Monotonic but bounded-gradient relative drift penalty.

    v1.4 used raw/scale directly, so an early structure ratio of ~680 dominated
    all factual/target losses. v1.5 uses log1p(raw/scale), optionally capped,
    keeping the penalty sensitive to drift while preventing scale explosions.
    """
    s=max(float(scale),1e-12)
    ratio=raw/s
    val=torch.log1p(ratio.clamp_min(0.0))
    if cap is not None: val=torch.clamp(val,max=float(cap))
    return val


def bounded_pair_drift(pair_metrics, scales, cap=8.0):
    nd=max(float(pair_metrics['protected_disease_drift'])/max(float(scales['disease_scale']),1e-12),0.0)
    ns=max((float(pair_metrics['protected_structure_drift'])+float(scales.get('lordosis_weight',0.25))*float(pair_metrics.get('protected_lordosis_drift',0.0)))/max(float(scales['structure_scale']),1e-12),0.0)
    bd=min(math.log1p(nd),float(cap));bs=min(math.log1p(ns),float(cap))
    return {'bounded_disease_drift':bd,'bounded_structure_drift':bs,'bounded_combined_drift':bd+0.5*bs}


def mechanism_score(pair_metrics, mode, scales, score_cfg):
    base=float(pair_metrics['target_success_rate'])+0.5*float(pair_metrics['target_margin_success_rate'])
    if mode=='target_only':return base
    b=bounded_pair_drift(pair_metrics,scales,score_cfg.get('log1p_cap',8.0))
    return base-float(score_cfg['disease_weight'])*b['bounded_disease_drift']-float(score_cfg['structure_weight'])*b['bounded_structure_drift']


def _factual_forward(model,d,patients,device):
    ii=torch.as_tensor(patients,dtype=torch.long)
    return model(d['global_source'][ii].to(device),d['raw_E'][ii].to(device),d['coordinate_state'][ii].to(device),d['task_quality'][ii].to(device),d['task_valid'][ii].to(device),False)


def _cf_forward(model,d,patients,rows,delta,meta,device):
    pat=torch.as_tensor(patients,dtype=torch.long);rr=np.asarray(rows,dtype=int)
    cfE,li,ti=make_cf_e(d['raw_E'],pat,rr,delta,meta)
    co=model(d['global_source'][pat].to(device),cfE.to(device),d['coordinate_state'][pat].to(device),d['task_quality'][pat].to(device),d['task_valid'][pat].to(device),False)
    return co,li,ti


def train_mode(model,phaseA_model,d,delta,meta,train_rows,val_rows,device,cfg,thresholds,base_factual,mode,seed,
               drift_scales=None,profile=None,target_reference_pair=None):
    """Historical v1.4 target-only trainer retained for attribution.

    v1.5 calls this only for target_only. Protected-drift refinement uses
    train_target_first_refinement below.
    """
    device=ensure_model_device(model,device);ensure_model_device(phaseA_model,device);phaseA_model.eval()
    pars=_trainable(model,'broad_context');opt=torch.optim.AdamW(pars,lr=cfg['lr'],weight_decay=cfg['weight_decay']);dv=(d['label_valid']*d['task_valid']);pw=pos_weight(d['labels'].to(device),dv.to(device));ref_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};pi=row_to_patient_indices(meta,d['serials']);sign=torch.as_tensor(meta['sign']).float();rng=np.random.default_rng(seed);all_pat=np.arange(len(d['serials']))
    val_pat=np.asarray(sorted(set(pi[val_rows].tolist())),dtype=int)
    score_cfg=cfg['normalized_drift']['selection_score']
    if drift_scales is None: drift_scales={'disease_scale':1.0,'structure_scale':1.0,'floor':1e-5,'lordosis_weight':0.25,'source':'FALLBACK_UNIT'}
    initial_pair=evaluate_pairs(model,d,delta,meta,val_rows,device,cfg['target_margin_logit'])
    best={'epoch':0,'score':mechanism_score(initial_pair,'target_only',drift_scales,score_cfg),'state':copy.deepcopy(model.state_dict()),'factual':base_factual,'pair':initial_pair,'normalized_pair':normalized_pair_drift(initial_pair,drift_scales),'mechanism_epoch_eligible':True}
    hist=[];bad=0
    for ep in range(1,cfg['max_epochs']+1):
        model.train();perm=rng.permutation(train_rows);losses=[];components={'target':[],'factual':[],'distill':[],'anchor':[]}
        for st in range(0,len(perm),cfg['batch_size']):
            rr=np.asarray(perm[st:st+cfg['batch_size']],dtype=int);pat=pi[rr];cfE,li,ti=make_cf_e(d['raw_E'],pat,rr,delta,meta)
            fact=_factual_forward(model,d,pat,device);co=model(d['global_source'][pat].to(device),cfE.to(device),d['coordinate_state'][pat].to(device),d['task_quality'][pat].to(device),d['task_valid'][pat].to(device),False)
            target=[]
            for j,r in enumerate(rr):
                l=int(li[j]);t=int(ti[j]);sg=sign[r].to(device);shift=sg*(co['disease_logits'][j,l,t]-fact['disease_logits'][j,l,t]);target.append(F.relu(torch.tensor(cfg['target_margin_logit'],device=device)-shift))
            extra=rng.choice(all_pat,size=min(cfg['factual_rehearsal_extra_patients'],len(all_pat)),replace=False);fp=np.unique(np.concatenate([pat.numpy(),extra]));fp_t=torch.tensor(fp,dtype=torch.long)
            fo=_factual_forward(model,d,fp_t,device)
            with torch.inference_mode():ba=_factual_forward(phaseA_model,d,fp_t,device)
            y=d['labels'][fp_t].to(device);vv=dv[fp_t].to(device);fl=masked_bce(fo['disease_logits'],y,vv,pw);dist=((fo['disease_logits']-ba['disease_logits'])**2*vv).sum()/vv.sum().clamp_min(1.)
            target_raw=torch.stack(target).mean();anchor=_param_anchor(model,ref_state,device)
            loss=cfg['target_loss_weight']*target_raw+cfg['factual_rehearsal_weight']*fl+cfg['phaseA_distill_weight']*dist+cfg['parameter_anchor_l2_weight']*anchor
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pars,cfg['max_grad_norm']);opt.step();losses.append(float(loss.detach()))
            for k,v in [('target',target_raw),('factual',fl),('distill',dist),('anchor',anchor)]:components[k].append(float(v.detach()))
        fm,_=evaluate(model,d,val_pat,device,thresholds);pm=evaluate_pairs(model,d,delta,meta,val_rows,device,cfg['target_margin_logit']);pres=preservation_pass(fm,base_factual,cfg['development_preservation_gate']);score=mechanism_score(pm,'target_only',drift_scales,score_cfg)
        hist.append({'epoch':ep,'loss':float(np.mean(losses)) if losses else float('nan'),'loss_components':{k:float(np.mean(v)) if v else float('nan') for k,v in components.items()},'preservation_pass':pres,'mechanism_epoch_eligible':bool(pres),'factual':fm,'pair':pm,'normalized_pair':normalized_pair_drift(pm,drift_scales),'score':score})
        if pres and score>best['score']+1e-9:
            best={'epoch':ep,'score':score,'state':copy.deepcopy(model.state_dict()),'factual':fm,'pair':pm,'normalized_pair':normalized_pair_drift(pm,drift_scales),'mechanism_epoch_eligible':True};bad=0
        elif pres: bad+=1
        else: bad=0
        if ep>=cfg['min_epochs'] and bad>=cfg['patience']:break
    model.load_state_dict(best['state']);return best,hist


def train_target_first_refinement(model,target_ref_model,d,delta,meta,train_rows,val_rows,device,cfg,thresholds,
                                  target_ref_factual,target_ref_pair,phaseA_factual,seed,drift_scales,profile):
    """v1.5 protected-drift refinement starting from target-only best.

    Hard design constraints:
      * initialization = target-only best checkpoint
      * trainable scope = cross_gate_raw + residual_heads only
      * factual distillation target = frozen target-only model
      * target-shift preservation = one-sided hinge vs frozen target-only shift
      * drift loss = bounded log1p(relative drift), not raw ratio
      * candidate epoch must preserve factual metrics and target response before
        drift-based score can select it
    """
    rcfg=cfg['target_first_refinement']
    device=ensure_model_device(model,device);ensure_model_device(target_ref_model,device);target_ref_model.eval();[q.requires_grad_(False) for q in target_ref_model.parameters()]
    scope=rcfg.get('trainable_scope','gate_and_heads_only');pars=_trainable(model,scope)
    lr=float(profile.get('lr',rcfg['lr']));opt=torch.optim.AdamW(pars,lr=lr,weight_decay=float(rcfg['weight_decay']))
    dv=(d['label_valid']*d['task_valid']);pw=pos_weight(d['labels'].to(device),dv.to(device));ref_state={k:v.detach().cpu().clone() for k,v in target_ref_model.state_dict().items()};pi=row_to_patient_indices(meta,d['serials']);sign=torch.as_tensor(meta['sign']).float();rng=np.random.default_rng(seed);all_pat=np.arange(len(d['serials']))
    val_pat=np.asarray(sorted(set(pi[val_rows].tolist())),dtype=int);score_cfg=rcfg['selection_score'];cap=float(rcfg.get('log1p_cap',8.0));warm=max(int(rcfg.get('warmup_epochs',6)),1)
    best={'epoch':0,'score':-1e18,'state':copy.deepcopy(model.state_dict()),'factual':target_ref_factual,'pair':target_ref_pair,'normalized_pair':normalized_pair_drift(target_ref_pair,drift_scales),'bounded_pair':bounded_pair_drift(target_ref_pair,drift_scales,cap),'mechanism_epoch_eligible':False,'trainable_scope':scope}
    hist=[];bad=0
    target_success_tol=float(rcfg['target_success_tolerance']);target_margin_tol=float(rcfg['target_margin_success_tolerance']);mean_shift_tol=float(rcfg['mean_signed_shift_tolerance'])
    for ep in range(1,int(rcfg['max_epochs'])+1):
        model.train();perm=rng.permutation(train_rows);losses=[];components={k:[] for k in ('target','target_preserve','factual','target_only_distill','anchor','disease_raw','structure_raw','disease_bounded','structure_bounded','drift_ramp')}
        ramp=min(1.0,float(ep)/float(warm))
        for st in range(0,len(perm),int(rcfg['batch_size'])):
            rr=np.asarray(perm[st:st+int(rcfg['batch_size'])],dtype=int);pat=pi[rr];cfE,li,ti=make_cf_e(d['raw_E'],pat,rr,delta,meta)
            fact=_factual_forward(model,d,pat,device);co=model(d['global_source'][pat].to(device),cfE.to(device),d['coordinate_state'][pat].to(device),d['task_quality'][pat].to(device),d['task_valid'][pat].to(device),False)
            with torch.inference_mode():
                ref_fact=_factual_forward(target_ref_model,d,pat,device);ref_co=target_ref_model(d['global_source'][pat].to(device),cfE.to(device),d['coordinate_state'][pat].to(device),d['task_quality'][pat].to(device),d['task_valid'][pat].to(device),False)
            target=[];target_pres=[];prot_d=[];prot_s=[]
            for j,r in enumerate(rr):
                l=int(li[j]);t=int(ti[j]);sg=sign[r].to(device)
                shift=sg*(co['disease_logits'][j,l,t]-fact['disease_logits'][j,l,t])
                ref_shift=sg*(ref_co['disease_logits'][j,l,t]-ref_fact['disease_logits'][j,l,t])
                target.append(F.relu(torch.tensor(float(rcfg['target_margin_logit']),device=device)-shift))
                target_pres.append(F.relu((ref_shift-float(rcfg['target_shift_tolerance_logit']))-shift))
                mask=torch.ones((5,3),dtype=torch.bool,device=device);mask[l,t]=False
                prot_d.append((co['disease_probabilities'][j][mask]-fact['disease_probabilities'][j][mask]).abs().mean())
                prot_s.append((co['structure_probabilities'][j]-fact['structure_probabilities'][j]).abs().mean()+float(drift_scales.get('lordosis_weight',0.25))*(torch.sigmoid(co['main_logits'][j,0])-torch.sigmoid(fact['main_logits'][j,0])).abs())
            extra=rng.choice(all_pat,size=min(int(rcfg['factual_rehearsal_extra_patients']),len(all_pat)),replace=False);fp=np.unique(np.concatenate([pat.numpy(),extra]));fp_t=torch.tensor(fp,dtype=torch.long)
            fo=_factual_forward(model,d,fp_t,device)
            with torch.inference_mode():ro=_factual_forward(target_ref_model,d,fp_t,device)
            y=d['labels'][fp_t].to(device);vv=dv[fp_t].to(device);fl=masked_bce(fo['disease_logits'],y,vv,pw);dist=((fo['disease_logits']-ro['disease_logits'])**2*vv).sum()/vv.sum().clamp_min(1.)
            target_raw=torch.stack(target).mean();tp_raw=torch.stack(target_pres).mean();pd_raw=torch.stack(prot_d).mean();ps_raw=torch.stack(prot_s).mean();pd_bound=bounded_relative_penalty(pd_raw,drift_scales['disease_scale'],cap);ps_bound=bounded_relative_penalty(ps_raw,drift_scales['structure_scale'],cap);anchor=_param_anchor(model,ref_state,device)
            loss=(float(rcfg['target_loss_weight'])*target_raw+float(rcfg['target_preservation_weight'])*tp_raw+float(rcfg['factual_rehearsal_weight'])*fl+float(rcfg['target_only_distill_weight'])*dist+float(rcfg['parameter_anchor_l2_weight'])*anchor+ramp*float(profile['disease_weight'])*pd_bound+ramp*float(profile['structure_weight'])*ps_bound)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pars,float(rcfg['max_grad_norm']));opt.step();losses.append(float(loss.detach()))
            for k,v in [('target',target_raw),('target_preserve',tp_raw),('factual',fl),('target_only_distill',dist),('anchor',anchor),('disease_raw',pd_raw),('structure_raw',ps_raw),('disease_bounded',pd_bound),('structure_bounded',ps_bound)]:components[k].append(float(v.detach()))
            components['drift_ramp'].append(float(ramp))
        fm,_=evaluate(model,d,val_pat,device,thresholds);pm=evaluate_pairs(model,d,delta,meta,val_rows,device,float(rcfg['target_margin_logit']))
        pres_target=preservation_pass(fm,target_ref_factual,rcfg['development_preservation_vs_target_only'])
        pres_phasea=preservation_pass(fm,phaseA_factual,rcfg['development_preservation_vs_phaseA'])
        target_ok=(pm['target_success_rate']>=float(target_ref_pair['target_success_rate'])-target_success_tol and pm['target_margin_success_rate']>=float(target_ref_pair['target_margin_success_rate'])-target_margin_tol and pm['mean_signed_target_logit_shift']>=float(target_ref_pair['mean_signed_target_logit_shift'])-mean_shift_tol)
        eligible=bool(pres_target and pres_phasea and target_ok)
        score=mechanism_score(pm,'target_plus_drift',drift_scales,score_cfg);norm=normalized_pair_drift(pm,drift_scales);bounded=bounded_pair_drift(pm,drift_scales,cap)
        hist.append({'epoch':ep,'loss':float(np.mean(losses)) if losses else float('nan'),'loss_components':{k:float(np.mean(v)) if v else float('nan') for k,v in components.items()},'preservation_vs_target_only':pres_target,'preservation_vs_phaseA':pres_phasea,'target_noninferior_to_target_only':target_ok,'mechanism_epoch_eligible':eligible,'factual':fm,'pair':pm,'normalized_pair':norm,'bounded_pair':bounded,'score':score,'profile':profile,'trainable_scope':scope})
        if eligible and score>best['score']+1e-9:
            best={'epoch':ep,'score':score,'state':copy.deepcopy(model.state_dict()),'factual':fm,'pair':pm,'normalized_pair':norm,'bounded_pair':bounded,'mechanism_epoch_eligible':True,'trainable_scope':scope};bad=0
        elif eligible: bad+=1
        else: bad=0
        if ep>=int(rcfg['min_epochs']) and bad>=int(rcfg['patience']):break
    model.load_state_dict(best['state']);return best,hist
