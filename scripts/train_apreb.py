#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lumbar_cf_report.apreb.dataset import APREBDataset
from lumbar_cf_report.apreb.checkpoints import load_init_ar_checkpoint, load_ar_checkpoint, load_anatomy_probe, safe_torch_load, save_ar_checkpoint
from lumbar_cf_report.apreb.planner_adapter import load_frozen_planner
from lumbar_cf_report.apreb.probes import prepare_probe_bundle, probe_logit, probe_probability
from lumbar_cf_report.apreb.donors import build_coordinate_candidate_cache, build_external_coordinate_candidate_cache, sample_counterfactual_targets
from lumbar_cf_report.apreb.metrics import per_slot_metrics


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--train_dataset',required=True);p.add_argument('--val_dataset',required=True)
    p.add_argument('--init_ar_checkpoint',required=True,help='Compatible v2 checkpoint used to initialize residual state')
    p.add_argument('--factual_teacher_ar_checkpoint',required=True,help='Frozen factual anatomy/residual teacher checkpoint')
    p.add_argument('--anatomy_probe_checkpoint',required=True);p.add_argument('--cross_task_probe_bundle',required=True);p.add_argument('--planner_checkpoint',required=True)
    p.add_argument('--output_dir',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch_size',type=int,default=16);p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--lr',type=float,default=4e-5);p.add_argument('--weight_decay',type=float,default=1e-4);p.add_argument('--seed',type=int,default=20260814)
    p.add_argument('--donor_topk',type=int,default=5);p.add_argument('--quality_weight',type=float,default=0.25);p.add_argument('--cf_slots_per_case',type=int,default=1)
    p.add_argument('--cf_alphas',default='0.25,0.5,0.75,1.0');p.add_argument('--cf_warmup_epochs',type=int,default=1);p.add_argument('--cf_ramp_epochs',type=int,default=4)
    # factual anchors
    p.add_argument('--w_evidence',type=float,default=1.0);p.add_argument('--w_pathology',type=float,default=0.75);p.add_argument('--w_residual_compact',type=float,default=1e-4)
    p.add_argument('--w_teacher_evidence',type=float,default=1.0);p.add_argument('--w_teacher_planner',type=float,default=0.50);p.add_argument('--w_original_planner',type=float,default=0.75)
    # multi-strength counterfactual objectives
    p.add_argument('--w_cf_planner_target',type=float,default=1.0);p.add_argument('--w_cf_probe_target',type=float,default=0.75)
    p.add_argument('--w_cf_endpoint_bce',type=float,default=0.35);p.add_argument('--w_cf_non_target',type=float,default=0.45)
    p.add_argument('--w_cf_path_ratio',type=float,default=0.50);p.add_argument('--path_ratio_rho',type=float,default=0.35);p.add_argument('--path_ratio_eps',type=float,default=0.005)
    p.add_argument('--low_strength_boost',type=float,default=1.0)
    p.add_argument('--w_cf_anatomy_segment',type=float,default=0.50);p.add_argument('--w_cf_anatomy_coordinate',type=float,default=0.25);p.add_argument('--w_cf_anatomy_guard',type=float,default=0.25)
    p.add_argument('--w_cf_monotonic_planner',type=float,default=0.25);p.add_argument('--w_cf_monotonic_probe',type=float,default=0.25);p.add_argument('--monotonic_margin',type=float,default=0.0)
    p.add_argument('--planner_direction_margin',type=float,default=0.08);p.add_argument('--probe_direction_margin',type=float,default=0.08)
    p.add_argument('--factual_cosine_floor',type=float,default=0.95);p.add_argument('--factual_planner_drift_target',type=float,default=0.06)
    p.add_argument('--min_positive',type=int,default=5);p.add_argument('--min_negative',type=int,default=5);p.add_argument('--overwrite',action='store_true')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def parse_alphas(s):
    vals=sorted(set(float(x) for x in s.split(',') if x.strip()))
    vals=[x for x in vals if 0.0 < x <= 1.0]
    if not vals: raise ValueError('cf_alphas must contain values in (0,1]')
    if abs(vals[-1]-1.0)>1e-8: vals.append(1.0)
    return vals

def batch_from_indices(ds,idx,device):
    idx=torch.as_tensor(idx).long()
    return {'indices':idx,'serials':ds.serials[idx].to(device),'segment_base':ds.segment_base[idx].to(device),'task_features':ds.task_features[idx].to(device),
            'coordinate_state':ds.coordinate_state[idx].to(device),'task_quality':ds.task_quality[idx].to(device),'task_valid':ds.task_valid[idx].to(device),
            'labels':ds.labels[idx].to(device),'label_valid':ds.label_valid[idx].to(device),'global_source':ds.global_source[idx].to(device)}

def pos_weight(ds):
    y,v=ds.labels,ds.label_valid;pos=(y*v).sum(0);neg=((1-y)*v).sum(0);return (neg/pos.clamp_min(1)).clamp(0.25,20.0)

def masked_bce(logits,labels,valid,pw):
    loss=F.binary_cross_entropy_with_logits(logits,labels,reduction='none',pos_weight=pw);return (loss*valid).sum()/valid.sum().clamp_min(1)

def cosine_loss(a,b): return (1-F.cosine_similarity(a,b,dim=-1)).mean()

def cf_ramp(epoch,warmup,ramp_epochs):
    if epoch<=warmup:return 0.0
    if ramp_epochs<=0:return 1.0
    return float(min(1.0,max(0.0,(epoch-warmup)/float(ramp_epochs))))

def freeze_anatomy_pathway(model):
    names=['coordinate_encoder','anatomy_encoder','task_embedding','anatomy_evidence_decoder','segment_decoder','coordinate_decoder']
    for name in names:
        m=getattr(model,name)
        for p in m.parameters():p.requires_grad=False
        m.eval()
    return names

def enforce_frozen_eval(model,names):
    for n in names:getattr(model,n).eval()

def parameter_report(model):
    return {'trainable_parameters':int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            'frozen_parameters':int(sum(p.numel() for p in model.parameters() if not p.requires_grad)),
            'trainable_tensors':[n for n,p in model.named_parameters() if p.requires_grad]}


def factual_eval(model,teacher,planner,ds,device,batch_size,minp,minn):
    model.eval();teacher.eval();planner.eval();ys=[];vs=[];ps=[];recon=[];tcos=[];tdrift=[];odrift=[]
    with torch.no_grad():
        for st in range(0,len(ds),batch_size):
            en=min(len(ds),st+batch_size);seg=ds.segment_base[st:en].to(device);x=ds.task_features[st:en].to(device);c=ds.coordinate_state[st:en].to(device)
            o=model(seg,x,c);to=teacher(seg,x,c);ys.append(ds.labels[st:en]);vs.append(ds.label_valid[st:en]);ps.append(o['pathology_probabilities'].cpu())
            recon.append(F.cosine_similarity(o['reconstructed_task_features'],x,dim=-1).mean().cpu());tcos.append(F.cosine_similarity(o['reconstructed_task_features'],to['reconstructed_task_features'],dim=-1).mean().cpu())
            g=ds.global_source[st:en].to(device);q=ds.task_quality[st:en].to(device);tv=ds.task_valid[st:en].to(device)
            pcur=planner(g,o['reconstructed_task_features'],c,q,tv)['slot_probabilities'];pt=planner(g,to['reconstructed_task_features'],c,q,tv)['slot_probabilities'];po=planner(g,x,c,q,tv)['slot_probabilities']
            tdrift.append((pcur-pt).abs().mean().cpu());odrift.append((pcur-po).abs().mean().cpu())
    y,v,p=torch.cat(ys),torch.cat(vs),torch.cat(ps);_,agg=per_slot_metrics(y,p,v,minp,minn)
    agg.update({'evidence_reconstruction_cosine':float(torch.stack(recon).mean()),'teacher_evidence_cosine':float(torch.stack(tcos).mean()),
                'planner_probability_drift_vs_teacher':float(torch.stack(tdrift).mean()),'planner_probability_drift_vs_original_X':float(torch.stack(odrift).mean())})
    return agg


def quick_multistrength_eval(model,tr,va,planner,anatomy_probe,probes,val_cache,device,alphas,max_pairs=256):
    model.eval();rows=[];trajectory={}
    with torch.no_grad():
        for ri in range(len(va)):
            if len(trajectory)>=max_pairs:break
            ro=model(va.segment_base[ri:ri+1].to(device),va.task_features[ri:ri+1].to(device),va.coordinate_state[ri:ri+1].to(device));xf=ro['reconstructed_task_features']
            pf=planner(va.global_source[ri:ri+1].to(device),xf,va.coordinate_state[ri:ri+1].to(device),va.task_quality[ri:ri+1].to(device),va.task_valid[ri:ri+1].to(device))['slot_probabilities']
            for li in range(5):
                for ti in range(3):
                    key=(ri,li,ti)
                    if key not in val_cache or not val_cache[key]:continue
                    di=val_cache[key][0].index;do=model(tr.segment_base[di:di+1].to(device),tr.task_features[di:di+1].to(device),tr.coordinate_state[di:di+1].to(device));direction=2*float(tr.labels[di,li,ti])-1;slot=1+ti*5+li
                    own=probes.get(f'level{li}_source{ti}_target{ti}');fact=xf[:,li,ti];r0=ro['residual'][:,li,ti];rd=do['residual'][:,li,ti]
                    traj=[]
                    for alpha in alphas:
                        rmix=r0+alpha*(rd-r0);cf=model.decode_single_slot(ro['anatomy'][:,li],rmix,va.coordinate_state[ri:ri+1,li].to(device),ti);xx=xf.clone();xx[:,li,ti]=cf
                        pc=planner(va.global_source[ri:ri+1].to(device),xx,va.coordinate_state[ri:ri+1].to(device),va.task_quality[ri:ri+1].to(device),va.task_valid[ri:ri+1].to(device))['slot_probabilities']
                        ps=direction*float((pc[0,slot]-pf[0,slot]).cpu());ind=np.nan
                        if own is not None:ind=direction*float((probe_probability(own,cf)-probe_probability(own,fact)).cpu())
                        cross=[]
                        for ot in range(3):
                            if ot==ti:continue
                            st=probes.get(f'level{li}_source{ti}_target{ot}')
                            if st is not None:cross.append(float((probe_probability(st,cf)-probe_probability(st,fact)).abs().cpu()))
                        t=torch.tensor([ti],device=device);af=anatomy_probe(fact,t);ac=anatomy_probe(cf,t);anat=float((1-F.cosine_similarity(af['segment_prediction'],ac['segment_prediction'],dim=-1)).cpu())
                        traj.append((alpha,ps,ind,float(np.mean(cross)) if cross else np.nan,anat))
                    trajectory[(ri,li,ti,di)]=traj
                    if len(trajectory)>=max_pairs:break
                if len(trajectory)>=max_pairs:break
    if not trajectory:return {'pairs':0}
    flat=[x for trj in trajectory.values() for x in trj];end=[trj[-1] for trj in trajectory.values()]
    mono_p=[];mono_i=[]
    for trj in trajectory.values():
        p=[x[1] for x in trj];mono_p.append(all(p[j+1]>=p[j]-1e-6 for j in range(len(p)-1)))
        iv=[x[2] for x in trj]
        if all(np.isfinite(iv)):mono_i.append(all(iv[j+1]>=iv[j]-1e-6 for j in range(len(iv)-1)))
    ratios=[]
    for _,_,ind,cross,_ in flat:
        if np.isfinite(ind) and np.isfinite(cross) and ind>0.02:ratios.append(cross/max(ind,1e-4))
    return {'pairs':len(trajectory),'endpoint_directional_accuracy':float(np.mean([x[1]>0 for x in end])),
            'endpoint_independent_directional_accuracy':float(np.mean([x[2]>0 for x in end if np.isfinite(x[2])])),
            'mean_cross_task_drift':float(np.nanmean([x[3] for x in flat])),'mean_anatomy_drift':float(np.nanmean([x[4] for x in flat])),
            'planner_monotonic_fraction':float(np.mean(mono_p)),'independent_monotonic_fraction':None if not mono_i else float(np.mean(mono_i)),
            'mean_cross_to_target_ratio':None if not ratios else float(np.mean(ratios))}


def main():
    a=parse_args();out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()) and not a.overwrite:raise RuntimeError(f'refusing non-empty output_dir: {out}')
    out.mkdir(parents=True,exist_ok=True);seed_all(a.seed);device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu');alphas=parse_alphas(a.cf_alphas)
    tr=APREBDataset(a.train_dataset);va=APREBDataset(a.val_dataset)
    model,init_obj=load_init_ar_checkpoint(a.init_ar_checkpoint);model.to(device);teacher,_=load_ar_checkpoint(a.factual_teacher_ar_checkpoint);teacher.to(device).eval()
    for p in teacher.parameters():p.requires_grad=False
    frozen=freeze_anatomy_pathway(model);anatomy_probe,_=load_anatomy_probe(a.anatomy_probe_checkpoint);anatomy_probe.to(device).eval();planner,_=load_frozen_planner(a.planner_checkpoint);planner.to(device).eval()
    probes=prepare_probe_bundle(safe_torch_load(a.cross_task_probe_bundle),device)
    train_cache=build_coordinate_candidate_cache(tr,a.donor_topk,a.quality_weight);val_cache=build_external_coordinate_candidate_cache(va,tr,a.donor_topk,a.quality_weight)
    (out/'donor_cache_audit.json').write_text(json.dumps({'train_candidate_keys':len(train_cache),'val_candidate_keys':len(val_cache),'topk':a.donor_topk,'quality_weight':a.quality_weight,'contract':'same level + opposite target label + train donor; matching uses normalized C and Q only'},indent=2),encoding='utf-8')
    preport=parameter_report(model);(out/'parameter_freeze_audit.json').write_text(json.dumps({'frozen_modules':frozen,**preport},indent=2),encoding='utf-8')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=a.weight_decay);pw=pos_weight(tr).to(device)
    loader=DataLoader(TensorDataset(torch.arange(len(tr))),batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers);rng=random.Random(a.seed+991)
    best=-1e9;best_state=None;best_row=None;history=[]

    for epoch in range(1,a.epochs+1):
        model.train();enforce_frozen_eval(model,frozen);ramp=cf_ramp(epoch,a.cf_warmup_epochs,a.cf_ramp_epochs)
        sums={k:0.0 for k in ['loss','factual','evidence','pathology','teacher_evidence','teacher_planner','original_planner','cf_total','cf_planner_target','cf_probe_target','cf_endpoint_bce','cf_non_target','cf_path_ratio','cf_anatomy_segment','cf_anatomy_coordinate','cf_anatomy_guard','cf_monotonic_planner','cf_monotonic_probe']};steps=0;cf_pairs=0
        for (idx_t,) in loader:
            idx=idx_t.long();b=batch_from_indices(tr,idx,device);o=model(b['segment_base'],b['task_features'],b['coordinate_state'])
            with torch.no_grad():to=teacher(b['segment_base'],b['task_features'],b['coordinate_state'])
            evidence=F.smooth_l1_loss(o['reconstructed_task_features'],b['task_features'])+0.25*cosine_loss(o['reconstructed_task_features'],b['task_features']);pathology=masked_bce(o['pathology_logits'],b['labels'],b['label_valid'],pw);compact=o['residual'].square().mean()
            teacher_evidence=F.smooth_l1_loss(o['reconstructed_task_features'],to['reconstructed_task_features'])+0.25*cosine_loss(o['reconstructed_task_features'],to['reconstructed_task_features'])
            with torch.no_grad():
                p_teacher=planner(b['global_source'],to['reconstructed_task_features'],b['coordinate_state'],b['task_quality'],b['task_valid'])['slot_probabilities'];p_orig=planner(b['global_source'],b['task_features'],b['coordinate_state'],b['task_quality'],b['task_valid'])['slot_probabilities']
            p_fact=planner(b['global_source'],o['reconstructed_task_features'],b['coordinate_state'],b['task_quality'],b['task_valid'])['slot_probabilities']
            teacher_planner=F.smooth_l1_loss(p_fact,p_teacher);original_planner=F.smooth_l1_loss(p_fact,p_orig)
            factual=a.w_evidence*evidence+a.w_pathology*pathology+a.w_residual_compact*compact+a.w_teacher_evidence*teacher_evidence+a.w_teacher_planner*teacher_planner+a.w_original_planner*original_planner

            z=lambda:torch.zeros((),device=device)
            cf_planner_target=z();cf_probe_target=z();cf_endpoint_bce=z();cf_non_target=z();cf_path_ratio=z();cf_anatomy_segment=z();cf_anatomy_coordinate=z();cf_anatomy_guard=z();cf_monotonic_planner=z();cf_monotonic_probe=z();cf_total=z()
            pairs=sample_counterfactual_targets(idx.tolist(),tr,train_cache,rng,a.cf_slots_per_case)
            if ramp>0 and pairs:
                cf_pairs+=len(pairs);donor_indices=sorted(set(p[4] for p in pairs));dmap={di:j for j,di in enumerate(donor_indices)};db=batch_from_indices(tr,donor_indices,device);do=model(db['segment_base'],db['task_features'],db['coordinate_state'])
                bi_list=[p[0] for p in pairs];li_list=[p[2] for p in pairs];ti_list=[p[3] for p in pairs];donor_y=torch.tensor([p[5] for p in pairs],device=device,dtype=torch.float32);direction=donor_y*2-1
                fact_target=torch.cat([o['reconstructed_task_features'][bi:bi+1,li,ti] for bi,li,ti in zip(bi_list,li_list,ti_list)],0)
                r_rec=torch.cat([o['residual'][bi:bi+1,li,ti] for bi,li,ti in zip(bi_list,li_list,ti_list)],0);r_don=torch.cat([do['residual'][dmap[p[4]]:dmap[p[4]]+1,p[2],p[3]] for p in pairs],0)
                anatomy=torch.cat([o['anatomy'][bi:bi+1,li] for bi,li in zip(bi_list,li_list)],0);coord=torch.stack([b['coordinate_state'][bi,li] for bi,li in zip(bi_list,li_list)],0);tidx=torch.tensor(ti_list,device=device,dtype=torch.long)
                slots=torch.tensor([1+ti*5+li for li,ti in zip(li_list,ti_list)],device=device);bis=torch.tensor(bi_list,device=device)
                p0=p_fact[bis,slots]
                af=anatomy_probe(fact_target,tidx);seg_gt=torch.stack([b['segment_base'][bi,li] for bi,li in zip(bi_list,li_list)],0);coord_gt=coord
                prev_plan=None;prev_probe=None;prev_probe_mask=None;target_terms=[];probe_terms=[];endpoint_terms=[];non_terms=[];ratio_terms=[];aseg_terms=[];acoord_terms=[];aguard_terms=[];mono_p_terms=[];mono_i_terms=[]
                for alpha in alphas:
                    rmix=r_rec+float(alpha)*(r_don-r_rec)
                    # grouped decode because task differs across sampled pairs
                    cf_parts=[]
                    for j,(li,ti) in enumerate(zip(li_list,ti_list)):
                        cf_parts.append(model.decode_single_slot(anatomy[j:j+1],rmix[j:j+1],coord[j:j+1],ti))
                    cf_target=torch.cat(cf_parts,0);x_cf=o['reconstructed_task_features'].clone()
                    for j,(bi,li,ti) in enumerate(zip(bi_list,li_list,ti_list)):x_cf[bi,li,ti]=cf_target[j]
                    pc=planner(b['global_source'],x_cf,b['coordinate_state'],b['task_quality'],b['task_valid']);pt=pc['slot_probabilities'][bis,slots];signed_p=direction*(pt-p0.detach())
                    target_terms.append(F.relu(float(alpha)*a.planner_direction_margin-signed_p).mean())
                    if abs(alpha-1.0)<1e-8:endpoint_terms.append(F.binary_cross_entropy_with_logits(pc['slot_logits'][bis,slots],donor_y))
                    if prev_plan is not None:mono_p_terms.append(F.relu(prev_plan-signed_p+a.monotonic_margin).mean())
                    prev_plan=signed_p

                    own_signed=[];own_mask=[];non_pair=[]
                    for j,(li,ti) in enumerate(zip(li_list,ti_list)):
                        own=probes.get(f'level{li}_source{ti}_target{ti}')
                        if own is None:own_signed.append(torch.zeros((),device=device));own_mask.append(False)
                        else:
                            qf=probe_probability(own,fact_target[j:j+1]).detach();qc=probe_probability(own,cf_target[j:j+1]);own_signed.append(direction[j]*(qc-qf).squeeze(0));own_mask.append(True)
                        dr=[]
                        for ot in range(3):
                            if ot==ti:continue
                            st=probes.get(f'level{li}_source{ti}_target{ot}')
                            if st is not None:
                                pf=probe_probability(st,fact_target[j:j+1]).detach();pp=probe_probability(st,cf_target[j:j+1]);dr.append((pp-pf).abs().squeeze(0))
                        non_pair.append(torch.stack(dr).mean() if dr else torch.zeros((),device=device))
                    own_signed=torch.stack(own_signed);mask=torch.tensor(own_mask,device=device,dtype=torch.bool);non_pair=torch.stack(non_pair)
                    if mask.any():
                        probe_terms.append(F.relu(float(alpha)*a.probe_direction_margin-own_signed[mask]).mean())
                        if abs(alpha-1.0)<1e-8:
                            logs=[];ys=[]
                            for j,(li,ti) in enumerate(zip(li_list,ti_list)):
                                own=probes.get(f'level{li}_source{ti}_target{ti}')
                                if own is not None:logs.append(probe_logit(own,cf_target[j:j+1]).squeeze());ys.append(donor_y[j])
                            if logs:endpoint_terms.append(F.binary_cross_entropy_with_logits(torch.stack(logs).view(-1),torch.stack(ys).view(-1)))
                        if prev_probe is not None and prev_probe_mask is not None:
                            mm=mask & prev_probe_mask
                            if mm.any():mono_i_terms.append(F.relu(prev_probe[mm]-own_signed[mm]+a.monotonic_margin).mean())
                        prev_probe=own_signed;prev_probe_mask=mask
                        target_budget=torch.where(mask,own_signed.detach().clamp_min(0),signed_p.detach().clamp_min(0))
                    else:
                        target_budget=signed_p.detach().clamp_min(0)
                    wlow=1.0+a.low_strength_boost*(1.0-float(alpha));non_terms.append(wlow*non_pair.mean())
                    ratio_terms.append(wlow*F.relu(non_pair-(a.path_ratio_eps+a.path_ratio_rho*target_budget)).mean())

                    ac=anatomy_probe(cf_target,tidx);aseg=(1-F.cosine_similarity(ac['segment_prediction'],af['segment_prediction'].detach(),dim=-1));acoord=(ac['coordinate_prediction']-af['coordinate_prediction'].detach()).square().mean(-1)
                    aseg_terms.append(wlow*aseg.mean());acoord_terms.append(wlow*acoord.mean())
                    seg_fact_err=1-F.cosine_similarity(af['segment_prediction'].detach(),seg_gt,dim=-1);seg_cf_err=1-F.cosine_similarity(ac['segment_prediction'],seg_gt,dim=-1)
                    coord_fact_err=(af['coordinate_prediction'].detach()-coord_gt).square().mean(-1);coord_cf_err=(ac['coordinate_prediction']-coord_gt).square().mean(-1)
                    aguard_terms.append(wlow*(F.relu(seg_cf_err-seg_fact_err).mean()+0.25*F.relu(coord_cf_err-coord_fact_err).mean()))
                avg=lambda xs:torch.stack(xs).mean() if xs else z()
                cf_planner_target=avg(target_terms);cf_probe_target=avg(probe_terms);cf_endpoint_bce=avg(endpoint_terms);cf_non_target=avg(non_terms);cf_path_ratio=avg(ratio_terms);cf_anatomy_segment=avg(aseg_terms);cf_anatomy_coordinate=avg(acoord_terms);cf_anatomy_guard=avg(aguard_terms);cf_monotonic_planner=avg(mono_p_terms);cf_monotonic_probe=avg(mono_i_terms)
                cf_total=(a.w_cf_planner_target*cf_planner_target+a.w_cf_probe_target*cf_probe_target+a.w_cf_endpoint_bce*cf_endpoint_bce+a.w_cf_non_target*cf_non_target+a.w_cf_path_ratio*cf_path_ratio+
                          a.w_cf_anatomy_segment*cf_anatomy_segment+a.w_cf_anatomy_coordinate*cf_anatomy_coordinate+a.w_cf_anatomy_guard*cf_anatomy_guard+a.w_cf_monotonic_planner*cf_monotonic_planner+a.w_cf_monotonic_probe*cf_monotonic_probe)
            loss=factual+ramp*cf_total;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();steps+=1
            vals={'loss':loss,'factual':factual,'evidence':evidence,'pathology':pathology,'teacher_evidence':teacher_evidence,'teacher_planner':teacher_planner,'original_planner':original_planner,'cf_total':cf_total,'cf_planner_target':cf_planner_target,'cf_probe_target':cf_probe_target,'cf_endpoint_bce':cf_endpoint_bce,'cf_non_target':cf_non_target,'cf_path_ratio':cf_path_ratio,'cf_anatomy_segment':cf_anatomy_segment,'cf_anatomy_coordinate':cf_anatomy_coordinate,'cf_anatomy_guard':cf_anatomy_guard,'cf_monotonic_planner':cf_monotonic_planner,'cf_monotonic_probe':cf_monotonic_probe}
            for k,v in vals.items():sums[k]+=float(v.detach().cpu())

        fv=factual_eval(model,teacher,planner,va,device,max(32,a.batch_size),a.min_positive,a.min_negative);qv=quick_multistrength_eval(model,tr,va,planner,anatomy_probe,probes,val_cache,device,alphas)
        recon=float(fv.get('evidence_reconstruction_cosine') or 0);drift=float(fv.get('planner_probability_drift_vs_original_X') or 1);floor_pen=2.0*max(0,a.factual_cosine_floor-recon);drift_pen=2.0*max(0,drift-a.factual_planner_drift_target)
        score=float(fv.get('eligible_macro_auc') or 0)+0.35*recon+0.20*float(qv.get('endpoint_directional_accuracy') or 0)+0.15*float(qv.get('endpoint_independent_directional_accuracy') or 0)+0.10*float(qv.get('planner_monotonic_fraction') or 0)-0.10*float(qv.get('mean_cross_task_drift') or 0)-0.10*float(qv.get('mean_anatomy_drift') or 0)-0.03*float(qv.get('mean_cross_to_target_ratio') or 0)-floor_pen-drift_pen
        row={'epoch':epoch,'cf_ramp':ramp,'cf_pairs':cf_pairs,'cf_alphas':alphas,**{f'train_{k}':sums[k]/max(1,steps) for k in sums},'val_factual':fv,'val_cf_quick':qv,'selection_score':score};history.append(row);print(json.dumps(row,ensure_ascii=False))
        if score>best:best=score;best_row=row;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if best_state is None:raise RuntimeError('no best state captured')
    model.load_state_dict(best_state,strict=True);final_f=factual_eval(model,teacher,planner,va,device,max(32,a.batch_size),a.min_positive,a.min_negative);final_q=quick_multistrength_eval(model,tr,va,planner,anatomy_probe,probes,val_cache,device,alphas,max_pairs=1024)
    ic=init_obj['config'];cfg={**{k:ic[k] for k in ['segment_dim','task_dim','coordinate_dim','global_dim','anatomy_dim','residual_dim','task_embedding_dim','coordinate_latent_dim','hidden_dim','dropout']},
        'version':'APREB-Multi-Strength-Path-Consistent-Counterfactual-Bottleneck','training':'frozen_anatomy_plus_factual_teacher_plus_original_planner_anchor_plus_multi_strength_path_consistency',
        'seed':a.seed,'init_checkpoint':str(Path(a.init_ar_checkpoint).resolve()),'factual_teacher_checkpoint':str(Path(a.factual_teacher_ar_checkpoint).resolve()),'frozen_modules':frozen,'parameter_report':preport,'cf_alphas':alphas,
        'loss_weights':{k:getattr(a,k) for k in vars(a) if k.startswith('w_')},'path_ratio':{'rho':a.path_ratio_rho,'eps':a.path_ratio_eps,'low_strength_boost':a.low_strength_boost}}
    ckpt=save_ar_checkpoint(out,model,cfg,{'best_selection_score':best,'best_epoch':best_row['epoch'],'val_factual':final_f,'val_cf_quick':final_q,'history':history})
    summary={'status':'finished','checkpoint':str(ckpt.resolve()),'config':cfg,'best_selection_score':best,'best_epoch':best_row['epoch'],'val_factual':final_f,'val_cf_quick':final_q,
             'claim_scope':'A remains frozen while R is trained along a continuous multi-strength model-internal intervention trajectory. This does not estimate population disease causal effects.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()


