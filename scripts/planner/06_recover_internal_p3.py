#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,traceback,torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import safe_load,dumpj
from lumbar_cf_report.planner.p3_replay import normalize_pair_split,pair_primary_mask,subset_pairs,scan_historical_artifacts,candidate_deltas_from_artifacts,_schema,load_frozen_e0
from lumbar_cf_report.planner.original_p3_replay import replay_original_internal

def _s(v):
    if torch.is_tensor(v) and v.numel()==1:v=v.item()
    if isinstance(v,np.generic):v=v.item()
    return str(v)

def _list(v):
    if torch.is_tensor(v):return v.detach().cpu().reshape(-1).tolist()
    if isinstance(v,np.ndarray):return v.reshape(-1).tolist()
    return list(v)

def col(df,*names):
    low={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in low:return low[n.lower()]
    for c in df.columns:
        if any(n.lower() in str(c).lower() for n in names):return c
    return None

def derive_sign_from_bank(cfg,pairs):
    """Derive transition sign from actual recipient/donor labels when possible."""
    b=safe_load(cfg['feature_bank']);sk=next((k for k in ('serials','source_serial','case_ids') if k in b),None);yk=next((k for k in ('y','labels') if k in b),None)
    if sk is None or yk is None:return None,'FEATURE_BANK_MISSING_SERIALS_OR_LABELS'
    ss=[_s(v) for v in b[sk]];mp={s:i for i,s in enumerate(ss)};y=torch.as_tensor(b[yk]).float()
    if y.ndim!=3 or tuple(y.shape[1:])!=(5,3):return None,f'UNEXPECTED_LABEL_SHAPE_{tuple(y.shape)}'
    if pairs.get('donor') is None:return None,'PAIR_BANK_HAS_NO_DONOR'
    signs=[];bad=[]
    for i,(r,don,l,t) in enumerate(zip(pairs['recipient'],pairs['donor'],torch.as_tensor(pairs['level']).tolist(),torch.as_tensor(pairs['task']).tolist())):
        r=_s(r);don=_s(don)
        if r not in mp or don not in mp:bad.append((i,r,don,'serial_missing'));signs.append(0.);continue
        a=float(y[mp[r],int(l),int(t)]);bval=float(y[mp[don],int(l),int(t)]);sg=np.sign(bval-a)
        if abs(bval-a)<0.5:bad.append((i,r,don,f'labels_same_{a}_{bval}'))
        signs.append(float(sg))
    if bad:return None,{'reason':'RECIPIENT_DONOR_LABEL_SIGN_NOT_UNIQUE','examples':bad[:20],'bad_count':len(bad)}
    return torch.tensor(signs,dtype=torch.float32),'DERIVED_FROM_RECIPIENT_DONOR_LABEL_DIFFERENCE'

def scales_and_sign_from_csv(cfg,pairs,sign_fallback):
    rep={'status':'NOT_USED'}
    try:
        import pandas as pd
        f=Path(cfg['p3_metrics_csv'])
        if not f.exists():return None,sign_fallback,{'status':'CSV_NOT_FOUND','path':str(f)}
        df=pd.read_csv(f);rep={'status':'READ','path':str(f),'columns':list(df.columns)}
        sc=col(df,'split','cohort');rc=col(df,'recipient_serial','recipient','recipient_id');lc=col(df,'level_index','level');tc=col(df,'task_index','task');kc=col(df,'local_scale','selected_scale','step_scale');sigc=col(df,'orientation_sign','minority_sign','target_sign','sign');sel=col(df,'is_selected','selected','chosen','winner')
        if None in (rc,lc,tc,kc):return None,sign_fallback,{**rep,'status':'REQUIRED_COLUMNS_MISSING','identified':{'split':sc,'recipient':rc,'level':lc,'task':tc,'scale':kc,'sign':sigc,'selected':sel}}
        z=df.copy()
        if sc is not None:z=z[z[sc].astype(str).str.lower().str.contains('internal')]
        for names,val in [(('p3_lambda','lambda'),cfg['p3_fixed_contract']['lambda']),(('specificity_alpha','alpha'),cfg['p3_fixed_contract']['alpha']),(('beta',),cfg['p3_fixed_contract']['beta'])]:
            cc=col(z,*names)
            if cc is not None:z=z[np.isclose(pd.to_numeric(z[cc],errors='coerce'),float(val),atol=1e-9)]
        if sel is not None:
            ss=z[sel];num=pd.to_numeric(ss,errors='coerce');mask=ss.astype(str).str.lower().isin(['1','true','yes','selected','winner']) | (num==1)
            if mask.any():z=z[mask]
        # Use only groups with a unique scale/sign under frozen contract.
        scale_mp={};sign_mp={}
        for key,g in z.groupby([rc,lc,tc],dropna=False):
            vals=pd.to_numeric(g[kc],errors='coerce').dropna().unique()
            if len(vals)==1:scale_mp[(str(key[0]),int(key[1]),int(key[2]))]=float(vals[0])
            if sigc is not None:
                sv=pd.to_numeric(g[sigc],errors='coerce').dropna().unique()
                if len(sv)==1:sign_mp[(str(key[0]),int(key[1]),int(key[2]))]=float(sv[0])
        scales=[];csv_sign=[];missing=[]
        for r,l,t in zip(pairs['recipient'],torch.as_tensor(pairs['level']).tolist(),torch.as_tensor(pairs['task']).tolist()):
            key=(_s(r),int(l),int(t))
            if key not in scale_mp:missing.append(key);scales.append(float('nan'))
            else:scales.append(scale_mp[key])
            csv_sign.append(sign_mp.get(key,float('nan')))
        if missing:return None,sign_fallback,{**rep,'status':'SCALE_MATCH_INCOMPLETE','matched':len(scales)-len(missing),'expected':len(scales),'missing_examples':missing[:20]}
        scale=torch.tensor(scales,dtype=torch.float32)
        cs=torch.tensor(csv_sign,dtype=torch.float32)
        if torch.isfinite(cs).all():sign=cs;sign_source='P3_METRICS_CSV'
        else:sign=sign_fallback;sign_source='FALLBACK_LABEL_DERIVED' if sign_fallback is not None else 'UNRESOLVED'
        return scale,sign,{**rep,'status':'PASS','matched':len(scales),'sign_source':sign_source,'identified':{'split':sc,'recipient':rc,'level':lc,'task':tc,'scale':kc,'sign':sigc,'selected':sel}}
    except Exception as e:return None,sign_fallback,{**rep,'status':'ERROR','error':repr(e)}

def formula_suffix(name):
    return str(name).split('::')[-1]

def main_impl(cfg,out):
    devp=safe_load(out/'02_exact_p3_development.pt');locked=devp['source_formula'];locked_suffix=formula_suffix(locked);exact_specs=devp.get('exact_replay_candidates',[]);locked_suffixes=[]
    for z in exact_specs:
        sf=z.get('suffix') if isinstance(z,dict) else formula_suffix(z)
        if sf and sf not in locked_suffixes:locked_suffixes.append(sf)
    if locked_suffix not in locked_suffixes:locked_suffixes.insert(0,locked_suffix)
    pair_bank=safe_load(cfg['pair_bank']);dumpj({'pair_bank_schema':_schema(pair_bank)},out/'06_INTERNAL_PAIR_BANK_SCHEMA.json')
    allpairs=normalize_pair_split(pair_bank,'internal');pm=pair_primary_mask(allpairs);pairs=subset_pairs(allpairs,pm)
    rep={'status':'UNRESOLVED_INTERNAL_P3','source_formula_locked_from_teacher_validated_development':locked,'locked_formula_suffix':locked_suffix,'development_teacher_exact_formula_suffixes':locked_suffixes,'pair_rows_total':allpairs['rows'],'primary_pair_rows':pairs['rows'],'transition_shape_contract':'[rows,3,128] full level-wise P3 response'}
    if pairs['rows']<=0:raise RuntimeError('no internal primary P3 pairs')
    sign,label_sign_rep=derive_sign_from_bank(cfg,pairs);rep['sign_derivation']=label_sign_rep
    # v1.4: local_scale comes from the frozen INTERNAL pair-bank rows themselves.
    # The Development P3 metrics CSV is not an Internal metadata source and is only a fallback audit path.
    scale_source=None;scale=None;csvrep={'status':'NOT_NEEDED_PAIR_BANK_PRIMARY'}
    bank_scale=pairs.get('local_scale')
    if bank_scale is not None:
        try:
            z=torch.as_tensor(bank_scale).float().reshape(-1)
            if len(z)==pairs['rows'] and torch.isfinite(z).all() and bool((z>0).all()):
                scale=z;scale_source='FROZEN_PAIR_BANK_INTERNAL_PRIMARY_ROWS'
        except Exception:
            scale=None
    if scale is None:
        scale,sign,csvrep=scales_and_sign_from_csv(cfg,pairs,sign);scale_source='P3_METRICS_CSV_FALLBACK' if scale is not None else None
    rep['p3_metrics_csv']=csvrep;rep['local_scale_source']=scale_source
    if sign is None:raise RuntimeError('internal P3 sign could not be resolved from recipient/donor labels or fallback metadata')
    need_scale=any(sf in ('beta_local_scale_normalized_direction','beta_local_scale','sign_beta_local_scale','local_scale','sign_local_scale') for sf in locked_suffixes)
    if need_scale and scale is None:raise RuntimeError(f'internal local_scale unresolved from frozen pair bank and fallback metadata, but locked formula requires it: {locked_suffix}')
    if scale is None:scale=torch.ones(pairs['rows'],dtype=torch.float32);scale_source='UNIT_SCALE_FORMULA_DOES_NOT_REQUIRE_LOCAL_SCALE'
    rep['local_scale_summary']={'source':scale_source,'n':int(len(scale)),'min':float(scale.min()),'median':float(scale.median()),'max':float(scale.max())}

    meta={'recipient_serial':pairs['recipient'],'level':pairs['level'],'task':pairs['task'],'sign':sign,'local_scale':scale,'beta':cfg['p3_fixed_contract']['beta'],'p3_lambda':cfg['p3_fixed_contract']['lambda']}
    # Internal recovery uses the same exact frozen E0 anchor; no Stage1.20 forward call.
    E0,e0_serials=load_frozen_e0(cfg['frozen_old_factual_evidence'])
    artifacts,inventory=scan_historical_artifacts(cfg,pairs['rows']);dumpj({'rows_expected':pairs['rows'],'artifacts':inventory},out/'06_INTERNAL_HISTORICAL_P3_ARTIFACT_INVENTORY.json')
    cands,prov=candidate_deltas_from_artifacts(artifacts,meta,E0,e0_serials,pairs['rows'],cfg)
    eligible=[]
    for name,delta in cands.items():
        if formula_suffix(name) not in locked_suffixes:continue
        if torch.as_tensor(delta).shape!=(pairs['rows'],3,128):continue
        if prov.get(name,{}).get('alignment')=='ROW_META_PRESENT_BUT_NONUNIQUE_ALIGNMENT_FAILED':continue
        eligible.append((name,torch.as_tensor(delta).float(),prov.get(name,{})))
    rep['eligible_candidates']=[{'name':n,'shape':list(x.shape),'provenance':p} for n,x,p in eligible]
    if not eligible:
        # No frozen Internal1179 tensor was found. Replay the original Stage1.21-P3
        # implementation on its original X/E/C + protected Jacobian. This is a
        # deterministic frozen-method replay, not a new Internal hyperparameter search.
        try:
            delta,meta_src,source_rep=replay_original_internal(cfg,cfg.get('device','cuda'))
            if tuple(delta.shape)!=(pairs['rows'],3,128):
                raise RuntimeError(f'original-source replay returned {tuple(delta.shape)}')
            # Verify row identity against the normalized pair bank before accepting.
            src_r=[_s(x) for x in meta_src['recipient_serial']];dst_r=[_s(x) for x in pairs['recipient']]
            src_l=torch.as_tensor(meta_src['level']).long();dst_l=torch.as_tensor(pairs['level']).long()
            src_t=torch.as_tensor(meta_src['task']).long();dst_t=torch.as_tensor(pairs['task']).long()
            if src_r!=dst_r or not torch.equal(src_l,dst_l) or not torch.equal(src_t,dst_t):
                raise RuntimeError('original-source replay row order does not exactly match frozen internal pair bank')
            meta=meta_src
            name='original_stage1_21_source_replay::normalized_3seed_global_lambda::beta_local_scale_normalized_direction'
            provenance={**source_rep,'alignment':'EXACT_ORIGINAL_PAIR_ORDER'}
            torch.save({'status':'RESOLVED_BY_ORIGINAL_STAGE1_21_SOURCE_REPLAY','delta':delta.cpu(),'meta':meta,'source_formula':name,'locked_development_formula':locked,'source_provenance':provenance,'transition_shape_contract':'[rows,3,128] full level-wise response'},out/'06_exact_p3_internal.pt')
            rep.update({'status':'RESOLVED_BY_ORIGINAL_STAGE1_21_SOURCE_REPLAY','selected_candidate':name,'selected_provenance':provenance,'shape':list(delta.shape)})
            dumpj(rep,out/'06_INTERNAL_P3_RECOVERY.json');print(json.dumps(rep,indent=2,default=str));return True
        except Exception as source_err:
            rep['original_source_replay_failure']={'error':repr(source_err),'traceback':traceback.format_exc()}
            dumpj(rep,out/'06_INTERNAL_P3_RECOVERY.json')
            raise RuntimeError(f'no frozen Internal P3 asset and original-source replay failed: {source_err}') from source_err
    # Frozen duplicate caches should agree. If they do not, fail closed rather than silently pick one.
    reference=eligible[0][1];agreement=[]
    for n,x,p in eligible:
        agreement.append({'name':n,'max_abs_vs_first':float((x-reference).abs().max()),'mean_abs_vs_first':float((x-reference).abs().mean())})
    rep['candidate_agreement']=agreement
    consistent=[z for z in agreement if z['max_abs_vs_first']<=1e-5]
    if len(eligible)>1 and len(consistent)!=len(eligible):
        # Prefer an explicitly row-aligned candidate only if all such aligned candidates agree.
        aligned=[z for z in eligible if z[2].get('alignment')=='ROW_ORDER_ALIGNED_BY_RECIPIENT_LEVEL_TASK']
        if not aligned:raise RuntimeError('multiple frozen internal P3 assets disagree and none has explicit row metadata alignment')
        reference=aligned[0][1]
        if any(float((x-reference).abs().max())>1e-5 for _,x,_ in aligned):raise RuntimeError('explicitly aligned internal P3 assets disagree')
        chosen=aligned[0]
    else:chosen=eligible[0]
    name,delta,provenance=chosen
    torch.save({'status':'RESOLVED_FROM_FROZEN_HISTORICAL_P3_ASSET','delta':delta.cpu(),'meta':meta,'source_formula':name,'locked_development_formula':locked,'source_provenance':provenance,'transition_shape_contract':'[rows,3,128] full level-wise response'},out/'06_exact_p3_internal.pt')
    rep.update({'status':'RESOLVED_FROM_FROZEN_HISTORICAL_P3_ASSET','selected_candidate':name,'selected_provenance':provenance,'shape':list(delta.shape)})
    dumpj(rep,out/'06_INTERNAL_P3_RECOVERY.json');print(json.dumps(rep,indent=2,default=str));return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out)
    try:ok=main_impl(cfg,out)
    except Exception as e:
        old={}
        try:old=json.load(open(out/'06_INTERNAL_P3_RECOVERY.json'))
        except Exception:pass
        rep={**old,'status':'UNRESOLVED_INTERNAL_P3','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'note':'Internal mechanism evaluation is blocked, but factual evaluation remains valid.'};dumpj(rep,out/'06_INTERNAL_P3_RECOVERY.json');print(json.dumps(rep,indent=2));ok=False
    if not ok:raise SystemExit(42)
if __name__=='__main__':main()
