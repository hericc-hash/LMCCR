from __future__ import annotations
import json,random,difflib,math
from .sanitize import sanitize_reference
from .scaffold import build_scaffold,build_similarity_scaffold

def read_jsonl(p):
    out=[]
    with open(p,encoding='utf-8') as f:
        for line in f:
            if line.strip():out.append(json.loads(line))
    return out

def _bin(xs):return [int(float(x)>.5) for x in xs]

def build_train388(path):
    base={}
    for r in read_jsonl(path):
        if r.get('sample_type')!='oracle_core_sft':continue
        if 'direct_draft' not in r:raise RuntimeError('oracle row missing direct_draft')
        s=str(r['serial']);q=dict(r);q['slot_labels']=_bin(q['core_binary']);q['r31_input']='LOCKED_CORE_PLUS_CORE_REDACTED_SCAFFOLD'
        if s in base:raise RuntimeError('duplicate oracle serial '+s)
        base[s]=q
    if len(base)!=388:raise RuntimeError(f'Expected 388 oracle rows, got {len(base)}')
    return [base[s] for s in sorted(base,key=lambda x:(0,int(x)) if x.isdigit() else (1,x))]

def enrich_internal(inputs,v5_rows):
    v={str(r['serial']):r for r in v5_rows};out=[]
    for r in inputs:
        s=str(r['serial']);z=v.get(s)
        if z is None:raise RuntimeError('v5 row missing '+s)
        if not isinstance(r.get('direct_draft'),str):raise RuntimeError('Stage3-A input missing direct_draft '+s)
        q=dict(r);q['reference_raw']=z['reference_raw'];q['slot_labels']=z['slot_labels'];q['slot_valid']=z['slot_valid'];out.append(q)
    return out

def split_internal(rows,selection,holdout):
    a={str(x) for x in selection};b={str(x) for x in holdout}
    if len(a)!=50 or len(b)!=50 or a&b:raise RuntimeError('Internal split invalid')
    by={str(r['serial']):r for r in rows}
    if set(by)!=(a|b):raise RuntimeError(f'Internal serial mismatch n={len(by)}')
    return [by[str(x)] for x in selection],[by[str(x)] for x in holdout]

def _hmean(a,b):
    a=float(a);b=float(b)
    return 0.0 if a<=0 or b<=0 else 2*a*b/(a+b)

def scaffold_data_gate(summary,cfg):
    """Stable two-tier gate.

    Hard gates protect factual safety. Language preservation is judged jointly rather
    than allowing one noisy proxy (e.g. mean safe chars) to veto an otherwise valid
    388-case scaffold set.
    """
    zero=float(summary['scaffold_parser_zero_rate'])
    nonempty=float(summary['scaffold_nonempty_fraction'])
    fallback=float(summary['scaffold_hard_fallback_fraction'])
    total=float(summary['mean_scaffold_payload_chars'])
    safe=float(summary['mean_scaffold_safe_payload_chars'])
    preserved=float(summary['mean_preserved_safe_fraction'])

    safe_ref=max(float(cfg.get('language_quality_safe_payload_reference_chars',40.0)),1e-6)
    safe_norm=min(safe/safe_ref,1.0)
    joint=_hmean(safe_norm,preserved)

    hard={
        'parser_zero': zero==1.0,
        'nonempty': nonempty>=float(cfg.get('diagnostic_minimum_nonempty_fraction',0.95)),
        'fallback': fallback<=float(cfg.get('maximum_fallback_fraction',0.10)),
    }
    language={
        'total_payload_floor': total>=float(cfg.get('language_quality_minimum_total_payload_chars',60.0)),
        'safe_payload_floor': safe>=float(cfg.get('language_quality_minimum_safe_payload_chars',25.0)),
        'preserved_safe_floor': preserved>=float(cfg.get('language_quality_minimum_preserved_safe_fraction',0.35)),
        'joint_safe_language_quality': joint>=float(cfg.get('language_quality_minimum_joint_hmean',0.65)),
    }
    return {
        'pass': all(hard.values()) and all(language.values()),
        'hard_checks':hard,
        'language_checks':language,
        'metrics':{
            'mean_total_payload_chars':total,
            'mean_safe_payload_chars':safe,
            'mean_preserved_safe_fraction':preserved,
            'safe_payload_normalized':safe_norm,
            'joint_safe_language_quality_hmean':joint,
        },
        'principle':'Clinical16 safety uses hard gates; scaffold language sufficiency uses permissive floors plus a joint safe-payload/preservation score so a single proxy cannot false-reject the full388 set.'
    }

def materialize(rows,lex,parser_mod,parse_fn,normalize_fn,scaffold_cfg,copy_cfg,seed):
    base=[];met=[]
    for r in rows:
        # Input scaffold: Core-redacted Direct preserving local order/punctuation.
        scaffold,sm=build_scaffold(r['direct_draft'],parser_mod,parse_fn,normalize_fn,scaffold_cfg)
        # Natural factual target: reference surgically reconciled to Oracle Core.
        target,tm=sanitize_reference(r,lex,parser_mod,parse_fn,normalize_fn)
        # IMPORTANT v3.2 fix: compare like-with-like. Redact Clinical16 facts from the
        # target too, so safe-copy eligibility reflects linguistic scaffold overlap
        # instead of being artificially depressed by target Core phrases.
        target_scaffold,tsm=build_similarity_scaffold(target,lex,parser_mod,parse_fn,normalize_fn,scaffold_cfg)
        sim=difflib.SequenceMatcher(None,scaffold,target_scaffold).ratio()
        q=dict(r)
        q['linguistic_scaffold']=scaffold
        q['scaffold_meta']=sm
        q['target_text']=target
        q['target_sanitizer_meta']=tm
        q['target_linguistic_scaffold']=target_scaffold
        q['target_scaffold_meta']=tsm
        q['scaffold_target_similarity']=sim
        q['sample_type']='r31v32_scaffold_main'
        base.append(q);met.append((sm,tm,tsm,sim))

    out=list(base);aux=0;eligible_count=0
    if copy_cfg.get('enabled',True):
        thr=float(copy_cfg.get('minimum_scaffold_target_similarity',0.52))
        cap=int(len(base)*float(copy_cfg.get('maximum_aux_fraction',0.35)))
        reps=int(copy_cfg.get('repeat_count',1))
        eligible=sorted([q for q in base if q['scaffold_target_similarity']>=thr],key=lambda x:x['scaffold_target_similarity'],reverse=True)
        eligible_count=len(eligible)
        for q in eligible:
            for _ in range(reps):
                if aux>=cap:break
                z=dict(q);z['sample_type']='r31v32_safe_copy_aux';out.append(z);aux+=1
            if aux>=cap:break
    random.Random(seed).shuffle(out)
    stages={}
    for _,t,_,_ in met:stages[t['stage']]=stages.get(t['stage'],0)+1
    summary={
        'development388_train_cases':len(base),'materialized_rows':len(out),'safe_copy_aux_rows':aux,
        'safe_copy_eligible_rows':eligible_count,
        'target_sanitizer_stages':stages,
        'scaffold_parser_zero_rate':sum(1 for s,_,_,_ in met if s['parser_zero'])/len(met),
        'target_scaffold_parser_zero_rate':sum(1 for _,_,s,_ in met if s['parser_zero'])/len(met),
        'scaffold_nonempty_fraction':sum(1 for s,_,_,_ in met if not s['empty'])/len(met),
        'scaffold_hard_fallback_fraction':sum(1 for s,_,_,_ in met if s['hard_fallback'])/len(met),
        'mean_scaffold_payload_chars':sum(s['payload_chars'] for s,_,_,_ in met)/len(met),
        'mean_scaffold_safe_payload_chars':sum(s['safe_payload_chars'] for s,_,_,_ in met)/len(met),
        'mean_scaffold_placeholders':sum(s['placeholder_count'] for s,_,_,_ in met)/len(met),
        'mean_preserved_safe_fraction':sum(s['preserved_safe_fraction'] for s,_,_,_ in met)/len(met),
        'mean_scaffold_target_similarity':sum(x for *_,x in met)/len(met),
        'safe_copy_similarity_definition':'SequenceMatcher(strict_core_redacted_direct_scaffold, best_effort_parser_zero_target_similarity_projection)',
        'target_similarity_projection_modes':{},
    }
    modes={}
    for _,_,tsm,_ in met:
        mode=str(tsm.get('similarity_projection_mode','UNKNOWN'));modes[mode]=modes.get(mode,0)+1
    summary['target_similarity_projection_modes']=modes
    summary['target_similarity_projection_non_strict_fraction']=sum(v for k,v in modes.items() if k!='STRICT_AFTER_EXACT_LEX_REDACTION')/len(met)
    summary['data_gate']=scaffold_data_gate(summary,scaffold_cfg)
    return out,summary
