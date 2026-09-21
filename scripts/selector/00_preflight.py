from __future__ import annotations
import sys,os,json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,dump_json,read_jsonl,sha256
from lumbar_cf_report.selection.discovery import latest_complete
from lumbar_cf_report.selection.eval_bridge import parser_source,load_parser,parse,normalize
from lumbar_cf_report.selection.stage23_exact import load_runtime
from lumbar_cf_report.selection.source_s4 import discover_source

ALLOWED_TAGS={'greedy','greedy_final','sample0','sample1','sample2','sample3'}

def _pool_contract(path,development):
    rows=read_jsonl(path);expected=388 if development else 50
    if len(rows)!=expected or len({str(r.get('serial')) for r in rows})!=expected:return False,{'rows':len(rows),'unique':len({str(r.get('serial')) for r in rows})}
    bad=[];mins=99;maxs=0
    for r in rows:
        cs=r.get('candidates',[]);mins=min(mins,len(cs));maxs=max(maxs,len(cs));tags={str(x.get('tag')) for x in cs}
        if not tags.issubset(ALLOWED_TAGS) or not {'greedy','sample0','sample1','sample2','sample3'}.issubset(tags):bad.append(str(r.get('serial')))
        if development:
            for x in cs:
                if not all(k in x for k in ['target_fact_f1','target_slot_acc','rouge','bleu','tp','fp','gt_pos']):bad.append('target:'+str(r.get('serial')));break
        else:
            forbidden=[]
            for k in ['reference_raw','reference','slot_labels','GT','rouge','bleu']:
                if k in r:forbidden.append(k)
            for x in cs:
                for k in ['target_fact_f1','target_slot_acc','rouge','bleu','tp','fp','fn','gt_pos','avg_logprob']:
                    if k in x:forbidden.append('cand:'+k)
            if forbidden:bad.append('leak:'+str(r.get('serial'))+':'+','.join(sorted(set(forbidden))))
    return len(bad)==0,{'rows':len(rows),'min_candidates':mins,'max_candidates':maxs,'bad':bad[:12]}

def main():
    c=load_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));out=Path(os.environ['OUT']);out.mkdir(parents=True,exist_ok=True)
    s23,_=latest_complete(c['stage23']['output_prefixes'],c['stage23']['required_files'],'STAGE23_DIR','Stage2.3-v1.7')
    v32,_=latest_complete(c['v32']['output_prefixes'],c['v32']['required_files'],'R31_V32_DIR','R3.1-v3.2')
    s3,_=latest_complete(c['stage3a']['output_prefixes'],c['stage3a']['required_files'],'STAGE3A_DIR','Stage3-A')
    source=discover_source(c['source_s4']);cfg23,ck23,data23,_,_=load_runtime(s23)
    r23=load_json(c['assets']['r23_frozen_config']);sel=[str(x) for x in r23['internal50_dev_serials']];hold=[str(x) for x in r23['internal50_holdout_serials']]
    ps=parser_source(c['parser']['module']);psha=sha256(ps);mod=load_parser(c['parser']['module']);vec=normalize(parse(mod,'所见：L4/5椎间盘膨出。\n结论：L4/5椎间盘膨出。'))
    dev_ok,dev_meta=_pool_contract(source/'03_development_candidates/DEV388_S4_CANDIDATE_POOL.jsonl',True)
    sel_ok,sel_meta=_pool_contract(source/'05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl',False)
    labels=read_jsonl(source/'05_selection_pool/SELECTION50_EVAL_LABELS.jsonl');label_ok=len(labels)==50 and {str(r['serial']) for r in labels}==set(sel)
    out_s=str(out.resolve());unsafe=any(x in out_s for x in ['stage3C_r30_','stage3C_r31_','stage3C_r32_fact_aware_','stage3C_r32_lite_','stage3C_r32_s4_expanded_candidate_reranking_realizer_v2_','stage3C_r32_s4_language_aware_reranker_v2_1_'])
    proto=c['candidate_protocol'];checks={
      'base_model':Path(c['assets']['base_model']).exists(),'stage23_runtime':True,'stage23_has_internal':'internal' in data23,
      'v32_assets':(v32/'02_training/TRAINING_COMPLETE.json').exists(),'stage3a_inputs':(s3/c['stage3a']['inputs_file']).exists(),
      'parser_sha':psha==c['parser']['expected_sha256'],'parser16':len(vec)>=16,'split50_50':len(sel)==50 and len(hold)==50 and not(set(sel)&set(hold)),'cuda':torch.cuda.is_available(),
      'i49_forbidden':c['independent49_policy']=='FORBIDDEN','source_s4_dev_pool':dev_ok,'source_s4_selection_pool':sel_ok,'source_selection_labels':label_ok,
      's4_full_coverage':bool(proto.get('full_stochastic_coverage')) and int(proto.get('stochastic_samples_per_case',0))==4,
      's4_no_beam':not bool(proto.get('beam_search',True)),'s4_no_logprob':not bool(proto.get('teacher_forced_logprob',True)),'text_encoder_no_reference':not bool(c['text_encoder'].get('use_reference',True)),'text_encoder_no_gt':not bool(c['text_encoder'].get('use_gt',True)),'text_encoder_no_generation':not bool(c['text_encoder'].get('use_generation',True)),
      'output_not_frozen_or_previous_stage':not unsafe
    }
    # Keep exact frozen adapters for text representation and holdout generation.
    tr=load_json(v32/'02_training/TRAINING_COMPLETE.json');language_adapter=tr.get('language_adapter_merged');r31v1_adapter=tr.get('r31_v1_adapter_merged');scaffold_adapter=tr.get('scaffold_adapter')
    checks['frozen_encoder_adapters']=all(x and Path(x).exists() for x in [language_adapter,r31v1_adapter,scaffold_adapter])
    rep={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'source_s4_out':str(source),'source_dev_pool_contract':dev_meta,'source_selection_pool_contract':sel_meta,'stage23_dir':str(s23),'v32_dir':str(v32),'stage3a_dir':str(s3),'stage23_thresholds':[float(x) for x in ck23['thresholds'].view(-1)],'parser_source':str(ps),'parser_sha256':psha,'selection_serials':sel,'holdout_serials':hold,'protocol':'v2.2 reuses frozen S4 candidate texts, extracts frozen-v3.2-Qwen contextual embeddings without generation, and trains only FactNet + text-level listwise selector. Development388 GT/reference are supervision only. Selection50 pool remains deployment-safe. Holdout50 locked until selection PASS. Independent49 forbidden.','language_adapter':language_adapter,'r31v1_adapter':r31v1_adapter,'scaffold_adapter':scaffold_adapter}
    dump_json(rep,out/'00_PREFLIGHT.json');print(json.dumps(rep,ensure_ascii=False,indent=2))
    if rep['status']!='PASS':raise RuntimeError('R3.2-S4-v2.2 preflight failed')
if __name__=='__main__':main()
