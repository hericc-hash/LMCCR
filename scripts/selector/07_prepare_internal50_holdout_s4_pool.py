from __future__ import annotations
import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,dump_json,read_jsonl,append_jsonl,write_jsonl
from lumbar_cf_report.selection.eval_bridge import load_parser
from lumbar_cf_report.selection.stage23_exact import load_runtime
from lumbar_cf_report.selection.internal import base_internal_subset,prepare_internal_subset
from lumbar_cf_report.selection.modeling import load_v32
from lumbar_cf_report.selection.candidates import generate_greedy,generate_samples,dedupe_pool
from lumbar_cf_report.selection.pool import add_greedy_final,enrich_candidates
from lumbar_cf_report.selection.adaptive import difficulty

def main():
    c=load_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));o=Path(os.environ['OUT'])
    if not (o/'07_selection_gate/SELECTION_PASS.flag').exists():raise RuntimeError('Selection50 did not pass; holdout50 remains untouched')
    pf=load_json(o/'00_PREFLIGHT.json');r23=load_json(c['assets']['r23_frozen_config']);serials=[str(x) for x in r23['internal50_holdout_serials']];dep=o/'08_holdout_pool/HOLDOUT50_S4_DEPLOYABLE_POOL.jsonl';lab=o/'08_holdout_pool/HOLDOUT50_EVAL_LABELS.jsonl';done={str(r['serial']) for r in read_jsonl(dep)} if dep.exists() else set();cfg23,ck23,data23,_,_=load_runtime(pf['stage23_dir']);allrows=base_internal_subset(Path(pf['stage3a_dir'])/c['stage3a']['inputs_file'],c['assets']['frozen_v5_jsonl'],serials);mod=load_parser(c['parser']['module']);rows=prepare_internal_subset(allrows,serials,cfg23,ck23,data23,c['stage23']['lordosis_threshold'],mod,c['scaffold_channel'],'cpu');lex=load_json(o/'01_development/CANONICAL_LEXICALIZER.json')
    if not lab.exists():write_jsonl([{'serial':str(r['serial']),'reference_raw':r['reference_raw'],'slot_labels':r['slot_labels'],'slot_valid':r['slot_valid']} for r in rows],lab)
    tok,model=load_v32(c['assets']['base_model'],pf['language_adapter'],pf['r31v1_adapter'],pf['scaffold_adapter'],c['device']);fresh_sequences=0
    for i,r in enumerate(rows,1):
        s=str(r['serial'])
        if s in done:continue
        g=generate_greedy(tok,model,r,c['generation']);pool,_=add_greedy_final(r,[g],lex,mod,c['precision_patch']);base=enrich_candidates(r,dedupe_pool(pool),mod,with_targets=False);meta=difficulty(r,base,c.get('adaptive_expansion',{})) if c.get('adaptive_expansion') else {'score':0.0};pool=[(x['tag'],x['text']) for x in base];pool+=generate_samples(tok,model,r,c['generation'],c['seed']+100000+int(s),count=4);fresh_sequences+=5
        cand=enrich_candidates(r,dedupe_pool(pool),mod,with_targets=False);append_jsonl({'serial':s,'core_binary':r['core_binary'],'slot_valid':r['slot_valid'],'planner_probs':r['planner_probs'],'planner_thresholds':r['planner_thresholds'],'direct_draft':r['direct_draft'],'direct_parser_vec':r['direct_parser_vec'],'linguistic_scaffold':r['linguistic_scaffold'],'adaptive_meta':meta,'expanded':True,'s4_full_coverage':True,'candidates':cand},dep);print(f'[R3.2-S4-v2.2 holdout pool] {i}/50 serial={s} n={len(cand)}',flush=True)
    allr=read_jsonl(dep)
    if len(allr)!=50 or len({str(r['serial']) for r in allr})!=50:raise RuntimeError(f'Holdout S4 pool incomplete/duplicate rows={len(allr)} unique={len({str(r["serial"]) for r in allr})}')
    forbidden=[]
    for r in allr:
        forbidden += [k for k in ['reference_raw','reference','slot_labels','GT','rouge','bleu'] if k in r]
        for x in r['candidates']:forbidden += [k for k in ['target_fact_f1','target_slot_acc','rouge','bleu','tp','fp','fn','gt_pos','avg_logprob'] if k in x]
    if forbidden:raise RuntimeError('Holdout deployable pool leakage '+str(sorted(set(forbidden))))
    ns=[len(r['candidates']) for r in allr];tc={}
    for r in allr:
        for x in r['candidates']:tc[x['tag']]=tc.get(x['tag'],0)+1
    dump_json({'status':'PASS','cases':50,'mean_candidates':sum(ns)/50,'min_candidates':min(ns),'max_candidates':max(ns),'tag_counts':tc,'fresh_generated_sequences_this_run':fresh_sequences,'holdout_access':'ONLY_AFTER_SELECTION_PASS','protocol':'Full S4: greedy/final + sample0..sample3 for all holdout cases; no beam/logprob'},o/'08_holdout_pool/POOL_SUMMARY.json')
if __name__=='__main__':main()
