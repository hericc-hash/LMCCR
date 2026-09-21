from __future__ import annotations
import os,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,dump_json,read_jsonl,write_jsonl
from lumbar_cf_report.selection.features import extract,extract_case,language_proxy_from_case_features
from lumbar_cf_report.selection.text_embedder import load_embedding_cache,cache_to_case_embeddings
from lumbar_cf_report.selection.reranker import load_bundle,predict_fact,predict_text_listwise,choose
from lumbar_cf_report.selection.eval_bridge import load_parser
from lumbar_cf_report.selection.metrics import eval_texts,parsed_vec

def _norm_score(x):
    x=np.asarray(x,float);lo=float(x.min());hi=float(x.max());return (x-lo)/(hi-lo+1e-8) if len(x)>1 else np.zeros_like(x)

def core_fid(cand,case):
    p=np.asarray(cand['parser_vec'],int);c=np.asarray(case['core_binary'],int);v=np.asarray(case['slot_valid'],int)>0
    tp=int(((p==1)&(c==1)&v).sum());fp=int(((p==1)&(c==0)&v).sum());gp=int(((c==1)&v).sum());d=tp+fp+gp
    return 2*tp/d if d else 0.

def select_cases(cases,Ecase,fact_b,text_b,prof,device):
    out=[];fact_expected=fact_b[0].get('feature_names',[]);text_expected=text_b[0].get('feature_names',[])
    for case,ec in zip(cases,Ecase):
        Xf=[]
        for cand in case['candidates']:
            x,n=extract(case,cand)
            if fact_expected and n!=fact_expected:raise RuntimeError('Holdout factual feature schema differs from training')
            Xf.append(x)
        Xt,nt=extract_case(case)
        if text_expected and nt!=text_expected:raise RuntimeError('Holdout text-tabular feature schema differs from training')
        fp=np.mean([predict_fact(b,np.asarray(Xf),device) for b in fact_b],axis=0);ts=np.mean([_norm_score(predict_text_listwise(b,ec,Xt,device)) for b in text_b],axis=0);px=language_proxy_from_case_features(Xt,nt);j,scores,fact_scores,elig=choose(fp,[core_fid(x,case) for x in case['candidates']],ts,px,prof);x=case['candidates'][j]
        out.append({'serial':str(case['serial']),'selected_tag':x['tag'],'generated':x['text'],'candidate_parser_vec':x['parser_vec'],'predicted_fact_heads':{'fact_f1':float(fp[j,0]),'slot_acc':float(fp[j,1])},'predicted_text_listwise_score':float(ts[j]),'language_proxy':float(px[j]),'selected_score':float(scores[j]),'candidate_fact_scores':[float(v) for v in fact_scores],'factual_safe_candidate_indices':elig})
        print(f'[R3.2-S4-v2.2 holdout] serial={case["serial"]} -> {x["tag"]} safe={len(elig)}/{len(case["candidates"])}',flush=True)
    return out

def eval_join(selected,labels,pool,mod):
    lm={str(r['serial']):r for r in labels};cm={str(r['serial']):r['core_binary'] for r in pool};rows=[{**p,**lm[str(p['serial'])]} for p in selected];m=eval_texts(mod,[r['generated'] for r in rows],[r['reference_raw'] for r in rows],[r['slot_labels'] for r in rows],[r['slot_valid'] for r in rows],[cm[str(r['serial'])] for r in rows]);pr=sum(1 for r in rows if len(parsed_vec(mod,r['generated']))==16)/len(rows);return rows,m,pr

def main():
    c=load_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));o=Path(os.environ['OUT']);tr=load_json(o/'04_reranker/TRAINING_COMPLETE.json');prof=load_json(o/'04_reranker/SELECTED_PROFILE.json');fact_b=[load_bundle(p) for p in tr['fact_models']];text_b=[load_bundle(p) for p in tr['text_listwise_models']];pool=read_jsonl(o/'08_holdout_pool/HOLDOUT50_S4_DEPLOYABLE_POOL.jsonl');labels=read_jsonl(o/'08_holdout_pool/HOLDOUT50_EVAL_LABELS.jsonl');cache=load_embedding_cache(o/'08_holdout_pool/HOLDOUT50_TEXT_EMBEDDINGS.npz');Ecase=cache_to_case_embeddings(pool,cache,True)
    sel=select_cases(pool,Ecase,fact_b,text_b,prof,c['reranker']['device']);write_jsonl(sel,o/'09_holdout/HOLDOUT_SELECTED_PREDICTIONS.jsonl');mod=load_parser(c['parser']['module']);hrows,hm,hpr=eval_join(sel,labels,pool,mod)
    srows=read_jsonl(o/'07_selection_gate/SELECTED_WITH_EVAL_FIELDS.jsonl');spool=read_jsonl(o/'05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl');scm={str(r['serial']):r['core_binary'] for r in spool};hcm={str(r['serial']):r['core_binary'] for r in pool};allrows=srows+hrows;cores=[scm.get(str(r['serial']),hcm.get(str(r['serial']))) for r in allrows];combined=eval_texts(mod,[r['generated'] for r in allrows],[r['reference_raw'] for r in allrows],[r['slot_labels'] for r in allrows],[r['slot_valid'] for r in allrows],cores);cpr=sum(1 for r in allrows if len(parsed_vec(mod,r['generated']))==16)/len(allrows)
    g=c['holdout_gate'];checks={'holdout_clinical':hm['clinical_f1']>=g['minimum_holdout_clinical_f1'],'holdout_rouge':hm['rouge_l']>=g['minimum_holdout_rouge_l'],'holdout_bleu':hm['bleu4']>=g['minimum_holdout_bleu4'],'combined100_clinical':combined['clinical_f1']>=g['minimum_combined100_clinical_f1'],'combined100_rouge':combined['rouge_l']>=g['minimum_combined100_rouge_l'],'combined100_bleu':combined['bleu4']>=g['minimum_combined100_bleu4'],'parse_rate':hpr>=g['minimum_parse_rate'] and cpr>=g['minimum_parse_rate']};rep={'pass':all(checks.values()),'checks':checks,'holdout50':hm,'combined_internal100':combined,'holdout_parse_rate':hpr,'combined_parse_rate':cpr,'hard_target':g,'s4_holdout_pool':load_json(o/'08_holdout_pool/POOL_SUMMARY.json'),'profile':prof};dump_json(rep,o/'09_holdout/HOLDOUT_AND_COMBINED_GATE.json');write_jsonl(hrows,o/'09_holdout/HOLDOUT_SELECTED_WITH_EVAL_FIELDS.jsonl');print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
