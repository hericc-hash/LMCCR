from __future__ import annotations
import os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,dump_json,read_jsonl,write_jsonl
from lumbar_cf_report.selection.features import extract,extract_case,language_proxy_from_case_features
from lumbar_cf_report.selection.text_embedder import load_embedding_cache,cache_to_case_embeddings
from lumbar_cf_report.selection.reranker import load_bundle,predict_fact,predict_text_listwise,choose

def _norm_score(x):
    x=np.asarray(x,float);lo=float(x.min());hi=float(x.max());return (x-lo)/(hi-lo+1e-8) if len(x)>1 else np.zeros_like(x)

def core_fid(cand,case):
    p=np.asarray(cand['parser_vec'],int);c=np.asarray(case['core_binary'],int);v=np.asarray(case['slot_valid'],int)>0
    tp=int(((p==1)&(c==1)&v).sum());fp=int(((p==1)&(c==0)&v).sum());gp=int(((c==1)&v).sum());d=tp+fp+gp
    return 2*tp/d if d else 0.

def main():
    c=load_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));o=Path(os.environ['OUT']);tr=load_json(o/'04_reranker/TRAINING_COMPLETE.json');prof=load_json(o/'04_reranker/SELECTED_PROFILE.json');fact_b=[load_bundle(p) for p in tr['fact_models']];text_b=[load_bundle(p) for p in tr['text_listwise_models']];cases=read_jsonl(o/'05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl');cache=load_embedding_cache(o/'02_text_embeddings/SELECTION50_TEXT_EMBEDDINGS.npz');Ecase=cache_to_case_embeddings(cases,cache,True);selected=[]
    fact_expected=fact_b[0].get('feature_names',[]);text_expected=text_b[0].get('feature_names',[])
    for case,ec in zip(cases,Ecase):
        Xf=[]
        for cand in case['candidates']:
            x,n=extract(case,cand)
            if fact_expected and n!=fact_expected:raise RuntimeError('Selection factual feature schema differs from training')
            Xf.append(x)
        Xt,nt=extract_case(case)
        if text_expected and nt!=text_expected:raise RuntimeError('Selection text-tabular feature schema differs from training')
        fp=np.mean([predict_fact(b,np.asarray(Xf),c['reranker']['device']) for b in fact_b],axis=0);ts=np.mean([_norm_score(predict_text_listwise(b,ec,Xt,c['reranker']['device'])) for b in text_b],axis=0);px=language_proxy_from_case_features(Xt,nt);cf=[core_fid(x,case) for x in case['candidates']]
        j,scores,fact_scores,elig=choose(fp,cf,ts,px,prof);cand=case['candidates'][j]
        selected.append({'serial':str(case['serial']),'selected_tag':cand['tag'],'generated':cand['text'],'candidate_parser_vec':cand['parser_vec'],'predicted_fact_heads':{'fact_f1':float(fp[j,0]),'slot_acc':float(fp[j,1])},'predicted_text_listwise_score':float(ts[j]),'language_proxy':float(px[j]),'selected_score':float(scores[j]),'candidate_scores':[float(x) for x in scores],'candidate_fact_scores':[float(x) for x in fact_scores],'factual_safe_candidate_indices':elig,'factual_safe_candidate_tags':[case['candidates'][k]['tag'] for k in elig],'candidate_tags':[x['tag'] for x in case['candidates']]})
        print(f'[R3.2-S4-v2.2 selection] serial={case["serial"]} -> {cand["tag"]} safe={len(elig)}/{len(case["candidates"])}',flush=True)
    write_jsonl(selected,o/'06_selection/SELECTED_PREDICTIONS.jsonl');dist={};safe=[]
    for r in selected:dist[r['selected_tag']]=dist.get(r['selected_tag'],0)+1;safe.append(len(r['factual_safe_candidate_indices']))
    dump_json({'status':'PASS','cases':len(selected),'tag_distribution':dist,'mean_factual_safe_set_size':float(np.mean(safe)),'profile':prof,'reference_or_gt_visible_to_reranker':False,'pool':'frozen full S4','selector':'FactNet + frozen-Qwen contextual embedding + case-level listwise text ranker'},o/'06_selection/SELECTION_DECISION_TRACE.json')
if __name__=='__main__':main()
