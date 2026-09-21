from __future__ import annotations
import json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,dump_json,read_jsonl
from lumbar_cf_report.selection.features import extract,extract_case,language_proxy_from_case_features
from lumbar_cf_report.selection.text_embedder import load_embedding_cache,cache_to_case_embeddings
from lumbar_cf_report.selection.reranker import fit_fact,fit_text_listwise,predict_fact,predict_text_listwise,save_bundle,search_profile,strip_choice,listwise_diagnostics

def core_fidelity(cand,case):
    p=np.asarray(cand['parser_vec'],int);c=np.asarray(case['core_binary'],int);v=np.asarray(case['slot_valid'],int)>0
    tp=int(((p==1)&(c==1)&v).sum());fp=int(((p==1)&(c==0)&v).sum());gp=int(((c==1)&v).sum());d=tp+fp+gp
    return 2*tp/d if d else 0.

def main():
    c=load_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));rc=c['reranker'];o=Path(os.environ['OUT']);cases=read_jsonl(o/'03_development_candidates/DEV388_S4_CANDIDATE_POOL.jsonl')
    if len(cases)!=388:raise RuntimeError(f'Expected 388 Development cases, got {len(cases)}')
    cache=load_embedding_cache(o/'02_text_embeddings/DEV388_TEXT_EMBEDDINGS.npz');Ecase=cache_to_case_embeddings(cases,cache,True)
    Xf=[];Xt=[];E=[];Yf=[];Ylang=[];groups=[];ranges=[];fact_names=None;text_names=None;rw=float(rc['language_balance_rouge_weight']);bw=float(rc['language_balance_bleu_weight'])
    for case,ec in zip(cases,Ecase):
        st=len(Xf);XL,nl=extract_case(case)
        if text_names is None:text_names=nl
        elif nl!=text_names:raise RuntimeError('Text tabular feature schema drift')
        if len(XL)!=len(case['candidates']) or len(ec)!=len(case['candidates']):raise RuntimeError('Candidate feature/embedding alignment mismatch')
        for j,cand in enumerate(case['candidates']):
            xf,nf=extract(case,cand)
            if fact_names is None:fact_names=nf
            elif nf!=fact_names:raise RuntimeError('Factual feature schema drift')
            Xf.append(xf);Xt.append(XL[j]);E.append(ec[j]);Yf.append([float(cand['target_fact_f1']),float(cand['target_slot_acc'])]);Ylang.append(rw*float(cand['rouge'])+bw*float(cand['bleu']));groups.append(str(case['serial']))
        ranges.append((st,len(Xf)))
    forbidden=['reference','rouge','bleu','slot_label','gt_','logprob']
    for label,names in [('factual',fact_names),('text_tabular',text_names)]:
        bad=[n for n in names if any(k in n.lower() for k in forbidden)]
        if bad:raise RuntimeError(label+' inference feature firewall violation '+str(bad[:12]))
    Xf=np.asarray(Xf,np.float32);Xt=np.asarray(Xt,np.float32);E=np.asarray(E,np.float32);Yf=np.asarray(Yf,np.float32);Ylang=np.asarray(Ylang,np.float32)
    folds=int(rc['cv_folds']);oof_fact=np.zeros((len(Xf),2),np.float32);oof_text=np.zeros(len(Xf),np.float32);fold_reports=[]
    for fold in range(folds):
        train_idx=np.asarray([i for i,g in enumerate(groups) if int(g)%folds!=fold],int);test_idx=np.asarray([i for i,g in enumerate(groups) if int(g)%folds==fold],int)
        fb,_=fit_fact(Xf[train_idx],Yf[train_idx],[groups[i] for i in train_idx],seed=c['seed']+fold,epochs=rc['epochs_fact'],lr=rc['lr_fact'],device=rc['device'])
        tb,_=fit_text_listwise(E[train_idx],Xt[train_idx],Ylang[train_idx],Yf[train_idx,0],[groups[i] for i in train_idx],seed=c['seed']+100+fold,epochs=rc['epochs_text_listwise'],lr=rc['lr_text_listwise'],device=rc['device'],batch_cases=rc['listwise_batch_cases'],train_fact_margin=rc['listwise_train_fact_margin'],target_temp=rc['listwise_target_temperature'],pred_temp=rc['listwise_pred_temperature'])
        oof_fact[test_idx]=predict_fact(fb,Xf[test_idx],rc['device']);oof_text[test_idx]=predict_text_listwise(tb,E[test_idx],Xt[test_idx],rc['device'])
        fold_reports.append({'fold':fold,'train_candidates':int(len(train_idx)),'test_candidates':int(len(test_idx)),'fact_train_loss':fb['train_loss'],'text_listwise_train_loss':tb['train_loss'],'fact_epochs':fb['epochs_ran'],'text_epochs':tb['epochs_ran']})
        print(f'[R3.2-S4-v2.2 CV] fold={fold+1}/{folds} train={len(train_idx)} test={len(test_idx)}',flush=True)
    diag=listwise_diagnostics(oof_text,Ylang,Yf[:,0],groups,rc['listwise_train_fact_margin'])
    fact_by=[];text_by=[];cf_by=[];proxy_by=[]
    for case,(a,b) in zip(cases,ranges):
        fact_by.append(oof_fact[a:b]);text_by.append(oof_text[a:b]);cf_by.append([core_fidelity(x,case) for x in case['candidates']]);XL,n=extract_case(case);proxy_by.append(language_proxy_from_case_features(XL,n))
    lt=(float(c['selection_gate']['minimum_rouge_l']),float(c['selection_gate']['minimum_bleu4']));primary,top_primary=search_profile(cases,fact_by,text_by,cf_by,proxy_by,float(rc['primary_profile_target_f1']),rc['profile_grid'],lt);secondary,top_secondary=search_profile(cases,fact_by,text_by,cf_by,proxy_by,float(rc['secondary_profile_target_f1']),rc['profile_grid'],lt)
    p1=strip_choice(primary);p2=strip_choice(secondary);p1['role']='PRIMARY_TEXT_LISTWISE_F1_070';p2['role']='SECONDARY_TEXT_LISTWISE_F1_072';note='Selected only from Development388 OOF. FactNet creates a deployment-safe factual candidate set; frozen-Qwen contextual text embeddings + listwise ranker select language realization inside it. No GT/reference at inference.';p1['selection_rule_note']=note;p2['selection_rule_note']=note
    dump_json(p1,o/'04_reranker/SELECTED_PROFILE_PRIMARY.json');dump_json(p2,o/'04_reranker/SELECTED_PROFILE_SECONDARY_F1_072.json');dump_json(p1,o/'04_reranker/SELECTED_PROFILE.json')
    report={'folds':fold_reports,'primary_profile':p1,'secondary_profile_f1_072':p2,'top50_primary_profiles':[strip_choice(x) for x in top_primary],'top50_secondary_profiles':[strip_choice(x) for x in top_secondary],'oof_text_listwise_diagnostics':diag,'fact_feature_count':len(fact_names),'text_tabular_feature_count':len(text_names),'text_embedding_dim':int(E.shape[1]),'fact_feature_names':fact_names,'text_tabular_feature_names':text_names,'firewall_pass':True,'source_pool_summary':load_json(o/'03_development_candidates/DEV_POOL_SUMMARY.json'),'supervision_note':'Development388 GT/reference-derived factual and ROUGE/BLEU are training/listwise targets only. Frozen text encoder input is Planner + Direct + scaffold + candidate only.'}
    dump_json(report,o/'04_reranker/OOF_CV_REPORT.json')
    fact_models=[];text_models=[]
    for seed in rc['final_seeds']:
        fb,_=fit_fact(Xf,Yf,groups,seed=int(seed),epochs=rc['epochs_fact'],lr=rc['lr_fact'],device=rc['device']);fp=o/f'04_reranker/fact_seed{seed}.pt';save_bundle(fb,fp,fact_names);fact_models.append(str(fp))
        tb,_=fit_text_listwise(E,Xt,Ylang,Yf[:,0],groups,seed=int(seed)+1000,epochs=rc['epochs_text_listwise'],lr=rc['lr_text_listwise'],device=rc['device'],batch_cases=rc['listwise_batch_cases'],train_fact_margin=rc['listwise_train_fact_margin'],target_temp=rc['listwise_target_temperature'],pred_temp=rc['listwise_pred_temperature']);tp=o/f'04_reranker/text_listwise_seed{seed}.pt';save_bundle(tb,tp,text_names);text_models.append(str(tp))
        print(f'[R3.2-S4-v2.2 final] seed={seed} fact_loss={fb["train_loss"]:.6f} text_loss={tb["train_loss"]:.6f}',flush=True)
    rep={'status':'PASS','development_cases':len(cases),'candidate_rows':len(Xf),'primary_profile':p1,'secondary_profile_f1_072':p2,'fact_models':fact_models,'text_listwise_models':text_models,'target_selection_gate':c['selection_gate'],'fact_feature_count':len(fact_names),'text_tabular_feature_count':len(text_names),'text_embedding_dim':int(E.shape[1]),'oof_text_listwise_diagnostics':diag,'firewall':'GT/reference/ROUGE/BLEU used only as Development supervision. Text encoder sees only deployment-safe Planner/Direct/scaffold/candidate. No candidate model logprob.'}
    dump_json(rep,o/'04_reranker/TRAINING_COMPLETE.json');print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
