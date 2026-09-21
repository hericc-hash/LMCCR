from __future__ import annotations
import sys,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lumbar_cf_report.selection.planner import Stage23UnifiedClinicalPlanner
from lumbar_cf_report.selection.features import extract,extract_case,language_proxy_from_case_features
from lumbar_cf_report.selection.text_embedder import build_prefix,input_fingerprint,encode_pool,cache_to_case_embeddings
from lumbar_cf_report.selection.reranker import fit_fact,fit_text_listwise,predict_fact,predict_text_listwise,search_profile,choose,listwise_diagnostics

def planner_smoke():
    cfg={'model':{'task_dim':128,'global_dim':256,'coordinate_dim':47,'slot_hidden':96,'global_hidden':64,'coordinate_hidden':32,'embedding_dim':16,'dropout':.1,'max_context_logit_residual':.75,'max_cross_slot_gate':.15,'initial_cross_slot_gate':.02,'plan_state_dim':128,'plan_token_dim':4096,'produce_plan_tokens':False}}
    m=Stage23UnifiedClinicalPlanner(cfg);o=m(torch.randn(2,256),torch.randn(2,5,3,128),torch.randn(2,5,47),torch.ones(2,5,3),torch.ones(2,5,3));assert o['main_probabilities'].shape==(2,16);print('STAGE23_EXACT_MODEL_SHAPE_OK')

def feature_smoke():
    cfg=json.loads((ROOT/'configs/selector/default.json').read_text());case={'core_binary':[0]*16,'slot_valid':[1]*16,'planner_probs':[.2]*16,'planner_thresholds':[.5]*16,'direct_parser_vec':[0]*16,'direct_draft':'所见：椎间盘T2WI信号减低，未见明显椎管狭窄。\n结论：腰椎退行性改变。','linguistic_scaffold':'所见：<CORE>，T2WI信号减低。\n结论：退行性改变。','adaptive_meta':{'score':1.2},'expanded':True,'reference_raw':'SHOULD_NOT_APPEAR','slot_labels':[1]*16,'candidates':[]}
    texts=['所见：椎间盘T2WI信号减低。\n结论：腰椎退行性改变。','所见：椎间盘信号减低，未见明显狭窄。\n结论：退变。','所见：T2WI信号减低。\n结论：腰椎退行性改变。']
    for i,t in enumerate(texts):case['candidates'].append({'tag':['greedy','sample0','sample1'][i],'text':t,'parser_vec':[0]*16})
    x,n=extract(case,case['candidates'][0]);X,n2=extract_case(case);assert len(x)==len(n) and X.shape[0]==3 and X.shape[1]==len(n2)
    bad=[z for z in n2 if any(k in z.lower() for k in ['reference','rouge','bleu','slot_label','gt_','logprob'])];assert not bad,bad
    px=language_proxy_from_case_features(X,n2);assert px.shape==(3,) and np.isfinite(px).all();assert cfg['generation']['sample_count']==4
    pref=build_prefix(case);assert 'SHOULD_NOT_APPEAR' not in pref and 'slot_labels' not in pref;fp=input_fingerprint(case,case['candidates'][0]);assert len(fp)==64
    print('TEXT_INPUT_FIREWALL_OK',len(n),'->',len(n2))


class _TinyTok:
    pad_token_id=0; eos_token_id=2
    def __call__(self,text,add_special_tokens=False):
        return {'input_ids':[3+(ord(ch)%31) for ch in str(text)] or [2]}
class _TinyBackbone(torch.nn.Module):
    def forward(self,input_ids,attention_mask=None,use_cache=False,return_dict=True):
        h=torch.nn.functional.one_hot((input_ids%16).long(),num_classes=16).float()
        return type('O',(),{'last_hidden_state':h})()
class _TinyModel:
    def __init__(self):self.model=_TinyBackbone()
    def eval(self):return self

def encoder_smoke():
    row={'serial':'x','core_binary':[0]*16,'slot_valid':[1]*16,'planner_probs':[.2]*16,'direct_draft':'direct','linguistic_scaffold':'<CORE> scaffold','candidates':[{'tag':'greedy','text':'candidate A'},{'tag':'sample0','text':'candidate B'}]}
    cfg={'text_encoder':{'batch_size':2,'prefix_max_tokens':80,'candidate_max_tokens':40}}
    cache=encode_pool(_TinyTok(),_TinyModel(),[row],cfg,'cpu');assert cache['embedding'].shape==(2,16);mapped=cache_to_case_embeddings([row],cache,True);assert mapped[0].shape==(2,16);print('FROZEN_TEXT_ENCODER_CACHE_OK',cache['embedding'].shape)

def reranker_smoke():
    rng=np.random.default_rng(3);E=[];Xt=[];Xf=[];Yf=[];Ylang=[];g=[];cases=[];ranges=[]
    for s in range(20):
        st=len(E);cs={'serial':str(s),'candidates':[]};style=rng.normal(size=12)
        for j in range(5):
            f=.67+.055*j+rng.normal(scale=.006);lang=.52+.055*(2-abs(j-2))+rng.normal(scale=.003)
            emb=style+.28*j+rng.normal(scale=.08,size=12);tab=np.r_[rng.normal(size=10),lang]
            E.append(emb);Xt.append(tab);Xf.append(np.r_[rng.normal(size=12),f]);Yf.append([min(.98,f),min(.99,f+.02)]);Ylang.append(lang);g.append(str(s));tp=6+j//2;fp=max(0,2-j//2);gp=8;cs['candidates'].append({'tag':f'c{j}','tp':tp,'fp':fp,'gt_pos':gp,'rouge':lang+.04,'bleu':lang-.02})
        cases.append(cs);ranges.append((st,len(E)))
    E=np.asarray(E,np.float32);Xt=np.asarray(Xt,np.float32);Xf=np.asarray(Xf,np.float32);Yf=np.asarray(Yf,np.float32);Ylang=np.asarray(Ylang,np.float32)
    fb,_=fit_fact(Xf,Yf,g,seed=2,epochs=70,lr=.003,device='cpu');tb,_=fit_text_listwise(E,Xt,Ylang,Yf[:,0],g,seed=4,epochs=90,lr=.003,device='cpu',batch_cases=8,train_fact_margin=.14,target_temp=.04)
    pf=predict_fact(fb,Xf,'cpu');pt=predict_text_listwise(tb,E,Xt,'cpu');assert pf.shape==(100,2) and pt.shape==(100,);diag=listwise_diagnostics(pt,Ylang,Yf[:,0],g,.14);assert diag['cases']==20 and diag['top1_exact_rate']>.45,diag
    pfb=[];ptb=[];cf=[];proxy=[]
    for a,b in ranges:pfb.append(pf[a:b]);ptb.append(pt[a:b]);cf.append([.68,.75,.82,.89,.95]);proxy.append(np.linspace(.2,.8,b-a))
    grid={'alpha_fact':[.6,.9],'fact_margin':[.03,.08,.16,.24],'text_model_weight':[.75,1.0]};prof,_=search_profile(cases,pfb,ptb,cf,proxy,.60,grid);assert prof['metrics']['clinical_f1']>=.60;j,s,fs,elig=choose(pfb[0],cf[0],ptb[0],proxy[0],{k:v for k,v in prof.items() if k!='choice'});assert 0<=j<5 and len(elig)>=1
    print('TEXT_LISTWISE_RERANKER_OK',diag,prof['rule'])

if __name__=='__main__':planner_smoke();feature_smoke();encoder_smoke();reranker_smoke();print('SELFTEST_OK')
