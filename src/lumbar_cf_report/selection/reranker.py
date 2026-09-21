from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn

FACT_HEAD_NAMES=['fact_f1','slot_acc']

class FactNet(nn.Module):
    def __init__(self,d,h=160):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Dropout(.10),nn.Linear(h,96),nn.GELU(),nn.Dropout(.06),nn.Linear(96,2))
    def forward(self,x):return torch.sigmoid(self.net(x))

class TextListwiseNet(nn.Module):
    def __init__(self,emb_dim,tab_dim,emb_proj=192,tab_proj=64,hidden=128):
        super().__init__()
        self.emb=nn.Sequential(nn.LayerNorm(emb_dim),nn.Linear(emb_dim,emb_proj),nn.GELU(),nn.Dropout(.12))
        self.tab=nn.Sequential(nn.Linear(tab_dim,tab_proj),nn.GELU(),nn.Dropout(.10))
        self.head=nn.Sequential(nn.Linear(emb_proj+tab_proj,hidden),nn.GELU(),nn.Dropout(.12),nn.Linear(hidden,1))
    def forward(self,e,t):
        return self.head(torch.cat([self.emb(e),self.tab(t)],dim=-1)).squeeze(-1)

def _seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

def _device(device):return torch.device(device if str(device).startswith('cuda') and torch.cuda.is_available() else 'cpu')

def _pair_tensor(values,groups,threshold,device,weight_scale=8.0):
    by={}
    for i,g in enumerate(groups):by.setdefault(str(g),[]).append(i)
    pairs=[]
    for ids in by.values():
        for a in range(len(ids)):
            for b in range(a+1,len(ids)):
                i,j=ids[a],ids[b];d=float(values[i]-values[j])
                if abs(d)>=threshold:pairs.append((i,j,1. if d>0 else -1.,min(4.,1.+abs(d)*weight_scale)))
    if not pairs:return None
    return (torch.tensor([[a,b] for a,b,_,_ in pairs],device=device,dtype=torch.long),torch.tensor([s for _,_,s,_ in pairs],device=device),torch.tensor([w for *_,w in pairs],device=device))

def fit_fact(X,Y,groups,seed=1,epochs=280,lr=1.8e-3,device='cpu'):
    _seed(seed);X=np.asarray(X,np.float32);Y=np.asarray(Y,np.float32);mean=X.mean(0);std=X.std(0);std[std<1e-5]=1.;Z=(X-mean)/std;dev=_device(device)
    xt=torch.tensor(Z,device=dev);yt=torch.tensor(Y,device=dev);m=FactNet(X.shape[1]).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1.5e-3)
    pairs=[(_pair_tensor(Y[:,0],groups,.018,dev),.18),(_pair_tensor(Y[:,1],groups,.018,dev),.06)];best=None;bestloss=1e9;pat=0
    for ep in range(int(epochs)):
        m.train();pred=m(xt);loss=(1.35*(pred[:,0]-yt[:,0])**2+.55*(pred[:,1]-yt[:,1])**2).mean()
        for h,(pdata,w) in enumerate(pairs):
            if pdata is None:continue
            pi,ps,pw=pdata;diff=(pred[pi[:,0],h]-pred[pi[:,1],h])*ps;loss=loss+w*(torch.nn.functional.softplus(-diff)*pw).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();lv=float(loss.detach().cpu())
        if lv<bestloss-1e-5:bestloss=lv;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};pat=0
        else:pat+=1
        if ep>90 and pat>50:break
    m.load_state_dict(best);m.eval();return {'kind':'fact','state_dict':best,'mean':mean,'std':std,'in_dim':X.shape[1],'hidden':160,'seed':seed,'train_loss':bestloss,'epochs_ran':ep+1,'head_names':FACT_HEAD_NAMES},m

def _group_indices(groups):
    by={}
    for i,g in enumerate(groups):by.setdefault(str(g),[]).append(i)
    return list(by.values())

def _batch_groups(group_ids,batch_size,shuffle,rng):
    order=np.arange(len(group_ids))
    if shuffle:rng.shuffle(order)
    for st in range(0,len(order),batch_size):yield [group_ids[i] for i in order[st:st+batch_size]]

def _pack_case_batch(E,T,lang,fact,batch_groups,tab_mean,tab_std,dev,train_fact_margin):
    B=len(batch_groups);K=max(len(x) for x in batch_groups);D=E.shape[1];F=T.shape[1]
    eb=np.zeros((B,K,D),np.float32);tb=np.zeros((B,K,F),np.float32);mask=np.zeros((B,K),bool);target=np.zeros((B,K),np.float32);safe=np.zeros((B,K),bool)
    for bi,ids in enumerate(batch_groups):
        n=len(ids);eb[bi,:n]=E[ids];tb[bi,:n]=(T[ids]-tab_mean)/tab_std;mask[bi,:n]=True
        lf=lang[ids];ff=fact[ids];best=float(np.max(ff));sm=ff>=best-float(train_fact_margin)-1e-12
        if not sm.any():sm[np.argmax(ff)]=True
        safe[bi,:n]=sm;target[bi,:n]=lf
    return (torch.tensor(eb,device=dev),torch.tensor(tb,device=dev),torch.tensor(mask,device=dev),torch.tensor(target,device=dev),torch.tensor(safe,device=dev))

def _listwise_loss(scores,mask,lang,safe,target_temp=.035,pred_temp=1.0):
    valid=mask&safe
    # At least one valid candidate per case by construction.
    neg=-1e4
    logits=scores/float(pred_temp);logits=logits.masked_fill(~valid,neg)
    target_logits=lang/float(target_temp);target_logits=target_logits.masked_fill(~valid,neg)
    q=torch.softmax(target_logits,dim=1).detach();logp=torch.log_softmax(logits,dim=1)
    return -(q*logp*valid.float()).sum(1).mean()

def fit_text_listwise(E,T,lang_target,fact_target,groups,seed=1,epochs=220,lr=8e-4,device='cpu',batch_cases=32,train_fact_margin=.10,target_temp=.035,pred_temp=1.0):
    _seed(seed);E=np.asarray(E,np.float32);T=np.asarray(T,np.float32);lang=np.asarray(lang_target,np.float32);fact=np.asarray(fact_target,np.float32);dev=_device(device)
    tab_mean=T.mean(0);tab_std=T.std(0);tab_std[tab_std<1e-5]=1.;m=TextListwiseNet(E.shape[1],T.shape[1]).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=2e-3)
    gids=_group_indices(groups);rng=np.random.default_rng(seed);best=None;bestloss=1e9;pat=0
    for ep in range(int(epochs)):
        m.train();ls=[]
        for bg in _batch_groups(gids,int(batch_cases),True,rng):
            eb,tb,mask,lt,safe=_pack_case_batch(E,T,lang,fact,bg,tab_mean,tab_std,dev,train_fact_margin)
            score=m(eb,tb);loss=_listwise_loss(score,mask,lt,safe,target_temp,pred_temp)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),4.);opt.step();ls.append(float(loss.detach().cpu()))
        lv=float(np.mean(ls))
        if lv<bestloss-1e-4:bestloss=lv;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};pat=0
        else:pat+=1
        if ep>70 and pat>35:break
    m.load_state_dict(best);m.eval()
    return {'kind':'text_listwise','state_dict':best,'tab_mean':tab_mean,'tab_std':tab_std,'emb_dim':E.shape[1],'tab_dim':T.shape[1],'emb_proj':192,'tab_proj':64,'hidden':128,'seed':seed,'train_loss':bestloss,'epochs_ran':ep+1,'train_fact_margin':float(train_fact_margin),'target_temperature':float(target_temp),'pred_temperature':float(pred_temp)},m

def _predict_fact(bundle,X,device='cpu'):
    X=np.asarray(X,np.float32);dev=_device(device);m=FactNet(int(bundle['in_dim']),int(bundle.get('hidden',160))).to(dev);m.load_state_dict(bundle['state_dict']);m.eval();z=(X-np.asarray(bundle['mean'],np.float32))/np.asarray(bundle['std'],np.float32)
    with torch.no_grad():return m(torch.tensor(z,device=dev)).cpu().numpy()

def predict_fact(bundle,X,device='cpu'):return _predict_fact(bundle,X,device)

def predict_text_listwise(bundle,E,T,device='cpu'):
    E=np.asarray(E,np.float32);T=np.asarray(T,np.float32);dev=_device(device);m=TextListwiseNet(int(bundle['emb_dim']),int(bundle['tab_dim']),int(bundle.get('emb_proj',192)),int(bundle.get('tab_proj',64)),int(bundle.get('hidden',128))).to(dev);m.load_state_dict(bundle['state_dict']);m.eval();zt=(T-np.asarray(bundle['tab_mean'],np.float32))/np.asarray(bundle['tab_std'],np.float32)
    with torch.no_grad():return m(torch.tensor(E,device=dev),torch.tensor(zt,device=dev)).cpu().numpy().reshape(-1)

def save_bundle(bundle,path,feature_names=None):
    q=dict(bundle);q['feature_names']=feature_names or [];Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(q,path)
def load_bundle(path):return torch.load(path,map_location='cpu',weights_only=False)

def aggregate_choices(cases,choice):
    tp=fp=gp=0;rr=[];bb=[];tags={}
    for c,j in zip(cases,choice):
        x=c['candidates'][int(j)];tp+=int(x['tp']);fp+=int(x['fp']);gp+=int(x['gt_pos']);rr.append(float(x['rouge']));bb.append(float(x['bleu']));tags[x['tag']]=tags.get(x['tag'],0)+1
    f=2*tp/(tp+fp+gp) if tp+fp+gp else 0.
    return {'clinical_f1':f,'rouge_l':float(np.mean(rr)),'bleu4':float(np.mean(bb)),'language_balance':float(.45*np.mean(rr)+.55*np.mean(bb)),'tags':tags,'tp':tp,'fp':fp,'gt_pos':gp}

def _minmax(x):
    x=np.asarray(x,float)
    if len(x)<=1:return np.zeros_like(x)
    lo,hi=float(x.min()),float(x.max())
    return (x-lo)/(hi-lo+1e-8)

def _choice(fact_pred,core_fidelity,text_score,proxy,alpha,delta,text_weight):
    fp=np.asarray(fact_pred,float);pred_fact=.75*fp[:,0]+.25*fp[:,1];fact=alpha*pred_fact+(1-alpha)*np.asarray(core_fidelity,float)
    mx=float(np.max(fact));elig=np.flatnonzero(fact>=mx-float(delta)-1e-12)
    if len(elig)==0:elig=np.asarray([int(np.argmax(fact))])
    ts=_minmax(text_score);px=_minmax(proxy);lang=float(text_weight)*ts+(1.-float(text_weight))*px;j=int(elig[int(np.argmax(lang[elig]))])
    return j,elig,fact,lang

def _profile_rows(cases,fact_preds_by_case,text_scores_by_case,core_fid_by_case,proxy_by_case,target_f1,grid,language_targets=(.55,.45)):
    rows=[]
    for alpha in grid['alpha_fact']:
      for delta in grid['fact_margin']:
       for tw in grid['text_model_weight']:
        ch=[];ss=[]
        for fp,ts,cf,px in zip(fact_preds_by_case,text_scores_by_case,core_fid_by_case,proxy_by_case):
            j,elig,_,_=_choice(fp,cf,ts,px,float(alpha),float(delta),float(tw));ch.append(j);ss.append(len(elig))
        met=aggregate_choices(cases,ch);ok=met['clinical_f1']+1e-12>=target_f1
        jt=min(met['rouge_l']/max(1e-8,float(language_targets[0])),met['bleu4']/max(1e-8,float(language_targets[1])));rows.append({'rule':'factual_safe_text_listwise','alpha_fact':float(alpha),'fact_margin':float(delta),'text_model_weight':float(tw),'eligible_f1':ok,'language_score':met['language_balance'],'joint_gate_score':float(jt),'mean_safe_set_size':float(np.mean(ss)),'metrics':met,'choice':ch})
    return rows

def search_profile(cases,fact_preds_by_case,text_scores_by_case,core_fid_by_case,proxy_by_case,target_f1,grid,language_targets=(.55,.45)):
    rows=_profile_rows(cases,fact_preds_by_case,text_scores_by_case,core_fid_by_case,proxy_by_case,target_f1,grid,language_targets);elig=[r for r in rows if r['eligible_f1']]
    # Gate-aware OOF selection: prioritize the weaker of ROUGE/BLEU relative to the requested joint target.
    if elig:best=max(elig,key=lambda r:(r['joint_gate_score'],r['language_score'],r['metrics']['bleu4'],r['metrics']['rouge_l'],r['metrics']['clinical_f1']))
    else:best=max(rows,key=lambda r:(r['metrics']['clinical_f1'],r['joint_gate_score'],r['language_score']))
    top=sorted(rows,key=lambda r:(r['eligible_f1'],r['joint_gate_score'],r['language_score'],r['metrics']['clinical_f1']),reverse=True)[:50];return best,top

def strip_choice(profile):return {k:v for k,v in profile.items() if k!='choice'}

def choose(fact_pred,core_fidelity,text_score,proxy,profile):
    j,elig,fact,lang=_choice(fact_pred,core_fidelity,text_score,proxy,float(profile['alpha_fact']),float(profile['fact_margin']),float(profile['text_model_weight']))
    score=np.full(len(lang),-1e9,dtype=float);score[elig]=lang[elig];return j,score.tolist(),fact.tolist(),[int(x) for x in elig.tolist()]

def listwise_diagnostics(scores,lang,fact,groups,fact_margin=.10):
    by={}
    for i,g in enumerate(groups):by.setdefault(str(g),[]).append(i)
    exact=0;reg=[];n=0
    for ids in by.values():
        f=fact[ids];best=float(np.max(f));safe=[ids[k] for k,x in enumerate(f) if x>=best-float(fact_margin)-1e-12]
        if not safe:continue
        pred=max(safe,key=lambda i:float(scores[i]));oracle=max(safe,key=lambda i:float(lang[i]));exact+=int(pred==oracle);reg.append(float(lang[oracle]-lang[pred]));n+=1
    return {'cases':n,'top1_exact_rate':exact/max(1,n),'mean_language_regret':float(np.mean(reg)) if reg else 0.0}
