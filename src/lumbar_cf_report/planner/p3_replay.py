from __future__ import annotations
import ast, importlib.util, inspect, os, shutil, traceback
from pathlib import Path
import numpy as np
import torch
from .io import safe_load, sha256_file, serials

# Production contract:
# - pair_bank['direction'] is a categorical orientation label, NOT a latent vector.
# - historical P3 response is level-wise [rows, 3, 128].

_LEVEL_MAP={
    'l1/2':0,'l1-2':0,'l1_l2':0,'l12':0,
    'l2/3':1,'l2-3':1,'l2_l3':1,'l23':1,
    'l3/4':2,'l3-4':2,'l3_l4':2,'l34':2,
    'l4/5':3,'l4-5':3,'l4_l5':3,'l45':3,
    'l5/s1':4,'l5-s1':4,'l5_s1':4,'l5s1':4,
}
_TASK_MAP={
    'disc':0,'stenosis':1,'nerve':2,
    'disc_hernia_or_bulge':0,'spinal_canal_stenosis':1,'nerve_root_compression':2,
}
_PAIR_ALIASES={
    'recipient':('recipient','recipient_serial','recipient_idx','recipient_id','case','case_serial','serial'),
    'donor':('donor','donor_serial','donor_idx','donor_id'),
    'level':('level','level_index','li','level_idx','segment','segment_index'),
    'task':('task','task_index','ti','task_idx','disease','disease_index'),
    'direction_label':('direction','pair_direction','orientation_label'),
    'orientation_active':('orientation_active','active_orientation','is_primary'),
    'sample_weight':('sample_weight','weight','pair_weight'),
    'local_scale':('local_scale','selected_scale','step_scale','scale'),
}
_META_ALIASES={
    'recipient_serial':('recipient_serial','recipient','recipient_id','case_serial','serial'),
    'level':('level','level_index','li','level_idx','segment_index'),
    'task':('task','task_index','ti','task_idx','disease_index'),
    'sign':('sign','orientation_sign','minority_sign','target_sign','signed_direction'),
    'local_scale':('local_scale','selected_scale','step_scale','scale'),
}

def _schema(x,path='root',depth=0,out=None):
    if out is None: out=[]
    if depth>5:return out
    if torch.is_tensor(x):
        out.append({'path':path,'type':'tensor','shape':list(x.shape),'dtype':str(x.dtype)});return out
    if isinstance(x,np.ndarray):
        out.append({'path':path,'type':'ndarray','shape':list(x.shape),'dtype':str(x.dtype)});return out
    if isinstance(x,dict):
        out.append({'path':path,'type':'dict','keys':[str(k) for k in list(x)[:120]]})
        for k,v in list(x.items())[:120]:_schema(v,f'{path}.{k}',depth+1,out)
    elif isinstance(x,(list,tuple)):
        out.append({'path':path,'type':type(x).__name__,'len':len(x)})
        for i,v in enumerate(x[:8]):_schema(v,f'{path}[{i}]',depth+1,out)
    elif isinstance(x,(str,int,float,bool,type(None))):
        out.append({'path':path,'type':type(x).__name__,'value':x})
    return out

def _lower_key_map(x):return {str(k).lower():k for k in x} if isinstance(x,dict) else {}

def _find_record_value(rec,aliases):
    if not isinstance(rec,dict):return None
    low=_lower_key_map(rec)
    for name in aliases:
        kk=low.get(name.lower())
        if kk is not None:return rec[kk]
    for nk in ('meta','metadata','pair','indices','index','info','row'):
        kk=low.get(nk)
        if kk is not None and isinstance(rec[kk],dict):
            v=_find_record_value(rec[kk],aliases)
            if v is not None:return v
    return None

def _as_list(vals):
    if vals is None:return None
    if torch.is_tensor(vals):return vals.detach().cpu().reshape(-1).tolist()
    if isinstance(vals,np.ndarray):return vals.reshape(-1).tolist()
    if isinstance(vals,(list,tuple)):return list(vals)
    return [vals]

def _scalar_seq(vals,kind):
    vals=_as_list(vals)
    if vals is None:return None
    out=[]
    for v in vals:
        if torch.is_tensor(v) and v.numel()==1:v=v.item()
        if isinstance(v,np.generic):v=v.item()
        if kind=='level' and isinstance(v,str):
            key=v.strip().lower().replace(' ','');v=_LEVEL_MAP.get(key,v)
        if kind=='task' and isinstance(v,str):
            key=v.strip().lower().replace(' ','_');v=_TASK_MAP.get(key,v)
        out.append(v)
    if kind in ('level','task'):
        try:return torch.tensor([int(v) for v in out],dtype=torch.long)
        except Exception:return out
    if kind in ('orientation_active','sample_weight','sign','local_scale'):
        try:return torch.tensor([float(v) for v in out],dtype=torch.float32)
        except Exception:return out
    return out

def _columnar(x):
    low=_lower_key_map(x);ret={}
    for name,aliases in _PAIR_ALIASES.items():
        ret[name]=None
        for alias in aliases:
            kk=low.get(alias.lower())
            if kk is not None:ret[name]=x[kk];break
    return ret

def _records(records):
    if not records:raise ValueError('pair split list is empty')
    if not all(isinstance(r,dict) for r in records):raise TypeError('pair split list must contain dict records')
    cols={k:[] for k in _PAIR_ALIASES};missing=[]
    for i,r in enumerate(records):
        vals={k:_find_record_value(r,a) for k,a in _PAIR_ALIASES.items()}
        req=[k for k in ('recipient','level','task') if vals[k] is None]
        if req:missing.append({'row':i,'missing':req,'keys':list(map(str,r.keys()))})
        for k in cols:cols[k].append(vals[k])
    if missing:raise KeyError(f'incomplete pair records: {missing[:8]}')
    for k in ('donor','direction_label','orientation_active','sample_weight','local_scale'):
        if all(v is None for v in cols[k]):cols[k]=None
    return cols

def normalize_pair_split(pair_bank,split):
    if not isinstance(pair_bank,dict):raise TypeError(f'pair_bank root must be dict, got {type(pair_bank)}')
    aliases=[split]+(['development','dev'] if split=='train' else ['val','validation','internal100'] if split=='internal' else [])
    low=_lower_key_map(pair_bank);x=None;used=None
    for a in aliases:
        kk=low.get(a.lower())
        if kk is not None:x=pair_bank[kk];used=str(kk);break
    if x is None:raise KeyError(f'pair_bank split={split} not found; keys={list(pair_bank)}')
    if isinstance(x,dict):cols=_columnar(x);kind='columnar_dict'
    elif isinstance(x,(list,tuple)):cols=_records(list(x));kind='record_list'
    else:raise TypeError(f'pair split must be dict or list-of-dict, got {type(x)}')
    ret={
      'recipient':_scalar_seq(cols['recipient'],'recipient'),
      'donor':_scalar_seq(cols['donor'],'donor') if cols['donor'] is not None else None,
      'level':_scalar_seq(cols['level'],'level'),
      'task':_scalar_seq(cols['task'],'task'),
      'direction_label':_scalar_seq(cols['direction_label'],'direction_label') if cols['direction_label'] is not None else None,
      'orientation_active':_scalar_seq(cols['orientation_active'],'orientation_active') if cols['orientation_active'] is not None else None,
      'sample_weight':_scalar_seq(cols['sample_weight'],'sample_weight') if cols['sample_weight'] is not None else None,
      'local_scale':_scalar_seq(cols['local_scale'],'local_scale') if cols['local_scale'] is not None else None,
      'schema_kind':kind,'source_split_key':used,
    }
    n=len(ret['recipient']);ret['rows']=n
    for k in ('level','task'):
        if len(ret[k])!=n:raise ValueError(f'{k} length mismatch')
    for k in ('donor','direction_label','orientation_active','sample_weight','local_scale'):
        if ret[k] is not None and len(ret[k])!=n:raise ValueError(f'{k} length mismatch')
    return ret

def pair_primary_mask(pairs):
    n=pairs['rows'];mask=torch.ones(n,dtype=torch.bool)
    labels=pairs.get('direction_label');ori=pairs.get('orientation_active')
    if labels is not None:
        lab=[str(x).strip().lower() for x in labels]
        recognized=[x in ('majority_to_minority','primary','forward') for x in lab]
        # Historical bank has explicit majority_to_minority vs reverse labels.
        if any(recognized):mask=torch.tensor(recognized,dtype=torch.bool)
    if ori is not None:
        o=torch.as_tensor(ori).float().reshape(-1)
        vals=set(round(float(x),6) for x in o.tolist())
        if vals.issubset({0.0,1.0}):mask &= o>=0.5
    return mask

def subset_pairs(pairs,mask):
    idx=torch.where(torch.as_tensor(mask,dtype=torch.bool))[0];ret={'rows':len(idx),'schema_kind':pairs.get('schema_kind'),'source_split_key':pairs.get('source_split_key')}
    for k in ('recipient','donor','level','task','direction_label','orientation_active','sample_weight','local_scale'):
        v=pairs.get(k)
        if v is None:ret[k]=None
        elif torch.is_tensor(v):ret[k]=v[idx]
        else:ret[k]=[v[int(i)] for i in idx]
    ret['source_indices']=idx
    return ret

def _ss(v):
    if torch.is_tensor(v) and v.numel()==1:v=v.item()
    if isinstance(v,np.generic):v=v.item()
    return str(v)

def _teacher_seq(teacher,key):
    v=teacher[key]
    if torch.is_tensor(v):return v.detach().cpu().reshape(-1)
    if isinstance(v,np.ndarray):return torch.as_tensor(v).reshape(-1)
    return list(v)

def match_teacher_to_pairs(teacher,pairs):
    tr=[_ss(v) for v in _as_list(teacher['recipient_serial'])]
    tl=torch.as_tensor(_as_list(teacher['level'])).long().reshape(-1)
    tt=torch.as_tensor(_as_list(teacher['task'])).long().reshape(-1)
    pr=[_ss(v) for v in pairs['recipient']];pl=torch.as_tensor(pairs['level']).long();pt=torch.as_tensor(pairs['task']).long();pm=pair_primary_mask(pairs)
    idx={}
    for i,(r,l,t) in enumerate(zip(pr,pl.tolist(),pt.tolist())):
        if bool(pm[i]):idx.setdefault((r,int(l),int(t)),[]).append(i)
    chosen=[];amb=[]
    for j,(r,l,t) in enumerate(zip(tr,tl.tolist(),tt.tolist())):
        c=idx.get((r,int(l),int(t)),[])
        if len(c)!=1:
            amb.append({'teacher_row':j,'recipient':r,'level':int(l),'task':int(t),'candidate_count':len(c),'candidates':c[:10]});chosen.append(-1)
        else:chosen.append(c[0])
    return torch.tensor(chosen,dtype=torch.long),amb

def _extract_vector_from_dict(dct,aliases,rows):
    if not isinstance(dct,dict):return None
    low=_lower_key_map(dct)
    for a in aliases:
        kk=low.get(a.lower())
        if kk is None:continue
        v=dct[kk]
        try:
            if a in _META_ALIASES['level']:x=_scalar_seq(v,'level')
            elif a in _META_ALIASES['task']:x=_scalar_seq(v,'task')
            elif a in _META_ALIASES['sign']:x=_scalar_seq(v,'sign')
            elif a in _META_ALIASES['local_scale']:x=_scalar_seq(v,'local_scale')
            else:x=_scalar_seq(v,'recipient')
            if x is not None and len(x)==rows:return x
        except Exception:pass
    return None

def _row_meta_from_parent(parent,rows):
    if not isinstance(parent,dict):return {}
    ret={}
    for k,aliases in _META_ALIASES.items():
        v=_extract_vector_from_dict(parent,aliases,rows)
        if v is not None:ret[k]=v
    # Try one level of common metadata containers.
    if len(ret)<3:
        for kk,v in parent.items():
            if isinstance(v,dict) and str(kk).lower() in ('meta','metadata','pair_meta','indices','rows','index'):
                for k,aliases in _META_ALIASES.items():
                    if k in ret:continue
                    z=_extract_vector_from_dict(v,aliases,rows)
                    if z is not None:ret[k]=z
    return ret

def recursive_tensor_candidates(x,path='root',rows=None,out=None,parent=None):
    if out is None:out=[]
    def add(v,p,par):
        try:t=torch.as_tensor(v).detach().cpu().float()
        except Exception:return
        if rows is not None and (t.ndim<1 or int(t.shape[0])!=int(rows)):return
        if (t.ndim==2 and t.shape[-1]==128) or (t.ndim==3 and tuple(t.shape[-2:])==(3,128)):
            out.append({'path':p,'tensor':t,'row_meta':_row_meta_from_parent(par, int(t.shape[0]))})
    if torch.is_tensor(x) or isinstance(x,np.ndarray):add(x,path,parent)
    elif isinstance(x,dict):
        for k,v in x.items():recursive_tensor_candidates(v,f'{path}.{k}',rows,out,x)
    elif isinstance(x,(list,tuple)):
        # Large record lists are not traversed element-by-element here; production P3 tensors are expected columnar.
        if len(x)<=256:
            for i,v in enumerate(x):recursive_tensor_candidates(v,f'{path}[{i}]',rows,out,parent)
    return out

def recursive_delta_candidates(x,path='root',out=None):
    if out is None:out=[]
    for c in recursive_tensor_candidates(x,path,None,[]):
        lp=c['path'].lower()
        if any(k in lp for k in ('delta','direction','response','counterfactual','cf_evidence')):
            out.append({'path':c['path'],'delta':c['tensor'],'row_meta':c.get('row_meta',{})})
    return out

def _artifact_files(roots,cfg):
    max_mb=float(cfg['replay'].get('artifact_scan_max_file_mb',512));max_files=int(cfg['replay'].get('artifact_scan_max_files',250));seen={};arr=[]
    for root in roots:
        p=Path(root)
        qs=[p] if p.is_file() else list(p.rglob('*')) if p.exists() else []
        for q in qs:
            if not q.is_file() or q.suffix.lower() not in ('.pt','.pth','.pkl'):continue
            try:size=q.stat().st_size/1024**2
            except Exception:continue
            if size>max_mb:continue
            name=str(q).lower()
            score=(8*('coordinate_cache' in name)+6*('teacher_cache' in name)+5*('p3' in name)+4*('cache' in name)+3*('evidence' in name)+3*('direction' in name)+2*('response' in name)-8*('final_planner' in name)-8*('optimizer' in name)-4*('checkpoint' in name))
            key=str(q.resolve())
            if key not in seen or score>seen[key][0]:seen[key]=(score,q)
    arr=sorted(seen.values(),key=lambda x:(-x[0],str(x[1])))[:max_files]
    return [q for _,q in arr]

def scan_historical_artifacts(cfg,rows):
    rep=[];candidates=[]
    for q in _artifact_files(cfg.get('historical_p3_cache_roots',[]),cfg):
        rec={'file':str(q),'sha256':None,'load_error':None,'tensors':[],'root_schema':[]}
        try:
            rec['sha256']=sha256_file(q);x=safe_load(q);cs=recursive_tensor_candidates(x,'root',rows,[])
            rec['root_schema']=_schema(x,depth=0)[:120]
            for c in cs:
                item={'file':str(q),'path':c['path'],'shape':list(c['tensor'].shape),'tensor':c['tensor'],'row_meta':c.get('row_meta',{})}
                candidates.append(item);rec['tensors'].append({'path':c['path'],'shape':list(c['tensor'].shape),'row_meta_keys':sorted(c.get('row_meta',{}).keys())})
        except Exception as e:rec['load_error']=repr(e)
        if rec['tensors'] or rec['load_error']:rep.append(rec)
    return candidates,rep

def _reorder_tensor_by_meta(t,row_meta,target_meta):
    """Reorder artifact rows to target recipient/level/task if sufficient metadata exists.
    Returns (tensor, alignment_note). If no metadata, leaves order unchanged and marks assumed.
    """
    keys=('recipient_serial','level','task')
    if not all(k in row_meta for k in keys):return t,'ROW_ORDER_ASSUMED_FROM_FROZEN_ARTIFACT'
    ar=[_ss(v) for v in _as_list(row_meta['recipient_serial'])];al=torch.as_tensor(row_meta['level']).long();at=torch.as_tensor(row_meta['task']).long()
    tr=[_ss(v) for v in _as_list(target_meta['recipient_serial'])];tl=torch.as_tensor(_as_list(target_meta['level'])).long();tt=torch.as_tensor(_as_list(target_meta['task'])).long()
    mp={}
    for i,k in enumerate(zip(ar,al.tolist(),at.tolist())):mp.setdefault((k[0],int(k[1]),int(k[2])),[]).append(i)
    idx=[]
    for r,l,q in zip(tr,tl.tolist(),tt.tolist()):
        z=mp.get((r,int(l),int(q)),[])
        if len(z)!=1:return t,'ROW_META_PRESENT_BUT_NONUNIQUE_ALIGNMENT_FAILED'
        idx.append(z[0])
    return t[torch.tensor(idx,dtype=torch.long)],'ROW_ORDER_ALIGNED_BY_RECIPIENT_LEVEL_TASK'

def load_frozen_e0(path):
    """Load the exact Stage1.21 frozen factual evidence anchor.

    Historical P3 qualified against this E0 tensor.  Stage2.3-B must reuse it
    verbatim; it must never reconstruct an anchor from Stage2.3 global_source.
    """
    x=safe_load(path)
    if not isinstance(x,dict):
        raise TypeError(f'frozen_old_factual_evidence root must be dict, got {type(x)}')
    E0=None
    for k in ('E0','e0','factual_evidence','frozen_factual_evidence'):
        if k in x:
            E0=torch.as_tensor(x[k]).detach().cpu().float();break
    if E0 is None:
        raise KeyError(f'frozen_old_factual_evidence missing E0; keys={list(x.keys())}')
    if E0.ndim!=4 or tuple(E0.shape[1:])!=(5,3,128):
        raise RuntimeError(f'E0 expected [N,5,3,128], got {tuple(E0.shape)}')
    ss=None
    for k in ('serials','source_serial','serial','ids','case_ids'):
        if k in x:
            ss=[_ss(v) for v in _as_list(x[k])];break
    if ss is None:
        raise KeyError(f'frozen_old_factual_evidence missing serials; keys={list(x.keys())}')
    if len(ss)!=len(E0):
        raise RuntimeError(f'E0/serial length mismatch: {len(E0)} vs {len(ss)}')
    if len(set(ss))!=len(ss):
        raise RuntimeError('frozen E0 serials are not unique')
    return E0,ss

def align_e0_to_serials(E0,e0_serials,target_serials):
    mp={_ss(s):i for i,s in enumerate(e0_serials)}
    idx=[];missing=[]
    for s in target_serials:
        q=_ss(s)
        if q not in mp:missing.append(q);idx.append(-1)
        else:idx.append(mp[q])
    if missing:
        raise RuntimeError(f'E0 serial alignment missing {len(missing)} cases; examples={missing[:12]}')
    return E0[torch.tensor(idx,dtype=torch.long)]

def old_level_from_meta(E0,e0_serials,meta):
    rs=[_ss(v) for v in _as_list(meta['recipient_serial'])]
    li=torch.as_tensor(_as_list(meta['level'])).long().reshape(-1)
    aligned=align_e0_to_serials(E0,e0_serials,rs)
    if len(aligned)!=len(li):
        raise RuntimeError('E0 recipient/level row mismatch')
    return torch.stack([aligned[i,int(li[i])] for i in range(len(li))],0).cpu()

def _normalize_full_direction(t,eps=1e-8):
    t=torch.as_tensor(t).float()
    if t.ndim!=3 or tuple(t.shape[1:])!=(3,128):
        raise ValueError(f'P3 direction must be [M,3,128], got {tuple(t.shape)}')
    flat=t.reshape(len(t),-1)
    n=flat.norm(dim=-1,keepdim=True).clamp_min(float(eps))
    return (flat/n).reshape_as(t)

def candidate_deltas_from_artifacts(artifact_candidates,teacher,E0,e0_serials,rows_expected,cfg,max_candidates=120):
    """Recover exact finite-step P3 deltas from frozen production artifacts.

    Priority semantics follow the historical implementation:
      * cf_evidence_level is an already-stepped level endpoint -> delta = CF - E0_level
      * p3_direction is the full 3-task unit direction -> delta = beta*local_scale*normalize(direction)
      * an explicitly named delta tensor can be used directly

    No sign multiplier is applied. Historical Stage1.21 P3 implements
      E* = factual + beta * local_scale * u
    where the orientation is already encoded in u / the recipient->minority pair.
    """
    M=int(rows_expected)
    task=torch.as_tensor(_as_list(teacher['task'])).long()
    old_level=old_level_from_meta(E0,e0_serials,teacher)
    scale=torch.as_tensor(_as_list(teacher['local_scale'])).float().reshape(-1)
    if len(scale)!=M:raise RuntimeError(f'local_scale rows {len(scale)} != {M}')
    beta=float(teacher.get('beta',cfg['p3_fixed_contract']['beta']))
    eps=float(cfg.get('replay',{}).get('direction_eps',1e-8))
    ranked=[]
    for a in artifact_candidates:
        lp=(a['file']+' '+a['path']).lower();score=0
        score+=40*('cf_evidence_level' in lp)+30*('p3_direction' in lp)+22*('counterfactual_evidence' in lp)+18*('counterfactual' in lp)+12*('delta' in lp)+5*('p3' in lp)-20*('random' in lp)
        ranked.append((score,a))
    ranked=sorted(ranked,key=lambda x:-x[0])
    out={};provenance={}
    for _,a in ranked:
        if len(out)>=max_candidates:break
        t,alignment=_reorder_tensor_by_meta(a['tensor'].float(),a.get('row_meta',{}),teacher)
        src=f"artifact::{a['file']}::{a['path']}";lp=src.lower()
        if t.shape==(M,3,128):
            if 'cf_evidence_level' in lp or (('counterfactual_evidence' in lp or 'counterfactual' in lp) and 'direction' not in lp and 'delta' not in lp):
                name=src+'::cf_level_minus_frozen_E0_level'
                out[name]=t-old_level
                provenance[name]={'alignment':alignment,'semantic':'historical_counterfactual_level_minus_frozen_E0_level','formula':'cf_level_minus_frozen_E0_level'}
            if 'p3_direction' in lp or ('direction' in lp and 'pair_bank' not in lp):
                u=_normalize_full_direction(t,eps)
                name=src+'::beta_local_scale_normalized_direction'
                out[name]=beta*scale[:,None,None]*u
                provenance[name]={'alignment':alignment,'semantic':'historical_finite_step_from_full_p3_direction','formula':'beta_local_scale_normalized_direction','beta':beta}
            if 'delta' in lp and 'direction' not in lp:
                name=src+'::direct_full_level_delta'
                out[name]=t
                provenance[name]={'alignment':alignment,'semantic':'explicit_historical_full_level_delta','formula':'direct_full_level_delta'}
        elif t.shape==(M,128):
            # Kept only for diagnostics; never allowed to become the exact production P3.
            z=torch.zeros((M,3,128),dtype=t.dtype);z[torch.arange(M),task]=t
            name=src+'::TARGET_ONLY_DIAGNOSTIC_DIRECT';out[name]=z
            provenance[name]={'alignment':alignment,'semantic':'target_only_diagnostic'}
    return out,provenance

def load_explicit_user_delta(cfg):
    env=cfg['replay'].get('user_delta_env','P3_DELTA_FILE');p=os.environ.get(env,'').strip()
    if not p:return None
    q=Path(p)
    if not q.exists():raise FileNotFoundError(f'{env}={p}')
    x=safe_load(q);cs=recursive_delta_candidates(x)
    return {'source':'user_delta','file':str(q),'candidates':cs}

def snapshot_sources(cfg,outdir):
    dst=Path(outdir)/'p3_source_snapshot';dst.mkdir(parents=True,exist_ok=True);rep=[]
    paths=[cfg.get('stage22c_p3_backend','')]
    for root in (cfg.get('p3_original_source_root',''),cfg.get('stageA_source_root','')):
        p=Path(root)
        if p.exists():paths.extend(sorted(p.rglob('*.py')))
    seen=set()
    for p0 in paths:
        p=Path(p0)
        if not p.exists() or not p.is_file() or str(p.resolve()) in seen:continue
        seen.add(str(p.resolve()));name=f'{len(rep):03d}_{p.name}';shutil.copy2(p,dst/name);text=p.read_text(errors='replace');defs=[]
        try:
            tree=ast.parse(text)
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):defs.append({'name':n.name,'kind':type(n).__name__,'line':getattr(n,'lineno',None)})
        except Exception:pass
        hits=[]
        for i,line in enumerate(text.splitlines()):
            if any(k in line.lower() for k in ('p3_direction','cf_evidence_level','local_scale','p3_lambda','specificity_alpha','beta','majority_to_minority')):
                hits.append({'line':i+1,'text':line[:300]})
                if len(hits)>=120:break
        rep.append({'source':str(p),'snapshot':str(dst/name),'sha256':sha256_file(p),'defs':defs[:240],'keyword_hits':hits})
    return rep

def import_and_inventory(path):
    p=Path(path);rep={'path':str(p),'exists':p.exists(),'callables':[],'import_error':None}
    if not p.exists():return None,rep
    try:
        spec=importlib.util.spec_from_file_location(f'_stage23b_probe_{abs(hash(str(p)))}',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for n,o in vars(mod).items():
            if n.startswith('_'):continue
            if inspect.isfunction(o) or inspect.isclass(o):
                try:sig=str(inspect.signature(o))
                except Exception:sig='?'
                rep['callables'].append({'name':n,'kind':'class' if inspect.isclass(o) else 'function','signature':sig})
        return mod,rep
    except Exception as e:
        rep['import_error']=repr(e);rep['traceback']=traceback.format_exc();return None,rep
