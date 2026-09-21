from __future__ import annotations
import gc, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,read_jsonl,sha256,dump_json
from lumbar_cf_report.selection.modeling import load_v32
from lumbar_cf_report.selection.text_embedder import encode_pool,save_embedding_cache


def _encoder_id(pre,cfg):
    return {'base_model':cfg['assets']['base_model'],'language_adapter':pre.get('language_adapter'),'r31v1_adapter':pre.get('r31v1_adapter'),'scaffold_adapter':pre.get('scaffold_adapter')}

def _cache_valid(npz_path,meta_path,pool_path,cfg,pre):
    if not npz_path.exists() or not meta_path.exists():return False
    try:m=load_json(meta_path)
    except Exception:return False
    return m.get('pool_sha256')==sha256(pool_path) and m.get('encoder_config')==cfg['text_encoder'] and m.get('encoder_id')==_encoder_id(pre,cfg) and m.get('status')=='PASS'


def _encode_one(tok,model,pool_path,npz_path,cfg,pre,device,split):
    meta_path=npz_path.with_suffix('.json')
    if _cache_valid(npz_path,meta_path,pool_path,cfg,pre):
        print(f'[TextEmbed] reuse {split}: {npz_path}',flush=True);return load_json(meta_path)
    rows=read_jsonl(pool_path);cache=encode_pool(tok,model,rows,cfg,device)
    meta={'status':'PASS','split':split,'pool_path':str(pool_path),'pool_sha256':sha256(pool_path),'cases':len(rows),'candidates':int(len(cache['serial'])),'embedding_dim':int(cache['embedding'].shape[1]),'embedding_dtype':str(cache['embedding'].dtype),'encoder_config':cfg['text_encoder'],'encoder_id':_encoder_id(pre,cfg),'reference_or_gt_used':False,'generator_calls':0}
    save_embedding_cache(cache,npz_path,meta);print(json.dumps(meta,ensure_ascii=False,indent=2));return meta


def main():
    cfg=load_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));out=Path(os.environ['OUT']);pre=load_json(out/'00_PREFLIGHT.json');devpool=out/'03_development_candidates/DEV388_S4_CANDIDATE_POOL.jsonl';selpool=out/'05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl'
    devnpz=out/'02_text_embeddings/DEV388_TEXT_EMBEDDINGS.npz';selnpz=out/'02_text_embeddings/SELECTION50_TEXT_EMBEDDINGS.npz'
    need=not(_cache_valid(devnpz,devnpz.with_suffix('.json'),devpool,cfg,pre) and _cache_valid(selnpz,selnpz.with_suffix('.json'),selpool,cfg,pre))
    if not need:
        rep={'status':'PASS','reused_all':True,'development':load_json(devnpz.with_suffix('.json')),'selection':load_json(selnpz.with_suffix('.json'))};dump_json(rep,out/'02_text_embeddings/EMBEDDING_COMPLETE.json');print(json.dumps(rep,ensure_ascii=False,indent=2));return
    tok,model=load_v32(cfg['assets']['base_model'],pre['language_adapter'],pre['r31v1_adapter'],pre['scaffold_adapter'],cfg['reranker']['device'])
    devmeta=_encode_one(tok,model,devpool,devnpz,cfg,pre,cfg['reranker']['device'],'Development388')
    selmeta=_encode_one(tok,model,selpool,selnpz,cfg,pre,cfg['reranker']['device'],'Internal50-selection')
    rep={'status':'PASS','reused_all':False,'development':devmeta,'selection':selmeta,'frozen_encoder':'v3.2 Qwen3 hidden states; no generation; no reference/GT'};dump_json(rep,out/'02_text_embeddings/EMBEDDING_COMPLETE.json');print(json.dumps(rep,ensure_ascii=False,indent=2))
    del model,tok;gc.collect()
    try:
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    except Exception:pass
if __name__=='__main__':main()
