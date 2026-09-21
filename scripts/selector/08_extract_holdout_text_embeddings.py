from __future__ import annotations
import gc,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,read_jsonl,sha256,dump_json
from lumbar_cf_report.selection.modeling import load_v32
from lumbar_cf_report.selection.text_embedder import encode_pool,save_embedding_cache

def main():
    c=load_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));o=Path(os.environ['OUT'])
    if not (o/'07_selection_gate/SELECTION_PASS.flag').exists():raise RuntimeError('Selection50 did not pass; holdout embedding forbidden')
    pool=o/'08_holdout_pool/HOLDOUT50_S4_DEPLOYABLE_POOL.jsonl';npz=o/'08_holdout_pool/HOLDOUT50_TEXT_EMBEDDINGS.npz';meta=npz.with_suffix('.json')
    if npz.exists() and meta.exists():
        m=load_json(meta)
        if m.get('pool_sha256')==sha256(pool) and m.get('encoder_config')==c['text_encoder']:
            print(json.dumps({'status':'PASS','reused':True,**m},ensure_ascii=False,indent=2));return
    pf=load_json(o/'00_PREFLIGHT.json');tok,model=load_v32(c['assets']['base_model'],pf['language_adapter'],pf['r31v1_adapter'],pf['scaffold_adapter'],c['reranker']['device']);rows=read_jsonl(pool);cache=encode_pool(tok,model,rows,c,c['reranker']['device']);m={'status':'PASS','split':'Internal50-holdout','pool_sha256':sha256(pool),'cases':len(rows),'candidates':int(len(cache['serial'])),'embedding_dim':int(cache['embedding'].shape[1]),'encoder_config':c['text_encoder'],'reference_or_gt_used':False,'generator_calls':0};save_embedding_cache(cache,npz,m);print(json.dumps(m,ensure_ascii=False,indent=2));del model,tok;gc.collect()
    try:
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    except Exception:pass
if __name__=='__main__':main()
