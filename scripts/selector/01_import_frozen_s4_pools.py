from __future__ import annotations
import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,dump_json,read_jsonl
from lumbar_cf_report.selection.source_s4 import copy_required_source

def main():
    o=Path(os.environ['OUT']);pf=load_json(o/'00_PREFLIGHT.json');source=Path(pf['source_s4_out']);copied=copy_required_source(source,o)
    dev=read_jsonl(o/'03_development_candidates/DEV388_S4_CANDIDATE_POOL.jsonl');sel=read_jsonl(o/'05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl')
    rep={'status':'PASS','source_s4_out':str(source),'development_cases':len(dev),'selection_cases':len(sel),'copied_files':copied,'generator_calls':0,'candidate_text_changed':False,'note':'Exact frozen S4 pools are reused. v2.2 changes selector representation/training only; candidate text is unchanged.'}
    dump_json(rep,o/'01_import/IMPORT_REPORT.json');print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
