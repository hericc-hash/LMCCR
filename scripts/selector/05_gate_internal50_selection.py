from __future__ import annotations
import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.selection.io import load_json,dump_json,read_jsonl,write_jsonl
from lumbar_cf_report.selection.eval_bridge import load_parser
from lumbar_cf_report.selection.metrics import eval_texts,parsed_vec

def main():
 c=load_json(Path(os.environ.get('LMCCR_SELECTOR_CONFIG', str(ROOT/'configs/selector/default.json'))));o=Path(os.environ['OUT']);flag=o/'07_selection_gate/SELECTION_PASS.flag';flag.unlink() if flag.exists() else None;pred=read_jsonl(o/'06_selection/SELECTED_PREDICTIONS.jsonl');labels={str(r['serial']):r for r in read_jsonl(o/'05_selection_pool/SELECTION50_EVAL_LABELS.jsonl')};mod=load_parser(c['parser']['module']);rows=[]
 for p in pred:rows.append({**p,**labels[str(p['serial'])]})
 texts=[r['generated'] for r in rows];refs=[r['reference_raw'] for r in rows];gt=[r['slot_labels'] for r in rows];valid=[r['slot_valid'] for r in rows];coremap={str(r['serial']):r['core_binary'] for r in read_jsonl(o/'05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl')};core=[coremap[str(r['serial'])] for r in rows];m=eval_texts(mod,texts,refs,gt,valid,core);parse_rate=sum(1 for t in texts if len(parsed_vec(mod,t))==16)/len(texts);g=c['selection_gate'];checks={'clinical_f1':m['clinical_f1']>=g['minimum_clinical_f1'],'rouge_l':m['rouge_l']>=g['minimum_rouge_l'],'bleu4':m['bleu4']>=g['minimum_bleu4'],'parse_rate':parse_rate>=g['minimum_parse_rate']};rep={'pass':all(checks.values()),'checks':checks,'metrics':m,'parse_rate':parse_rate,'hard_target':g,'comparators':c['comparators'],'adaptive_pool':load_json(o/'05_selection_pool/POOL_SUMMARY.json')};dump_json(rep,o/'07_selection_gate/SELECTION_GATE.json');write_jsonl(rows,o/'07_selection_gate/SELECTED_WITH_EVAL_FIELDS.jsonl');print(json.dumps(rep,ensure_ascii=False,indent=2))
 if rep['pass']:(o/'07_selection_gate/SELECTION_PASS.flag').write_text('PASS\n')
if __name__=='__main__':main()
