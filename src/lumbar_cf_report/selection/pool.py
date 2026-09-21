from __future__ import annotations
from .scaffold import build_scaffold
from .eval_bridge import parse,normalize
from .metrics import parsed_vec,case_f1,case_acc,rouge_one,bleu_one,case_counts
from .precision_patch import precision_finalize

def prepare_row(row,parser_mod,scaffold_cfg):
    q=dict(row);sc,meta=build_scaffold(q['direct_draft'],parser_mod,parse,normalize,scaffold_cfg);q['linguistic_scaffold']=sc;q['scaffold_meta']=meta;q['direct_parser_vec']=parsed_vec(parser_mod,q['direct_draft']).tolist();return q

def normalize_tag(tag):return {'existing_raw':'greedy','existing_final':'greedy_final'}.get(str(tag),str(tag))

def basic_candidate(tag,text,parser_mod):return {'tag':normalize_tag(tag),'text':str(text).strip(),'parser_vec':parsed_vec(parser_mod,str(text)).tolist()}

def enrich_candidates(row,pool,parser_mod,with_targets=True):
    out=[];seen=set();ref=row.get('reference_raw','');gt=row.get('slot_labels');valid=row['slot_valid']
    for tag,text in pool:
        text=str(text).strip();tag=normalize_tag(tag)
        if not text or text in seen:continue
        seen.add(text);pv=parsed_vec(parser_mod,text).tolist();rec={'tag':tag,'text':text,'parser_vec':pv}
        if with_targets:
            tp,fp,fn,tn=case_counts(pv,gt,valid);rec.update({'target_fact_f1':case_f1(pv,gt,valid),'target_slot_acc':case_acc(pv,gt,valid),'rouge':rouge_one(parser_mod,text,ref),'bleu':bleu_one(text,ref),'tp':tp,'fp':fp,'fn':fn,'gt_pos':tp+fn})
        out.append(rec)
    return out

def add_greedy_final(row,pool,lex,parser_mod,patch_cfg):
    g=next((text for tag,text in pool if normalize_tag(tag)=='greedy'),None)
    if not g:return pool,None
    final,meta=precision_finalize(g,row['core_binary'],row['slot_valid'],lex,parser_mod,parse,normalize,patch_cfg)
    if final.strip() and final.strip()!=g.strip():return list(pool)+[('greedy_final',final)],meta
    return list(pool),meta
