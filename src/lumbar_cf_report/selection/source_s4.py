from __future__ import annotations
import glob,os,shutil
from pathlib import Path


def discover_source(cfg):
    env_name=cfg.get('env_override','R32S4V22_SOURCE_OUT');env=os.environ.get(env_name,'').strip();cands=[]
    if env:cands.append(Path(env))
    for pat in cfg.get('output_prefixes',[]):cands.extend(Path(x) for x in glob.glob(pat))
    good=[]
    for p in cands:
        if not p.is_dir():continue
        if all((p/r).exists() for r in cfg.get('required_files',[])):good.append(p.resolve())
    if not good:
        raise FileNotFoundError('No complete frozen S4 source OUT found. Set '+env_name+' to a complete S4-v2 or S4-v2.1 output directory.')
    good.sort(key=lambda p:p.stat().st_mtime,reverse=True);return good[0]

def copy_required_source(source,out):
    source=Path(source);out=Path(out)
    mapping={
      '03_development_candidates/DEV388_S4_CANDIDATE_POOL.jsonl':'03_development_candidates/DEV388_S4_CANDIDATE_POOL.jsonl',
      '03_development_candidates/DEV_POOL_SUMMARY.json':'03_development_candidates/DEV_POOL_SUMMARY.json',
      '05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl':'05_selection_pool/SELECTION50_S4_DEPLOYABLE_POOL.jsonl',
      '05_selection_pool/SELECTION50_EVAL_LABELS.jsonl':'05_selection_pool/SELECTION50_EVAL_LABELS.jsonl',
      '05_selection_pool/POOL_SUMMARY.json':'05_selection_pool/POOL_SUMMARY.json',
      '01_development/CANONICAL_LEXICALIZER.json':'01_development/CANONICAL_LEXICALIZER.json'
    }
    optional={
      '06_selection_oracle_audit/SELECTION50_S4_ORACLE_CEILING.json':'00_source/SELECTION50_S4_ORACLE_CEILING_SOURCE.json',
      '07_selection_gate/SELECTION_GATE.json':'00_source/S4_V2_SELECTION_GATE_SOURCE.json',
      '04_reranker/OOF_CV_REPORT.json':'00_source/S4_V2_OOF_CV_REPORT_SOURCE.json',
      '10_FINAL_R32_S4_DECISION.json':'00_source/S4_V2_FINAL_DECISION_SOURCE.json',
      '10_FINAL_R32_S4_V21_DECISION.json':'00_source/S4_V21_FINAL_DECISION_SOURCE.json'
    }
    copied=[]
    for src_rel,dst_rel in mapping.items():
        src=source/src_rel;dst=out/dst_rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);copied.append(dst_rel)
    for src_rel,dst_rel in optional.items():
        src=source/src_rel
        if src.exists():
            dst=out/dst_rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);copied.append(dst_rel)
    return copied
