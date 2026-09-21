#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from lumbar_cf_report.planner.io import dumpj
from lumbar_cf_report.planner.data import build_split
from lumbar_cf_report.planner.phasea import find_phasea

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,help='Exact Stage2.3-v1.7 training config (not included in the freeze)' );p.add_argument('--out',required=True);a=p.parse_args();cfg=json.load(open(a.config));out=Path(a.out);pa=find_phasea(cfg)
 dev=build_split(cfg['development_mediation'],cfg['feature_bank']);internal=build_split(cfg['internal_mediation'],cfg['feature_bank'])
 torch.save({'development':dev,'internal':internal,'phaseA_candidate_path':pa['candidate'],'phaseA_candidate_sha256':pa['candidate_sha256']},out/'01_stage23b_data.pt')
 rep={s:{'cases':len(d['serials']),'raw_E_shape':list(d['raw_E'].shape),'valid_counts':(d['label_valid']*d['task_valid']).sum((0,1)).tolist()} for s,d in [('Development388',dev),('Internal100',internal)]};dumpj(rep,out/'01_data_summary.json');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
