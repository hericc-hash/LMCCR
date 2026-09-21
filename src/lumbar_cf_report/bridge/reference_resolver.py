from __future__ import annotations
from pathlib import Path
import json

REFERENCE_KEYS=(
    'reference_raw','reference','target_report','ground_truth_report','gt_report',
    'target','ground_truth','report_reference','reference_report'
)

def _load_jsonl(path):
    rows=[]
    with Path(path).open('r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def _serial(x):
    return str(x.get('serial')) if isinstance(x,dict) and x.get('serial') is not None else None

def _extract_ref(row):
    if not isinstance(row,dict): return ''
    for k in REFERENCE_KEYS:
        v=row.get(k)
        if isinstance(v,str) and v.strip(): return v.strip()
    return ''

def recover_frozen_references(plans, frozen_r23_config):
    """Recover references only from explicit frozen artifacts; never synthesize or infer them.
    Generation never depends on this. The returned references are only for optional language metrics.
    """
    by_serial={str(r['serial']): (r.get('reference') or '').strip() for r in plans}
    provenance={s:('mediation' if ref else None) for s,ref in by_serial.items()}
    candidates=[]
    p=frozen_r23_config.get('internal100_v5_jsonl')
    if p: candidates.append(('frozen_r23_internal100_v5_jsonl',Path(p)))
    diagnostics=[]
    for source_name,path in candidates:
        if not path.exists():
            diagnostics.append({'source':source_name,'path':str(path),'exists':False,'usable_refs':0})
            continue
        rows=_load_jsonl(path); usable=0
        for row in rows:
            s=_serial(row); ref=_extract_ref(row)
            if s is not None and ref:
                usable+=1
                if s in by_serial and not by_serial[s]:
                    by_serial[s]=ref; provenance[s]=source_name
        diagnostics.append({'source':source_name,'path':str(path),'exists':True,'n_rows':len(rows),'usable_refs':usable})
    coverage=sum(bool(v) for v in by_serial.values())
    return by_serial,provenance,{'coverage':coverage,'n':len(by_serial),'complete':coverage==len(by_serial),'sources':diagnostics,'policy':'frozen_explicit_sources_only_no_guessing'}
