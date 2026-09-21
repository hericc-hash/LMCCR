#!/usr/bin/env python3
"""Generate a v3.2 raw/precision-patched pair from a frozen Direct draft and Clinical16.

This is the realizer component, not final S4 neural selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from lumbar_cf_report.generation.generate import load_model, generate_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Exact v3.2 runtime config restored from the original run')
    parser.add_argument('--input', required=True, help='Private JSONL: core_binary, slot_valid, direct_draft')
    parser.add_argument('--lexicon', required=True, help='Frozen parser-calibrated precision lexicon JSON')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    paths = cfg['paths']
    lex = json.loads(Path(args.lexicon).read_text(encoding='utf-8'))
    tok, model = load_model(paths['base_model'], paths['language_adapter'], paths['dual_channel_adapter'],
                            paths['scaffold_adapter'], cfg['device'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding='utf-8') as inp, open(args.output, 'w', encoding='utf-8') as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            result = generate_one(tok, model, row, cfg['generation'], lex,
                                  'lumbar_cf_report.evaluation.report_metrics',
                                  cfg['scaffold_channel'], cfg['precision_patch'])
            out.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
