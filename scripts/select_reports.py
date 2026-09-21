"""Run the frozen v2.2 ensemble on an existing S4 pool and matching embedding cache."""
import argparse
from lumbar_cf_report.selection import FrozenSelector
from lumbar_cf_report.selection.io import load_json, read_jsonl, write_jsonl
from lumbar_cf_report.selection.text_embedder import load_embedding_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--fact', action='append', required=True, help='Repeat for every frozen FactNet seed')
    p.add_argument('--text', action='append', required=True, help='Repeat for every frozen TextListwise seed')
    p.add_argument('--profile', default='configs/selector/profile.json')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    selector = FrozenSelector.from_paths(a.fact, a.text, load_json(a.profile), a.device)
    write_jsonl(selector.select(read_jsonl(a.pool), load_embedding_cache(a.embeddings)), a.output)


if __name__ == '__main__':
    main()
