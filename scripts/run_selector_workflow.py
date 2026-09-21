"""Run exact v2.2 research procedures with configurable paths and holdout gating."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/selector/default.json'))
    p.add_argument('--out', required=True, help='Private output directory; use the same directory to resume')
    p.add_argument('--step', default='all', choices=['all']+[f'{i:02}' for i in range(12)])
    a = p.parse_args()
    env = dict(os.environ, LMCCR_SELECTOR_CONFIG=str(Path(a.config).resolve()), OUT=str(Path(a.out).resolve()))
    for i in range(12):
        if a.step != 'all' and a.step != f'{i:02}':
            continue
        if 7 <= i <= 9 and not (Path(env['OUT'])/'07_selection_gate/SELECTION_PASS.flag').exists():
            if a.step != 'all':
                p.error('Selection50 gate has not passed; holdout remains locked')
            continue
        script = next((ROOT/'scripts/selector').glob(f'{i:02}_*.py'))
        subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
