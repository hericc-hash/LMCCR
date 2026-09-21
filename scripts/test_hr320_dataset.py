"""Run real-image HR320 checks, one stage of full reproduction readiness."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    import torch
    from lumbar_cf_report.vision.hr320_runtime import run_prior_smoke
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--checkpoint', default='weights/coordinate_prior/hr320_v7.pt')
    p.add_argument('--output', default='outputs/hr320_smoke')
    p.add_argument('--cohort', default='Development388', choices=['Development388','Internal100','Independent49'])
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.threads < 1:
        p.error('--threads must be positive')
    torch.set_num_threads(a.threads)
    result = run_prior_smoke(a.dataset_root, a.checkpoint, a.output, a.cohort, a.limit, a.device)
    print(f"HR320 stage {result['status']}: {result['passed']}/{result['requested']}; full pipeline not implied.")
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
