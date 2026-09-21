#!/usr/bin/env python3
"""Validate the public repository and optionally its local, non-redistributed assets."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


CODE_PATHS = [
    "src/lumbar_cf_report/vision",
    "src/lumbar_cf_report/apreb",
    "src/lumbar_cf_report/ccmrt",
    "src/lumbar_cf_report/clinical",
    "src/lumbar_cf_report/planner",
    "src/lumbar_cf_report/p3",
    "src/lumbar_cf_report/bridge",
    "src/lumbar_cf_report/generation",
    "src/lumbar_cf_report/evaluation",
    "src/lumbar_cf_report/selection/reranker.py",
    "src/lumbar_cf_report/selection/text_embedder.py",
    "configs/selector/default.json",
    "configs/selector/profile.json",
    "configs/paper_pipeline.json",
    "configs/ccmrt.yaml",
]

ASSET_PATHS = {
    "coordinate prior": "weights/coordinate_prior/hr320_v7.pt",
    "visual evidence": "weights/visual_evidence/lumbar_visual_evidence.pt",
    "APREB": "weights/apreb/apreb.pt",
    "CCMRT": "weights/ccmrt/ccmrt.pt",
    "clinical planner": "weights/clinical_planner/05_stage2_3B_selected_candidate.pt",
    "exact planner config": "weights/clinical_planner/config_used.json",
    "Qwen3": "weights/third_party/Qwen3_4B_Instruct_2507/config.json",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true", help="skip private and licensed asset checks")
    parser.add_argument("--final", action="store_true", help="require a complete final frozen workflow")
    args = parser.parse_args()
    failures = []
    for relative in CODE_PATHS:
        if not (ROOT / relative).exists():
            failures.append(f"missing code path: {relative}")
    for module in ("lumbar_cf_report", "lumbar_cf_report.apreb", "lumbar_cf_report.ccmrt", "lumbar_cf_report.clinical",
                   "lumbar_cf_report.planner", "lumbar_cf_report.p3.pipeline", "lumbar_cf_report.bridge.inference",
                   "lumbar_cf_report.generation.generate", "lumbar_cf_report.selection", "lumbar_cf_report.evaluation.report_metrics"):
        try:
            importlib.import_module(module)
        except Exception as error:
            failures.append(f"cannot import {module}: {error}")
    try:
        json.loads((ROOT / "configs/paper_pipeline.json").read_text(encoding="utf-8"))
    except Exception as error:
        failures.append(f"invalid paper_pipeline.json: {error}")
    if not args.code_only:
        for label, relative in ASSET_PATHS.items():
            if not (ROOT / relative).exists():
                failures.append(f"missing local asset ({label}): {relative}")
    if args.final or not args.code_only:
        failures.append("full Planner/realizer training configs and private runtime assets still require validation; see docs/MERGE_STATUS.md")
    if failures:
        print("Setup check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Integrated code layout, including exact selector v2.2, is valid; this is not a real-weight end-to-end validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
