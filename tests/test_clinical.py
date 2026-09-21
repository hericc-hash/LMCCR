import json
from pathlib import Path

import numpy as np
import torch

from lumbar_cf_report.clinical.operating_points import binarize_main, load_thresholds
from lumbar_cf_report.clinical.planner import ExplicitClinicalPlanner
from lumbar_cf_report.clinical.schema import MAIN_SLOT_NAMES, PLAN_NODE_NAMES


ROOT = Path(__file__).resolve().parents[1]


def test_operating_points_and_slots():
    thresholds, payload = load_thresholds(ROOT / "configs" / "operating_points.json")
    assert len(MAIN_SLOT_NAMES) == 16
    assert thresholds["nerve"] == 0.28
    prediction = binarize_main(np.full((1, 16), 0.4), thresholds)
    assert prediction.shape == (1, 16)
    assert prediction[0, MAIN_SLOT_NAMES.index("nerve:L1/2")] == 1


def test_explicit_planner_contract():
    planner = ExplicitClinicalPlanner(10, 8, 7, hidden_size=24, projector_hidden_dim=16, plan_state_hidden=12, dropout=0.0)
    output = planner(
        torch.randn(2, 10),
        torch.randn(2, 5, 3, 8),
        torch.randn(2, 5, 7),
        torch.ones(2, 5, 3),
        torch.ones(2, 5, 3),
    )
    assert output["main_probabilities"].shape == (2, len(MAIN_SLOT_NAMES))
    assert output["plan_tokens"].shape == (2, len(PLAN_NODE_NAMES), 24)


def test_pipeline_config_is_relative_and_result_free():
    config = json.loads((ROOT / "configs" / "paper_pipeline.json").read_text(encoding="utf-8"))
    assert not any(str(value).startswith("/root/") for value in config["paths"].values() if isinstance(value, str))
    assert config["paths"]["output_root"] == "outputs"
