# Paper-to-code map

| Concept | Implementation / entry |
|---|---|
| Frozen visual frontend | `vision/` |
| A/R evidence bottleneck | `apreb/`; `scripts/train_apreb.py` |
| Matched residual transport | `ccmrt/`; `scripts/train_ccmrt.py` |
| P3 mechanism qualification | `p3/`; `scripts/run_p3.py` |
| Final factual Planner | `planner/model.py`; `scripts/planner/` |
| Exact config and checkpoint replay | `bridge/stage23_exact.py` |
| Raw-E → Clinical16 | `bridge/inference.py`; `scripts/predict_clinical16.py` |
| Core-redacted scaffold | `generation/scaffold.py`, `prompting.py` |
| Final v3.2 realizer | `generation/modeling.py`, `train.py`, `generate.py` |
| Precision repair | `generation/precision_patch.py` |
| Realizer component CLI | `scripts/generate_scaffold_report.py` |
| FactNet + TextListwise v2.2 | `selection/reranker.py`, `text_embedder.py`, `selector.py`; `scripts/run_selector_workflow.py` |
| Shared inference features | `selection/features.py`, `firewall.py` |
| Clinical16 report evaluation | `evaluation/report_metrics.py` |
| Bootstrap | `evaluation/bootstrap.py` |

Package paths are relative to `src/lumbar_cf_report/`. The standalone legacy scripts have been removed; compatibility modules required by replay remain. See [merge limitations](MERGE_STATUS.md).
