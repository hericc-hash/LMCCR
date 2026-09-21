# Source provenance

| Historical source | Integrated destination | Role |
|---|---|---|
| HR320-v7 | `vision/coordinate_prior.py` | Unchanged coordinate prior |
| 1.16a | `vision/` | Unchanged visual evidence |
| 1.18b-v3 | `apreb/` | Unchanged A/R bottleneck |
| 1.20-v2.1 | `ccmrt/` | Unchanged transport |
| Stage1.21-P3 | `p3/` | Final mechanism search |
| Stage2.3-B v1.7 resolved source | `planner/`, `scripts/planner/` | Final factual Planner and development procedures |
| Stage3-A bridge v1.3 | `bridge/` | Exact Planner config/load/replay and core schema |
| R3.1-v3.2 final patch | `generation/data.py`, `modeling.py`, `scaffold.py`, `train.py` | Final patch implementation |
| R3.1-v3 base / v3.1 helper / sanitizer lineage | Other `generation/` files | Unchanged required helpers, consolidated into one namespace |
| R3.2-S4-v2.1 shared feature source | `selection/features.py` | Shared tabular features only; not a substituted selector |
| R3.2-S4-v2.2 patch | `selection/`, `configs/selector/`, `scripts/selector/` | Exact source/config restored; see SELECTOR_V22_MANIFEST.json |
| Frozen Clinical16 parser | `evaluation/report_metrics.py`, `schema.py` | Supplied source copied byte-for-byte |
| Historical 1.18c and old P-only realizer | `clinical/` | Compatibility support retained; standalone legacy scripts removed |

Paths without a prefix refer to `src/lumbar_cf_report/`. The archive's Markdown is provenance/design material, not a command transcript. Server paths and historical PASS/result claims were not treated as local verification.

See [machine-readable provenance](CURATED_MERGE_MANIFEST.json) for the archive hash and per-file source hashes. Local adapter edits include import consolidation, explicit config arguments, and stricter runtime validation; final file hashes are also recorded. Historical namespaces are not selectable public models. Stage3-A's historical R2.3 runner is not promoted to the final v3.2 generator.
