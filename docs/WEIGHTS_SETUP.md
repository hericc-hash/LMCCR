# Frozen weights and runtime assets

No weights, adapters, patient features or reports are distributed. Obtain authorized local assets and preserve their original hashes/configurations.

The 2026-09-20 weight archive can now be installed locally with `scripts/install_weight_bundle.py`; publication remains undecided. See [installation and validation commands](LOCAL_REPRODUCTION.md). Raw archive names are mapped to the paths below without changing checkpoint bytes.

| Local location | Contents |
|---|---|
| `weights/coordinate_prior/` | HR320-v7 coordinate prior |
| `weights/visual_experts/` | Disc, dual-view Stenosis and dual-view Nerve checkpoints |
| `weights/visual_evidence/` | Frozen visual experts |
| `weights/apreb/` | APREB checkpoint |
| `weights/ccmrt/` | Frozen CCMRT and exact source/config snapshot required by P3 |
| `weights/clinical_planner/` | Selected Stage2.3-B v1.7 checkpoint, config_used and private replay assets |
| `weights/direct_auxiliary_planner/` | Historical auxiliary Planner; not a substitute for Stage2.3-B v1.7 |
| `weights/third_party/Qwen3_4B_Instruct_2507/` | Authorized Qwen3 base model |
| `weights/report_realizer/language_adapter/` | Frozen language adapter |
| `weights/report_realizer/dual_channel_adapter/` | Frozen dual-channel adapter |
| `weights/report_realizer/scaffold_adapter/` | Final v3.2 scaffold adapter |
| `weights/selector/` | v2.2 seed20260931 pair and frozen profile; two additional seed pairs still missing |

The installed checkpoint SHA256 values match the source expectations: `f9cdcb099e7e19a8a9c32eca2118bf16d5d62fddfbc8f444a029120caa8de210` (CCMRT) and `2a58bcbc2023151174c7461286033b5fc2c67960b52ddbd8b3c8b313f6c1ee2d` (Planner). All 23 archive payloads were verified. Original manifests and the installation receipt are kept locally under `weights/_packages/LMCCR_OPEN_SOURCE_WEIGHTS_20260920/`.

The original frozen downstream configs are still needed. `paper_pipeline.json` is a workflow manifest, not a substitute for their exact hyperparameters. Do not use legacy operating points with the final Planner.
