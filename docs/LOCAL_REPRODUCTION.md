# Local weights and reproduction status

## Scope

Latest update: the runtime supplement has been integrated and one actual Development case passed automatic slice selection, HR320, axial mapping and all three visual experts. See [the single-case report](SINGLE_CASE_RUNTIME_GAPS_RESULT.md). The original weight-only findings below are superseded where noted.

The 2026-09-20 bundle is integrated as **local-only assets**. No weights have been staged, committed, uploaded or approved for public release. Public release requires a separate data/performance and redistribution review. This document reports engineering readiness, not paper-level reproduction or clinical validation.

## Installation

From the repository root, using Python >=3.10:

```bash
python -m venv .venv
# Activate the environment using your operating system's command.
python -m pip install -e ".[dev,dicom,generation]"
python scripts/install_weight_bundle.py --archive /path/to/LMCCR_OPEN_SOURCE_WEIGHTS_20260920.tar.gz
```

The installer rejects path traversal, links, duplicate destinations, unlisted payloads, incorrect checksums and conflicting existing files. It verifies the entire bundle before copying runtime files. All 23 payload hashes in the supplied archive match. Optimizer state and checkpoint metadata are preserved byte-for-byte; installation is not a sanitized release export.

Copy `configs/reproduction.example.json` to `configs/local_assets.json`. Set `dataset_root` to the de-identified dataset and `base_model` to the local Qwen3-4B-Instruct-2507 directory. Other paths resolve relative to the repository, independent of the shell working directory for the weight validator. The base model remains in place; no download or redundant copy is required.

The three adapters are installed as `report_realizer/language_adapter`, `dual_channel_adapter`, and `scaffold_adapter`, preserving the original merge order. The final Planner is installed under the filename expected by the existing bridge. The P3 config points to the verified CCMRT checkpoint at `weights/ccmrt/ccmrt.pt`.

## Validation commands

```bash
python scripts/validate_weight_bundle.py --config configs/local_assets.json
python scripts/validate_weight_bundle.py --config configs/local_assets.json --require-full-pipeline
python scripts/test_hr320_dataset.py --dataset-root /path/to/LumbarMRI_CF_Report_Deidentified_Dataset_v1 --limit 8
python -m pytest
```

Weight validation rechecks installed file hashes, strictly loads available PyTorch model states, compares the selector profile, and checks the local Qwen tokenizer plus LoRA tensor shapes against every indexed base-model shard. It does **not** equate compatible checkpoints with full inference. Exit 0 means weight checks passed; `--require-full-pipeline` returns 2 when pipeline prerequisites are incomplete. Any weight-check failure returns 1. Detailed local results go to `outputs/weight_validation/weight_validation.json`.

HR320 has executed successfully on eight actual Development images with the supplied weight. That test uses the annotated center slice, reproduces percentile normalization and 320-square resizing, and outputs six raw-frame landmarks plus five canonical downstream coordinates. Landmark coordinates are read only after inference for diagnostics. It skips cases without a center annotation and explicitly reports that count; it is not an automatic slice-selection benchmark or held-out accuracy estimate. Predictions stay in the Git-ignored outputs directory.

## Remaining full-pipeline prerequisites

The local integration regression suite passes **43 tests**, including the existing frozen-source/selector contracts and new archive integrity, conflicting-weight protection, reserved metadata and Windows path protection, preprocessing and orientation checks. Syntax compilation also passes. The installed Python package versions are recorded locally in `outputs/weight_validation/environment.txt`.

- All three selector seed pairs are now installed and strictly loaded. Actual candidate selection on the patient remains unexecuted.
- The original Planner `config_used.json` is installed and verified. The two historical replay tensors are not installed; a separate local historical archive was discovered, but its tensors were not substituted for this patient's image-derived inputs.
- Automatic sagittal selection, HR320, axial mappings and the three experts now execute on one case. Full Raw-E/global features and normalized 47-D coordinates still require the frozen lordosis geometry constants and coordinate normalization statistics.
- Direct source/configuration and the realizer lexicalizer are now installed. The selected Direct epoch_2 prefix and its independent LoRA weights remain missing. The auxiliary Planner and three realizer adapters cannot replace them.
- P3 mechanism replay additionally requires the frozen source snapshot, pair/feature banks, empirical orientation selection, frozen evidence and protected Jacobian cache. This mechanism branch is distinct from ordinary factual inference.

The next data evaluation must preserve cohort boundaries and clearly distinguish component execution, factual MRI-to-report inference, and mechanism replay. No dataset labels or reference reports should be used as missing inference features or synthetic substitutes for absent assets.

The runtime-supplement regression run passed 48 tests; the subsequent minimal-statistics exporter passed two additional targeted tests (50 tests total). Compilation passed. The diagnostics intentionally return a nonzero exit code while factual execution is incomplete.
