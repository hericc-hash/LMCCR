# Reproducing the frozen workflow

The integrated core is testable, but a complete paper reproduction is currently blocked on the missing complete Planner/realizer run configurations and private assets. See [merge status](MERGE_STATUS.md). Do not infer hyperparameters or use old selector variants.

## Code checks

From the repository root, install `pip install -e ".[dev]"`, then run:

```bash
python scripts/check_setup.py --code-only
python -m pytest
python -m compileall -q src scripts tests
```

## Upstream and P3

The APREB/CCMRT training entries are preserved. Upstream visual feature export still requires the original authorized data/weights. P3 retains the supplied scientific parameters; only paths in `configs/p3.json` were made relative. It requires the original frozen CCMRT source snapshot, weights, pair bank, Jacobian cache, and frozen selection manifests.

```bash
python scripts/run_p3.py preflight --config configs/p3.json
python scripts/run_p3.py smoke --config configs/p3.json
python scripts/run_p3.py run --config configs/p3.json
python scripts/run_p3.py select --config configs/p3.json
```

P3 is not a factual Raw-E replacement. Selection artifacts referenced by this scientific replay remain private local inputs; no reference-only results were copied into the repository.

## Final Planner

`scripts/planner/` preserves the supplied preparation, CF training, context selection, nerve-head recovery and evaluation procedures. Each requires `--config` pointing to the original complete v1.7 config and `--out`. The freeze omitted that config, so those commands are not documented as immediately runnable training recipes.

For inference, provide the frozen v1.7 output directory with `config_used.json`, `05_stage2_3B_selected_candidate.pt`, `01_stage23b_data.pt` and `07_internal_factual_probs.pt`. The latter two are private replay assets. The CLI strictly loads and checks saved probabilities before inference:

```bash
python scripts/predict_clinical16.py --runtime weights/clinical_planner --input data/derived/raw_e_batch.pt --thresholds configs/local_thresholds.json --output outputs/clinical16.pt
```

The input mapping needs `global_source`, `raw_E`, `coordinate_state`, `task_quality`, `task_valid`, `lordosis_valid`. The threshold JSON must contain `source_cohort: "Development388"` and the exact frozen `thresholds` vector of length 16. `configs/operating_points.json` belongs to the old compatibility chain and must not be reused here.

## Realizer component

Provide a frozen Direct draft, Clinical16, the calibrated lexicalizer and the exact v3.2 configuration. `generate_scaffold_report.py` accepts private JSONL rows with `direct_draft`, `core_binary`, `slot_valid` and emits raw/precision-patched report pairs. Its config requires `paths.base_model`, `paths.language_adapter`, `paths.dual_channel_adapter`, `paths.scaffold_adapter`, `device`, `generation`, `scaffold_channel`, `precision_patch`. This wrapper is a new portable interface; its field layout does not claim to be the missing original config file.

```bash
python scripts/generate_scaffold_report.py --config configs/local_realizer.json --input data/derived/core_and_direct.jsonl --lexicon data/derived/frozen_lexicon.json --output outputs/realizer_component.jsonl
```

This is not full S4 candidate generation or final v2.2 selection. The exact original generator/selector config is required to complete that workflow. Do not label this intermediate output as the final selected report.

## Frozen selection and evaluation

The exact v2.2 package is integrated. Run `python scripts/run_selector_workflow.py --config configs/selector/default.json --out outputs/selector --step 00` for preflight after configuring local assets. `configs/selector/frozen_source.json` is a provenance snapshot with historical server paths, not the portable runtime config. Training/profile selection uses Development388 OOF only; evaluate the Internal50-selection gate before Internal50-holdout. Independent49 is reserved for frozen confirmation, never selection/tuning. Record code revision, environment, data and weight hashes for each real run.
