# Data setup

The public repository does not provide MRI images, clinical reports, serial-number mappings, or feature tensors. Use only de-identified data covered by your institutional approvals and data-use agreements.

## Expected files

Place private derived artifacts under `data/derived/`:

```text
development_feature_bank.pt
independent49_feature_bank.pt
train_mediation.pt
internal100_mediation.pt
independent49_mediation.pt
internal100_ccmrt_handoff.pt
internal100_direct_handoff.pt
independent49_ccmrt_handoff.pt
internal100_ccmrt_selection.jsonl
internal100_direct_selection.jsonl
independent49_ccmrt_selection.jsonl
```

These are historical upstream/mediation artifacts. Paths for those tools belong to `configs/ccmrt.yaml` and explicit local replay configurations, not the final workflow manifest.

## Final factual inference inputs

The final Planner consumes `global_source` `[N,256]`, `raw_E` `[N,5,3,128]`, `coordinate_state` `[N,5,47]`, `task_quality` and `task_valid` `[N,5,3]`. The CLI also requires `lordosis_valid` `[N]`, an input-availability mask, and explicit Development-frozen thresholds `[16]`. Reconstructed Ehat is not substituted for Raw-E.

Generation consumes a frozen `direct_draft`, `core_binary` and `slot_valid`; labels and reference text are not generation inputs. Exact Planner replay needs the private `01_stage23b_data.pt` and `07_internal_factual_probs.pt` from the frozen output directory. P3 needs original X/E/C features, pairs, frozen E0, Jacobians and frozen selection metadata, at the relative paths in `configs/p3.json`.

The v2.2 neural source is integrated. Full reproduction still needs missing frozen runtime assets and complete selector ensembles; see LOCAL_REPRODUCTION.md. Patient reports and embedding caches remain private.

## Core tensor contracts

The mediation cohort contains `serials`, `segment_base`, `task_features`, `coordinate_state`, `task_quality`, `task_valid`, disease `labels`/`label_valid`, `global_source`, planner `slot_labels`/`slot_valid`, structure flags, and focused report targets. Expected leading shapes are five lumbar levels and three tasks. The default coordinate state has 47 dimensions, global source 256 dimensions, and task evidence 128 dimensions.

The CCMRT handoff contains one record per recipient, level, and task. Each record includes recipient/donor identifiers, labels, factual evidence `[5,3,128]`, and counterfactual evidence `[5,3,128]`. Every recipient must have 15 records.

Selection manifests contain `recipient_serial`, `level_index`, and `task_index` at minimum. Internal comparisons must use matched selections for CCMRT and the direct baseline.

## Leakage boundary

Final training, threshold selection and OOF profile selection use Development388. Internal50-selection supplies the frozen gate, followed by Internal50-holdout only after PASS. Independent49 is frozen confirmation only, never training or selection. The old Internal100 operating-point policy belongs solely to the historical compatibility chain.
