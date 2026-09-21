# Architecture

The final factual chain is Raw-E → Stage2.3-B v1.7 → Clinical16 → frozen Direct draft / core-redacted scaffold → R3.1-v3.2 → R3.2-S4-v2.2 → report → canonical Clinical16 parser. The exact selector source and configuration are integrated; see [merge status](MERGE_STATUS.md).

## Evidence and mechanism

Vision, APREB and CCMRT retain the existing upstream implementations. P3 performs training-free finite-step functional search and orientation-specificity qualification for bounded counterfactual supervision. It is not biological causal identification, and its reconstructed evidence is not a substitute for factual Raw-E. Historical Stage2.2 is not an inference stage.

## Final Planner

`planner/model.py` is the exact supplied Stage2.3-v1.7 model definition. `bridge/stage23_exact.py` validates the model configuration and strictly loads checkpoint keys. `predict_clinical16.py` requires an Internal100 probability replay before generating core predictions. The Raw-E interface requires explicit Development-frozen thresholds and an input-availability mask; labels/references are not forwarded to the model.

## Language generation

A frozen Direct draft supplies linguistic context. Clinical16-sensitive spans are redacted in place using generic `<CORE>` markers, preserving safe local phrasing. The merged v3.2 realizer consumes the locked core and scaffold; its adapters are loaded in language → dual-channel → scaffold order. Precision repair is task/level aware and has no canonical-report fallback or LLM repair. Canonical helper text is retained only for lexicon calibration and training target sanitization.

## Final selection boundary

The intended v2.2 selector uses FactNet to establish a factual-safe candidate set and TextListwise to rank within that set using contextual embeddings and shared tabular features. The exact neural source/config was restored from the v2.2 patch; see SELECTOR_V22_MANIFEST.json. FrozenSelector and the staged selector workflow expose the integrated implementation.

Ground truth, references, and reference-derived ROUGE/BLEU are training/evaluation targets only. They are excluded from inference features. Development388 supplies training and OOF profile selection; Internal50-selection is the gate, then Internal50-holdout. Independent49 never supplies tuning data.
