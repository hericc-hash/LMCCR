# Evaluation protocol

The canonical parser is `lumbar_cf_report.evaluation.report_metrics`, copied from the curated freeze. Clinical16 uses the fixed order: lordosis, five Disc slots (L1/2 through L5/S1), five Stenosis slots, five Nerve slots. An unmentioned `None` slot normalizes to zero; never collect only numeric dictionary values or drop lordosis.

Primary report Clinical F1 is `compute_report_slot_metrics(...)["micro_f1"]` over all 16 valid slots. Disease15 is a separate diagnostic and must not replace Clinical16. Report-vs-core fidelity and report-vs-ground-truth performance are different endpoints.

Character metrics use the frozen implementation. If complete frozen reference text is unavailable, mark character/ROUGE/BLEU metrics unavailable; do not invent references. Clinical labels are evaluation inputs only, not generation or selection features.

The final protocol uses Development388 for training and OOF profile selection, Internal50-selection for its frozen gate, and Internal50-holdout only after passing the gate. Independent49 is frozen confirmation only. No patient-level results or measured paper outcomes are included here.

The exact v2.2 selector source/config is integrated. Tests cover source identity, synthetic neural selection, cache alignment and the inference feature firewall. Actual checkpoint and patient-cohort evaluation remains outstanding.
