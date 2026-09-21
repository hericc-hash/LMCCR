"""Whitelist inference data before evaluating shared selector features."""
from __future__ import annotations
from copy import deepcopy

from .features import extract_case

CASE_FIELDS = ('core_binary', 'slot_valid', 'planner_probs', 'planner_thresholds',
               'direct_parser_vec', 'direct_draft', 'linguistic_scaffold', 'expanded')
CANDIDATE_FIELDS = ('tag', 'text', 'parser_vec')


def deployment_case(case):
    """Reference/GT/target scores are discarded, including nested candidate fields.

    Preserve only deployment inputs for the exact v2.2 implementation.
    """
    clean = {k: deepcopy(case[k]) for k in CASE_FIELDS if k in case}
    clean['serial'] = str(case['serial']) if 'serial' in case else ''
    adaptive_fields = ('score', 'best_core_mismatch', 'best_high_conf_mismatch',
                       'best_stenosis_nerve_high_conf_mismatch', 'raw_final_parser_disagreement',
                       'planner_direct_high_conf_disagreement', 'text_pathology')
    meta = case.get('adaptive_meta') or {}
    clean['adaptive_meta'] = {key: float(meta.get(key, 0.0)) for key in adaptive_fields}
    clean['candidates'] = [{k: deepcopy(c[k]) for k in CANDIDATE_FIELDS if k in c}
                           for c in case.get('candidates', [])]
    if not clean['candidates']:
        raise ValueError('At least one candidate is required')
    return clean


def inference_features(case):
    return extract_case(deployment_case(case))


def require_final_selector():
    from .selector import FrozenSelector
    return FrozenSelector
