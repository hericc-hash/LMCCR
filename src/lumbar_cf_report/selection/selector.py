"""Portable boundary around the exact v2.2 frozen ensemble selection algorithm."""
from __future__ import annotations
import numpy as np
from .features import extract, extract_case, language_proxy_from_case_features, _core_f1
from .firewall import deployment_case
from .reranker import load_bundle, predict_fact, predict_text_listwise, choose
from .text_embedder import cache_to_case_embeddings


class FrozenSelector:
    def __init__(self, fact_bundles, text_bundles, profile, device='cpu'):
        if not fact_bundles or not text_bundles:
            raise ValueError('Both FactNet and TextListwise ensembles are required')
        if any(b.get('kind') != 'fact' for b in fact_bundles) or any(b.get('kind') != 'text_listwise' for b in text_bundles):
            raise ValueError('Expected exact v2.2 fact/text_listwise checkpoints')
        if profile.get('rule') != 'factual_safe_text_listwise':
            raise ValueError('Expected frozen factual_safe_text_listwise profile')
        self.profile = {k: profile[k] for k in ('rule', 'alpha_fact', 'fact_margin', 'text_model_weight')}
        for key in ('alpha_fact', 'fact_margin', 'text_model_weight'):
            if not np.isfinite(self.profile[key]) or not 0 <= self.profile[key] <= 1:
                raise ValueError('Invalid profile parameter: ' + key)
        self.fact_bundles = fact_bundles
        self.text_bundles = text_bundles
        self.device = device

    @classmethod
    def from_paths(cls, fact_paths, text_paths, profile, device='cpu'):
        return cls([load_bundle(p) for p in fact_paths], [load_bundle(p) for p in text_paths], profile, device)

    def select(self, rows, embedding_cache):
        """Use fingerprint-checked frozen embeddings; never consume evaluation targets."""
        cases = [deployment_case(row) for row in rows]
        embeddings = cache_to_case_embeddings(cases, embedding_cache, strict=True)
        results = []
        for case, ec in zip(cases, embeddings):
            Xf, nf = zip(*(extract(case, cand) for cand in case['candidates']))
            Xf = np.asarray(Xf)
            Xt, nt = extract_case(case)
            for bundle in self.fact_bundles:
                if bundle.get('feature_names') and bundle['feature_names'] != nf[0]:
                    raise ValueError('Factual feature schema differs from training')
            for bundle in self.text_bundles:
                if bundle.get('feature_names') and bundle['feature_names'] != nt:
                    raise ValueError('Text feature schema differs from training')
            if not all(np.isfinite(x).all() for x in (Xf, Xt, ec)):
                raise ValueError('Non-finite selector inputs')
            fp = np.mean([predict_fact(b, Xf, self.device) for b in self.fact_bundles], axis=0)
            scores = []
            for b in self.text_bundles:
                x = predict_text_listwise(b, ec, Xt, self.device)
                # The original selection script normalizes each ensemble member
                # before averaging. Do not average raw listwise logits.
                scores.append((x-x.min())/(x.max()-x.min()+1e-8) if len(x)>1 else np.zeros_like(x))
            ts = np.mean(scores, axis=0)
            if not np.isfinite(fp).all() or not np.isfinite(ts).all():
                raise ValueError('Non-finite selector predictions')
            proxy = language_proxy_from_case_features(Xt, nt)
            fidelity = [_core_f1(c['parser_vec'], case['core_binary'], case['slot_valid']) for c in case['candidates']]
            index, rank, factual, eligible = choose(fp, fidelity, ts, proxy, self.profile)
            cand = case['candidates'][index]
            results.append({'serial': case['serial'], 'selected_tag': cand['tag'], 'generated': cand['text'],
                            'candidate_parser_vec': cand['parser_vec'], 'selected_index': index,
                            'candidate_scores': rank, 'candidate_fact_scores': factual,
                            'factual_safe_candidate_indices': eligible})
        return results
