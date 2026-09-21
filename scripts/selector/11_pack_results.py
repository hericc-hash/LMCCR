from __future__ import annotations
import os,zipfile
from pathlib import Path

def main():
    o=Path(os.environ['OUT']);z=o/'STAGE3C_R32_S4_V22_TEXT_LISTWISE_RERANKER_RESULTS_SLIM.zip';allow=[
      '00_PREFLIGHT.json','00_source/SELECTION50_S4_ORACLE_CEILING_SOURCE.json','00_source/S4_V2_SELECTION_GATE_SOURCE.json','00_source/S4_V2_OOF_CV_REPORT_SOURCE.json','00_source/S4_V2_FINAL_DECISION_SOURCE.json','00_source/S4_V21_FINAL_DECISION_SOURCE.json',
      '01_import/IMPORT_REPORT.json','02_text_embeddings/EMBEDDING_COMPLETE.json','02_text_embeddings/DEV388_TEXT_EMBEDDINGS.json','02_text_embeddings/SELECTION50_TEXT_EMBEDDINGS.json','03_development_candidates/DEV_POOL_SUMMARY.json',
      '04_reranker/OOF_CV_REPORT.json','04_reranker/SELECTED_PROFILE.json','04_reranker/SELECTED_PROFILE_PRIMARY.json','04_reranker/SELECTED_PROFILE_SECONDARY_F1_072.json','04_reranker/TRAINING_COMPLETE.json',
      '05_selection_pool/POOL_SUMMARY.json','06_selection/SELECTION_DECISION_TRACE.json','06_selection/SELECTED_PREDICTIONS.jsonl','06_selection_oracle_audit/SELECTION50_S4_ORACLE_CEILING.json','07_selection_gate/SELECTION_GATE.json',
      '08_holdout_pool/POOL_SUMMARY.json','08_holdout_pool/HOLDOUT50_TEXT_EMBEDDINGS.json','09_holdout/HOLDOUT_SELECTED_PREDICTIONS.jsonl','09_holdout/HOLDOUT_AND_COMBINED_GATE.json','10_FINAL_R32_S4_V22_DECISION.json','10_FINAL_R32_S4_V22_DECISION.md']
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as f:
        for rel in allow:
            p=o/rel
            if p.exists():f.write(p,rel)
    print(z)
if __name__=='__main__':main()
