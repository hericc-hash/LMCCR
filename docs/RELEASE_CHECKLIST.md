# Release checklist

- [x] Restore the exact v2.2 TextListwise/text encoder source and frozen config (SELECTOR_V22_MANIFEST.json).
- [ ] Restore original Planner/realizer configurations and verify required Direct draft lineage.
- [x] Run local source/import/contract tests, including exact final selector firewall tests (32 passed; see PUBLICATION_AUDIT.md).
- [ ] Confirm clean installation and Linux CI in the publication environment.
- [ ] Validate actual checkpoint hashes and saved probability replay on authorized private assets.
- [ ] Verify final S4 generation and selection; do not substitute the intermediate realizer output.
- [ ] Confirm Development/Internal selection boundaries and no Independent49 tuning.
- [ ] Keep weights, patient data, caches and generated reports out of Git.
- [ ] Determine the final source license and third-party acknowledgements.
- [ ] After completion, synchronize package version 0.2.0 in pyproject, package and CITATION.
- [ ] Record the release commit, environment and reproducibility limitations.

Current merge status: [MERGE_STATUS.md](MERGE_STATUS.md).
