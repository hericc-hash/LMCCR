# Published weights

This private repository includes 22 approved checkpoints, listed with SHA256 hashes in [MANIFEST.json](MANIFEST.json). Eleven larger files use Git LFS; run `git lfs install` and `git lfs pull` after cloning. Four adapter configurations use the portable Qwen model identifier; checkpoint bytes are unchanged.

Patient data, generated reports, original metadata, installation receipts, duplicate packages and runtime asset archives remain excluded. The base Qwen model must be supplied separately. Configure its local path in the ignored `configs/local_assets.json`. The full pipeline still has unresolved runtime dependencies; see [local reproduction status](../docs/LOCAL_REPRODUCTION.md).
