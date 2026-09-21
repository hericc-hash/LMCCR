# Security and privacy

Do not open public issues containing patient identifiers, raw images, reports, access tokens, SSH credentials, or licensed model files. Report a suspected privacy or credential leak privately to the repository owner and rotate affected credentials immediately.

Checkpoint and tensor loaders use PyTorch pickle deserialization (`weights_only=False`) for compatibility with frozen artifacts containing metadata and NumPy objects. Load only trusted local artifacts whose provenance and hashes you have verified. A function named `safe_torch_load` is a version-compatibility helper, not a sandbox for untrusted files. Source snapshots loaded by replay tools are executable code and require the same trust.
