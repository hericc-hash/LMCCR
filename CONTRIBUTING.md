# Contributing

Keep contributions on the paper's reproducible mainline. New modules should use paper concepts (`vision`, `apreb`, `ccmrt`, `clinical`, `evaluation`) rather than chronological experiment labels. Do not commit patient data, reports, checkpoints, generated outputs, absolute machine paths, credentials, or historical experiment dumps.

Before opening a pull request, run:

```bash
python scripts/check_setup.py --code-only
pytest
```
