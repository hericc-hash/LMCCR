# Third-party sources and license status / 第三方来源与许可状态

The repository distributes the 22 selected project checkpoints listed in [weights/MANIFEST.json](weights/MANIFEST.json), including visual checkpoints and LoRA adapters. It does not distribute the Qwen base model or patient MRI data. Team-owned original software uses the LMCCR Noncommercial Research Software License 1.0; team-owned documentation and listed checkpoints use CC BY-NC 4.0 to the extent the team can license them. See [LICENSE](LICENSE). 本仓库采用分项非商业许可，第三方内容保留其原有许可；本项目的授权不覆盖无权许可的第三方权利。

Method papers, model references, and implementation attribution are in [README.md](README.md#references). Internal source provenance is recorded in [docs/PROVENANCE.md](docs/PROVENANCE.md). Upstream authors retain their rights, and upstream terms must be reviewed for the versions and assets actually used.

| Component | Official source | Role |
| --- | --- | --- |
| Qwen3-4B-Instruct-2507 | [Model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [license](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE) | External base model; model card identifies Apache-2.0 |
| PyTorch / torchvision | [PyTorch](https://github.com/pytorch/pytorch), [vision](https://github.com/pytorch/vision) | Tensor operations, training, visual backbones |
| Transformers / PEFT / Accelerate / safetensors | [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [Accelerate](https://github.com/huggingface/accelerate), [safetensors](https://github.com/huggingface/safetensors) | Generation, adapters, model loading |
| NumPy / pandas / SciPy / scikit-learn | [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/), [scikit-learn](https://scikit-learn.org/) | Numerical processing and evaluation |
| PyYAML / tqdm | [PyYAML](https://github.com/yaml/pyyaml), [tqdm](https://github.com/tqdm/tqdm) | Configuration and progress display |
| pydicom / Pillow | [pydicom](https://github.com/pydicom/pydicom), [Pillow](https://github.com/python-pillow/Pillow) | Optional DICOM and image processing |
| SentencePiece | [Official repository](https://github.com/google/sentencepiece) | Optional tokenizer support |
| pytest / Ruff / setuptools | [pytest](https://github.com/pytest-dev/pytest), [Ruff](https://github.com/astral-sh/ruff), [setuptools](https://github.com/pypa/setuptools) | Development and packaging |
| Git LFS | [Official project](https://git-lfs.com/) | Large checkpoint storage |

This is an attribution index, not a completed legal clearance or a software bill of materials for a locked environment. Dependency ranges are recorded in [pyproject.toml](pyproject.toml). Where weights or code inherit upstream restrictions, this repository does not supersede them.
