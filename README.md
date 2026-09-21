# LMCCR

**Lumbar MRI clinical planning, report generation, and controlled counterfactual research**

**腰椎 MRI 临床规划、报告生成与受控反事实研究工程**

[中文说明](#中文说明) · [English](#english) · [References / 引用](#references) · [Documentation / 文档](#documentation)

| Package version / 包版本 | Python | Status / 状态 | Weights / 权重 |
| --- | --- | --- | --- |
| `0.1.0` | `>=3.10` | Research integration / 研究整合版 | 22 checkpoints; 11 Git + 11 Git LFS |

> **Release scope / 发布范围:** Source and selected checkpoints are available. Full MRI-to-report execution and paper-level reproduction have **not** been established. 本仓库提供源码及选定检查点，**尚未完成完整 MRI → 报告流程验收及论文级复现**。

<a id="中文说明"></a>
## 中文说明

### 项目简介

LMCCR 面向腰椎 MRI 的视觉证据建模、结构化临床状态规划、报告生成与受控反事实机制研究。工程整合了冻结视觉前端、APREB、CCMRT、P3、Clinical16 规划接口、scaffold 报告生成器，以及最终 R3.2-S4-v2.2 神经选择器。

报告生成使用外部 **Qwen3-4B-Instruct-2507** 基础模型及项目适配器 [1, 2]，适配器采用 LoRA 方法并通过 PEFT 加载 [3, 7]。部分视觉专家使用 torchvision 的 ResNet-18 架构，并配置 Group Normalization [4, 5, 8]。这些基础方法的作者归属见文末引用；工程内部阶段编号不代表独立公开论文。

### 方法流程

```text
MRI → 冻结视觉前端 / Raw-E
    → Stage2.3-B v1.7 Planner → Clinical16
    → Direct 草稿 → core-redacted scaffold
    → R3.1-v3.2 候选生成与精确修补
    → R3.2-S4-v2.2 FactNet + TextListwise
    → 最终报告 → 冻结 Clinical16 解析与评价
```

P3 用于上游机制检验和有界反事实监督，不替代事实推理中的 Raw-E；Stage2.2 不作为当前运行时阶段。机制响应的方向性不等同于生物学因果关系。上图描述目标方法链，不能据此认定仓库已经提供完整的一键推理入口。

### 模块与开发版本

路径均相对 `src/lumbar_cf_report/`。包版本与历史模块开发编号分别管理。

| 模块 | 开发版本 | 主要职责 |
| --- | --- | --- |
| `vision/coordinate_prior.py` | HR320-v7 | 坐标先验 |
| `vision/`、`vision/experts/` | 1.16a 及专家实现 | 视觉证据与 Disc / Stenosis / Nerve 专家 |
| `apreb/` | 1.18b-v3 | 解剖/病理分解 |
| `ccmrt/` | 1.20-v2.1 | 匹配残差运输 |
| `p3/` | Stage1.21-P3 final | 有限步机制搜索 |
| `planner/` | Stage2.3-B v1.7 | 最终事实规划器及训练支持 |
| `bridge/` | Stage3-A v1.3 | 检查点加载、概率回放与 Clinical16 接口 |
| `generation/` | R3.1-v3.2 | scaffold 生成、适配器与精确修补 |
| `selection/` | R3.2-S4-v2.2 | FactNet + TextListwise 最终选择器 |
| `evaluation/` | Frozen Clinical16 parser | 临床槽解析、评价与统计 |

selector v2.2 使用补齐的原始源码与冻结配置；共享特征文件保留其历史来源，不用其他版本的 reranker 替代 v2.2。详细文件与来源映射见 [工程清单](docs/PROJECT_INVENTORY.md) 和 [来源记录](docs/PROVENANCE.md)。

### 安装与源码检查

先安装 Git、[Git LFS](https://git-lfs.com/) 和 Python >=3.10。以下命令在克隆后的工程目录执行。

```bash
git lfs install
git clone https://github.com/hericc-hash/LMCCR.git
cd LMCCR
git lfs pull
python -m venv .venv
```

激活虚拟环境，按操作系统选择一项：

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

安装依赖并检查源码：

```bash
python -m pip install -e ".[dev]"
python scripts/check_setup.py --code-only
python -m pytest
```

DICOM 处理与生成组件还需要可选依赖：

```bash
python -m pip install -e ".[dicom,generation]"
```

依赖范围以 [pyproject.toml](pyproject.toml) 为准，它不是原实验的完整锁定环境。完整 Qwen 推理的设备、显存和耗时尚未在本发布版验证，不提供未经测试的硬件最低配置承诺。

### 权重、数据与本地配置

- 仓库包含 **22 个权重文件**，约 **936 MiB**：11 个小文件直接保存在 Git，11 个大文件通过 Git LFS 保存；另包含 4 份 adapter 配置。
- 路径、文件大小及 SHA256 见 [weights/MANIFEST.json](weights/MANIFEST.json)。检查点二进制保持原始字节；adapter 配置中的基础模型路径已改为可移植的模型标识。
- **Qwen3-4B-Instruct-2507 基础模型不在仓库中**，请从 [官方模型卡][qwen-model] 获取并遵循其说明与许可。
- 患者影像、病例标识、参考报告、生成报告、特征缓存、本地路径配置、令牌、重复补件和安装回执不随仓库发布。
- 将 [configs/reproduction.example.json](configs/reproduction.example.json) 复制为被 Git 忽略的 `configs/local_assets.json`，在本地填写数据集、基础模型和输出路径。

示例配置还引用未发布的安装回执、原始运行配置和 selector 元数据。克隆仓库并拉取权重，不等于获得这些本地资产；`validate_weight_bundle.py` 面向原始本地安装包，不能作为干净克隆后的唯一就绪判据。原始包的安装与验证流程见 [本地复现记录](docs/LOCAL_REPRODUCTION.md)。

### 验证状态与限制

| 项目 | 当前状态 |
| --- | --- |
| 源码回归测试 | 发布前本地运行：50 项通过；不是临床性能指标 |
| 发布权重 | 22 个文件完成 SHA256 核验；11 个 LFS 对象上传成功 |
| 视觉组件 | 历史本地记录包含 HR320 及单病例自动选片、映射和专家执行 |
| 完整事实推理 | 尚未完成 MRI → 最终报告的全链路验收 |
| Qwen 生成、候选选择与队列评估 | 仍需实际运行与冻结协议验证 |
| P3 机制回放 | 还依赖独立的历史快照、特征库与缓存等资产 |

发布前网络中断使 LFS 远端下载复核未完成。克隆后请确认 `git lfs pull` 成功，并按清单核对文件大小和 SHA256。历史文档中的“已安装”表示原本地环境具备该资产，不意味着公开仓库包含该文件。最新整合缺项见 [全流程清单](docs/FULL_PIPELINE_CHECKLIST.md)。

### 贡献、隐私与许可

提交改动前运行源码检查与 pytest，遵循 [贡献说明](CONTRIBUTING.md)。不要在提交、Issue 或日志中加入患者信息、令牌或私有模型资产。冻结检查点和历史快照只能从可信来源加载，参见 [安全说明](SECURITY.md)。

本工程用于研究，尚未进行临床部署验证。**项目源码与项目权重的统一许可证尚未确定**；仓库公开可见不表示已授予开源使用或再分发许可。基础模型、依赖及其他第三方组件遵循各自条款，见 [许可状态](LICENSE_PENDING.md) 和 [第三方说明](THIRD_PARTY_LICENSES.md)。

<a id="english"></a>
## English

### Overview and architecture

LMCCR integrates visual evidence modeling, structured clinical planning, report generation, and controlled counterfactual mechanism research for lumbar MRI. Its generation components use **Qwen3-4B-Instruct-2507** [1, 2] with LoRA adapters loaded through PEFT [3, 7]. Selected visual experts use torchvision ResNet-18 backbones with Group Normalization [4, 5, 8].

The intended factual workflow is:

```text
MRI → frozen visual frontend / Raw-E
    → Stage2.3-B v1.7 Planner → Clinical16
    → Direct draft → core-redacted scaffold
    → R3.1-v3.2 candidate generation and precision patching
    → R3.2-S4-v2.2 FactNet + TextListwise selection
    → report → frozen Clinical16 parsing and evaluation
```

P3 supports mechanism testing and bounded counterfactual supervision; it does not replace factual Raw-E inputs. Stage2.2 is not a runtime stage. Mechanistic directionality is not evidence of biological causality. This diagram describes the intended method, not a completed end-to-end executable pipeline.

### Components and development versions

| Component | Development version | Location under `src/lumbar_cf_report/` |
| --- | --- | --- |
| Coordinate prior / visual evidence | HR320-v7 / 1.16a | `vision/` |
| Anatomy/pathology decomposition | APREB 1.18b-v3 | `apreb/` |
| Residual transport | CCMRT 1.20-v2.1 | `ccmrt/` |
| Mechanism search | Stage1.21-P3 final | `p3/` |
| Factual planner | Stage2.3-B v1.7 | `planner/` |
| Clinical16 bridge | Stage3-A v1.3 | `bridge/` |
| Report generation and patching | R3.1-v3.2 | `generation/` |
| Final neural selector | R3.2-S4-v2.2 | `selection/` |
| Parsing and evaluation | Frozen Clinical16 parser | `evaluation/` |

The package version is `0.1.0`; internal development labels are separate. The exact v2.2 selector source and frozen configuration are integrated. Shared feature lineage is documented separately; another reranker version is not substituted for v2.2.

### Installation and checks

Install Git, [Git LFS](https://git-lfs.com/), and Python >=3.10, then run:

```bash
git lfs install
git clone https://github.com/hericc-hash/LMCCR.git
cd LMCCR
git lfs pull
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -e ".[dev]"
python scripts/check_setup.py --code-only
python -m pytest
# Optional DICOM processing and generation dependencies:
python -m pip install -e ".[dicom,generation]"
```

[pyproject.toml](pyproject.toml) specifies dependency ranges, not the original experiment's locked environment. Full generation hardware requirements, VRAM usage, and latency have not been validated for this release.

### Checkpoints and local assets

The repository includes **22 checkpoint files**, approximately **936 MiB** in total: 11 regular Git files and 11 Git LFS files, plus four adapter configurations. [weights/MANIFEST.json](weights/MANIFEST.json) records paths, sizes, and SHA256 hashes. Checkpoint bytes are preserved; adapter configurations use a portable base-model identifier.

Obtain the **Qwen3-4B-Instruct-2507 base model separately** from its [official model card][qwen-model]. Patient images, identifiers, reference/generated reports, feature caches, credentials, local configuration, duplicate archives, and installation receipts are excluded.

Copy `configs/reproduction.example.json` to the Git-ignored `configs/local_assets.json` and configure local dataset, base-model, and output paths. The example also references unpublished installation receipts, original runtime configuration, and selector metadata. A clone plus `git lfs pull` does not supply those assets. `validate_weight_bundle.py` targets the original local installation bundle and is not a standalone readiness check for a clean clone.

### Validation and limitations

The pre-publication local regression run passed **50 tests**. SHA256 hashes were checked for all 22 checkpoints, and uploading all 11 LFS objects succeeded. An independent remote LFS download check was interrupted by connectivity failures; verify downloaded file sizes and hashes after cloning.

Historical local diagnostics exercised HR320 and selected visual stages, including automatic slice selection and expert execution on one case. These checks do not establish clinical accuracy, held-out performance, or full reproduction. Full MRI-to-report execution, actual Qwen generation, final candidate selection, and cohort evaluation remain unfinished. P3 replay additionally depends on separate historical snapshots, feature banks, and caches.

Historical documents record local integration stages. “Installed” does not mean “redistributed,” and later asset integration does not imply completed execution. See the [pipeline checklist](docs/FULL_PIPELINE_CHECKLIST.md) for remaining work.

### Contribution, privacy, and licensing

Follow [CONTRIBUTING.md](CONTRIBUTING.md), run the source checks and tests, and keep patient information and credentials out of commits and issues. Only load trusted checkpoints and executable source snapshots; see [SECURITY.md](SECURITY.md).

This is a research integration release without clinical deployment validation. **A project-wide source and checkpoint license has not yet been selected.** Public availability does not itself grant an open-source or redistribution license. Upstream models and dependencies retain their own terms; see [LICENSE_PENDING.md](LICENSE_PENDING.md) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

<a id="documentation"></a>
## Documentation / 文档

```text
LMCCR/
├── src/lumbar_cf_report/   # Research modules / 研究模块
├── scripts/               # Training, component inference and diagnostics / 组件入口
├── configs/               # Tracked examples and frozen selector configuration
├── tests/                 # Regression and interface tests / 回归与契约测试
├── weights/               # Approved checkpoints, LFS pointers and hash manifest
├── docs/                  # Design, provenance and integration records
├── data/                  # Instructions and placeholders; no patient data
└── outputs/               # Instructions; generated results are ignored
```

| Document / 文档 | Purpose / 用途 |
| --- | --- |
| [Project inventory](docs/PROJECT_INVENTORY.md) | File and version mapping / 文件与版本 |
| [Architecture](docs/ARCHITECTURE.md) | Method structure / 方法结构 |
| [Provenance](docs/PROVENANCE.md) | Internal source lineage / 内部源码来源 |
| [Selector manifest](docs/SELECTOR_V22_MANIFEST.json) | Exact v2.2 source provenance / v2.2 来源记录 |
| [Reproduction guide](docs/REPRODUCING_THE_PAPER.md) | Reproduction protocol / 复现协议 |
| [Data setup](docs/DATA_SETUP.md) | Local data preparation / 本地数据准备 |
| [Weight manifest](weights/MANIFEST.json) | Published checkpoint hashes / 发布权重哈希 |
| [Pipeline checklist](docs/FULL_PIPELINE_CHECKLIST.md) | Integration gaps / 整合缺项 |
| [Local reproduction record](docs/LOCAL_REPRODUCTION.md) | Historical local checks / 历史验证 |
| [Evaluation protocol](docs/EVALUATION_PROTOCOL.md) | Evaluation boundaries / 评价边界 |

<a id="references"></a>
## References and citation / 参考文献与引用

### Citing LMCCR / 引用本工程

A verified paper title, author list, and DOI for LMCCR have not been supplied. Until formal publication metadata is available, cite the repository and the exact commit used; do not treat the following software reference as a paper citation. LMCCR 的正式论文题名、作者及 DOI 尚待补充，现阶段请引用仓库及实际使用的提交版本。

```bibtex
@misc{lmccr_repository,
  title        = {LMCCR: Lumbar MRI Clinical Planning, Report Generation, and Controlled Counterfactual Research},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/hericc-hash/LMCCR},
  note         = {Software version 0.1.0; record the exact commit used}
}
```

### Upstream methods and software / 上游方法与软件

1. Qwen Team. **Qwen3 Technical Report** (2025). [Original report](https://arxiv.org/abs/2505.09388). General model-family reference; the exact checkpoint used here is identified separately below.
2. Qwen Team. **Qwen3-4B-Instruct-2507**. [Official model card][qwen-model] · [Model license](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE). Exact external base model for generation and text embeddings.
3. Hu, E. J., et al. **LoRA: Low-Rank Adaptation of Large Language Models** (2021). [Original paper](https://arxiv.org/abs/2106.09685). Adapter method; this repository uses PEFT rather than claiming to vendor the original `loralib` implementation.
4. He, K., Zhang, X., Ren, S., and Sun, J. **Deep Residual Learning for Image Recognition** (2015). [Original paper](https://arxiv.org/abs/1512.03385). ResNet architecture used by visual experts through torchvision.
5. Wu, Y., and He, K. **Group Normalization** (2018). [Original paper](https://arxiv.org/abs/1803.08494). Normalization used in the visual expert backbones.
6. Paszke, A., et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library** (2019). [Original paper](https://arxiv.org/abs/1912.01703). Tensor computation and training framework.
7. Hugging Face contributors. **PEFT**. [Official repository](https://github.com/huggingface/peft). Adapter loading and training implementation.
8. PyTorch contributors. **torchvision**. [Official repository](https://github.com/pytorch/vision). Visual backbone implementation.
9. Hugging Face contributors. **Transformers**. [Official repository](https://github.com/huggingface/transformers). Base-model and tokenizer loading. See its upstream citation instructions for the version used.

Other declared dependencies and optional tools are linked in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Internal source archives and adaptations are attributed through [PROVENANCE.md](docs/PROVENANCE.md) and the source manifests. Citations identify methods and implementations; they do not replace license grants or imply upstream endorsement. 上述引用用于说明方法及实现来源，不替代许可授权，也不表示上游作者认可本项目。

[qwen-model]: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
