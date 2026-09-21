# LMCCR

腰椎 MRI 临床规划、报告生成与受控反事实机制研究工程。已整合 curated freeze 核心源码及最终 selector v2.2 原始源码、配置；**Planner/realizer 完整冻结配置与真实资产验证仍待完成，尚不是完整端到端发布版**。详细状态见 [整合说明](docs/MERGE_STATUS.md)。

## 最终方法链

```text
冻结视觉前端 / Raw-E
  → Stage2.3-B v1.7 事实规划器
  → Clinical16
  → 冻结 Direct 草稿 → core-redacted scaffold
  → R3.1-v3.2 生成器与精确修补
  → R3.2-S4-v2.2 FactNet + TextListwise 选择器
  → 报告 → 冻结 Clinical16 解析与评价
```

P3 是上游机制检验和有界反事实监督模块，不替代事实推理的 Raw-E。Stage2.2 不作为运行时阶段。机制方向性不等同于生物学因果方向。

## 工程布局

| 目录 | 用途 |
|---|---|
| `src/lumbar_cf_report/vision/` | 保留 HR320-v7 坐标先验与 1.16a 视觉证据 |
| `src/lumbar_cf_report/apreb/` | 保留 1.18b-v3 解剖/病理分解 |
| `src/lumbar_cf_report/ccmrt/` | 保留 1.20-v2.1 残差运输 |
| `src/lumbar_cf_report/p3/` | 最终 P3 搜索与机制检验 |
| `src/lumbar_cf_report/planner/` | Stage2.3-B v1.7 模型及训练支持 |
| `src/lumbar_cf_report/bridge/` | 严格检查点加载、概率回放与 Clinical16 接口 |
| `src/lumbar_cf_report/generation/` | 合并后的 R3.1-v3.2 scaffold、修补、训练与生成 |
| `src/lumbar_cf_report/selection/` | 最终 v2.2 神经选择器、共享特征与输入边界 |
| `src/lumbar_cf_report/evaluation/` | 冻结 Clinical16 解析及评价 |
| `src/lumbar_cf_report/clinical/` | P3 历史检查点恢复及共享 schema 等兼容支持；旧独立脚本已移除 |
| `configs/`、`docs/`、`tests/` | 配置、来源、清单和契约测试 |

## 安装与检查

Python >=3.10。在仓库根目录执行：

```bash
python -m venv .venv
# 激活 .venv 后执行
python -m pip install -e ".[dev]"
python scripts/check_setup.py --code-only
python -m pytest
python -m compileall -q src scripts tests
```

生成还需 `pip install -e ".[generation]"`、支持本地 Qwen3 检查点的 Transformers 环境及授权权重。依赖范围不是原实验锁定环境。

`check_setup.py --code-only --final` 会明确报告待补项。源码检查通过不代表已完成私有数据、真实权重或端到端生成验证。

## 文档

- [工程文件与版本清单](docs/PROJECT_INVENTORY.md)
- [本次整合与待补项](docs/MERGE_STATUS.md)
- [方法架构](docs/ARCHITECTURE.md)
- [复现流程](docs/REPRODUCING_THE_PAPER.md)
- [数据准备](docs/DATA_SETUP.md) · [权重准备](docs/WEIGHTS_SETUP.md)
- [评价协议](docs/EVALUATION_PROTOCOL.md)
- [论文代码映射](docs/PAPER_CODE_MAP.md) · [来源映射](docs/PROVENANCE.md)
- [发布检查表](docs/RELEASE_CHECKLIST.md)
- [发布前审计与 legacy 清理记录](docs/PUBLICATION_AUDIT.md)
- [本地权重安装与全流程复现状态](docs/LOCAL_REPRODUCTION.md)

仓库包含经核验的 22 个正式权重：11 个通过普通 Git 保存，11 个通过 Git LFS 保存。克隆后执行 `git lfs pull`；文件大小与 SHA256 见 [权重清单](weights/MANIFEST.json)。患者数据、生成报告、本地配置、令牌及基础 Qwen 模型不包含在仓库内。最终源码许可证仍待项目所有者确定，见 `LICENSE_PENDING.md`。
