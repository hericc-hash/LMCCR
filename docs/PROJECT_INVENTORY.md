# 工程文件与开发版本清单

> Publication update (2026-09-21) / 发布更新：源码和选定的 22 个权重已上传；所有者已授权公开。下文未上传、仅本地、缺件等表述保留为历史整合记录，当前发布范围以 [README](../README.md) 和 [权重清单](../weights/MANIFEST.json) 为准。历史安装状态不代表文件已公开或端到端验证完成。

更新日期：2026-09-20。已整合后续 curated freeze 的核心模块；selector v2.2 原始源码和配置已补齐。

## 版本口径

- 当前 Python 包标记仍为 **0.1.0**，三处元数据保持一致。完整 0.2.0 发布待补件及最终验收后进行。
- 新方法链的开发版本见下表；包版本不代表模块的历史开发编号。
- Git 原仓库尚无提交；本次没有提交或上传 GitHub。
- 文件来源及源码 SHA256 见 [CURATED_MERGE_MANIFEST.json](CURATED_MERGE_MANIFEST.json)，可区分原样复制与适配文件。

## 模块清单

| 工程目录/文件 | 对应开发版本 | 用途与状态 |
|---|---|---|
| `vision/coordinate_prior.py` | HR320-v7 | 坐标先验；保留原实现 |
| `vision/` 及 `experts/` | 1.16a | 视觉编码、Disc/Stenosis/Nerve 专家；保留 |
| `apreb/` | 1.18b-v3 | A/R 瓶颈及训练支持；保留 |
| `ccmrt/` | 1.20-v2.1 | 匹配残差运输；保留 |
| `p3/` | Stage1.21-P3 final | 训练无关有限步机制搜索，不替代事实 Raw-E |
| `planner/` | Stage2.3-B v1.7 | 最终规划器模型、CF 训练支持；完整训练配置待补 |
| `bridge/` | Stage3-A v1.3 exact interface | 严格加载、概率回放、Raw-E→Clinical16 |
| `generation/` | R3.1-v3.2 | 遮盖 scaffold、精确修补、适配器加载及训练；冻结配置待补 |
| `selection/features.py` | supporting lineage 共享特征 | 266 维部署特征；不含旧版 neural reranker |
| `selection/firewall.py` | 本次集成接口 | 清理参考/GT 字段；清理后的特征接入 FrozenSelector |
| 最终神经 selector | R3.2-S4-v2.2 | TextListwise/text_embedder/frozen config 已整合 |
| `evaluation/report_metrics.py`、`schema.py` | curated frozen Clinical16 parser | 16 槽解析与报告评价，按来源原样复制 |
| `evaluation/bootstrap.py`、`planner_metrics.py` | 原包评价工具 | 原有统计支持 |
| `clinical/`、`runtime/` | 历史 1.18c / P-only / frozen adapters | 兼容和辅助，不是新最终事实链 |

以上源码目录均相对 `src/lumbar_cf_report/`。各模块内模型、数据、损失、指标及辅助文件逐项列在文末完整清单中。

## 入口与配置

| 路径 | 对应流程 |
|---|---|
| `scripts/train_apreb.py`、`train_ccmrt.py` | 保留的上游训练 |
| `scripts/run_p3.py` | P3 preflight/smoke/search/select |
| `scripts/planner/*.py` | v1.7 准备、训练、选择、神经头恢复和评价；要求显式原配置 |
| `scripts/predict_clinical16.py` | v1.7 严格加载与回放后的 Raw-E 推理 |
| `scripts/generate_scaffold_report.py` | v3.2 单组件生成，不冒充最终 S4 选择结果 |
| `scripts/check_setup.py` | 已集成源码检查；`--final` 检查待补状态 |
| `scripts/selector/*.py`、`scripts/run_selector_workflow.py` | v2.2 分步工作流 |
| `configs/paper_pipeline.json` | 新方法链与待补状态清单；不是原冻结运行配置 |
| `configs/p3.json` | 来自原 P3 配置，仅改相对路径 |
| `configs/ccmrt.yaml` | 原 CCMRT 配置 |
| `configs/operating_points.json` | 历史兼容链配置，不用于最终规划器阈值 |
| `tests/test_frozen_workflow.py` | 新冻结契约、源码一致性、v3.2 和共享特征边界测试 |
| `tests/test_apreb.py`、`test_ccmrt.py`、`test_clinical.py` | 保留的原工程测试 |

## 其他工程文件

`pyproject.toml` 定义打包、依赖和测试工具；`requirements.txt` 引用工程生成依赖；`.github/workflows/ci.yml` 执行源码检查与 pytest。`README.md`、`docs/`、引用与许可文档提供使用和发布说明。`data/`、`weights/`、`outputs/` 仅提交说明/占位文件。

Python 要求 `>=3.10`。依赖版本范围以 `pyproject.toml` 为准，不等同于原实验环境锁定版本。此次 CPU 测试环境见 [验证记录](MERGE_VALIDATION.json)。没有合并权重、病例报告、影像、嵌入缓存或 reference-only 实验结果。

## 文件清单

使用 `git ls-files --cached --others --exclude-standard` 获取当前完整清单。历史来源清单保留原始映射；selector 补丁另见 SELECTOR_V22_MANIFEST.json。
