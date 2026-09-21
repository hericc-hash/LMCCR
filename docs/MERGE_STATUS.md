# Curated freeze 整合状态

更新日期：2026-09-20。输入：`LMCCR_GITHUB_CURATED_FINAL_FREEZE_v1_20260919.zip`。

## 已整合

- 保留原 Vision / APREB / CCMRT 算法文件。
- 加入最终 P3 包、原始科学参数与相对路径配置。
- 加入 Stage2.3-B v1.7 模型、训练支持、上下文回退和神经任务头恢复脚本。
- 使用唯一模型定义连接精确加载与 Internal100 概率回放接口；推理显式要求 `raw_E`，不回退到 Ehat。
- 合并 v3.2 最终补丁与必要的 canonical/sanitize/prompt/precision-patch 辅助代码；不提供 v3/v3.1/v3.2 运行时切换。
- 原样采用包内 Clinical16 解析器与配套 schema；其文件及规划器模型文件可用来源 SHA256 验证。
- 历史临床模块保留兼容；旧独立训练/生成/评价脚本及其 legacy_pipeline.json 已删除。P3 恢复、共享 schema 及精确来源文件仍保留。
- 移除旧第三方模型占位说明；最终权重指向 Qwen3 与其适配器。

## 待补，未以旧版本替代

最终 selector v2.2 已从原始补丁恢复，包括 text_embedder.py、reranker.py、冻结配置和分步脚本，来源见 [SELECTOR_V22_MANIFEST.json](SELECTOR_V22_MANIFEST.json)。

| 缺件 | 影响 |
|---|---|
| Stage2.3-v1.7 完整训练配置 | 已有模型及训练源码，但完整重新训练需原运行配置 |
| R3.1-v3.2 完整冻结配置 | 已有生成/训练实现；精确复现需 scaffold、patch、generation、训练配置 |
| 原始 Direct 草稿生成来源/资产 | 当前接口接收已冻结 Direct 草稿；此包不构成从影像重新生成该草稿的完整源码链 |

`require_final_selector()` 返回实际 FrozenSelector。CURATED_MERGE_MANIFEST.json 中的缺失导入列表是首次整合的历史记录，后续 selector 补丁记录优先；不代表当前源码缺失。

## 发布状态

2026-09-20 已加入本地权重安装与验证入口。新权重包的 23 个有效载荷通过 SHA256 校验，权重公开发布仍待数据评估和所有者决定。当前权重及剩余全流程缺件见 [LOCAL_REPRODUCTION.md](LOCAL_REPRODUCTION.md)；权重可严格加载不代表 MRI 到报告的全流程已经通过。

包版本暂保留 **0.1.0**，表示尚未完成 0.2.0 发布收尾，而不是下游模型仍为旧版。待补齐源码、冻结配置并完成验收后，再同步 `pyproject.toml`、包 `__init__.py` 和 `CITATION.cff` 为 0.2.0。

当前只执行源码和合成输入验证；未运行真实训练、真实权重概率回放、Qwen 生成或患者级评价。Git 仓库原先没有提交；本次未提交、未推送。
