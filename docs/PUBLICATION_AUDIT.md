# GitHub 发布前审计

后续真实权重整合审计见 [WEIGHT_INTEGRATION_AUDIT.md](WEIGHT_INTEGRATION_AUDIT.md)。下文保留首轮源码审计记录；虚拟环境随后已修复，真实权重验证范围及最新测试结果以续审记录为准。

审计日期：2026-09-20。范围：当前工作区源码、文档、配置、Git 忽略规则与 CPU 合成测试；不包含患者数据或真实权重验证。

## 已完成的清理

- 删除 `scripts/legacy/` 下 8 个旧独立脚本及 README，并删除 `configs/legacy_pipeline.json`。
- 修正 README、架构、复现、来源、评价、文件清单及工作流配置中 selector v2.2 尚未补齐的过时说明。
- 源码安装检查增加 selector 实现与配置文件存在性检查。
- Git 忽略规则补充 `.env`、密钥文件及本地 `run/` 目录。
- 许可证待定说明中的旧模型名称改为当前 Qwen3。

## 有意保留

- `planner/planner_legacy.py`：`scripts/planner/02_recover_exact_p3.py` 仍调用其旧检查点恢复接口。
- `clinical/`：上述恢复依赖 ExplicitClinicalPlanner；runtime 和 evaluation 仍使用共享 schema。未将该兼容包整体删除。
- `configs/operating_points.json`：兼容模块的契约测试仍引用；禁止作为最终 Planner 阈值。
- `selection/legacy.py` 和配置中的 `legacy_reuse`：来自精确 v2.2 源码快照，来源哈希测试覆盖；不因名称含 legacy 而破坏冻结来源。
- `configs/selector/frozen_source.json`：保留原始服务器路径作为溯源快照；实际运行使用 default.json 和本地配置。
- 原始整合 manifest 和 MERGE_VALIDATION.json：历史记录，不是本次验证结果。

## 静态检查

清理后的首轮扫描覆盖 200 个 Git 候选文件（新增本审计文档前）：无超过 1 MB 的文件，无常见权重、影像、表格、JSONL 或密钥扩展名文件；所有 JSON 可解析，AST 检查未发现缺失的显式本地模块导入。常见令牌格式和私钥头搜索未发现匹配。此类扫描不等同于完整隐私或秘密检测。

## 发布仍需处理

1. 项目所有者确定源码许可证并补充 LICENSE；当前仓库仍保留 LICENSE_PENDING.md。
2. 补齐并确认 Planner/realizer 原始完整配置，以及 Direct 草稿生成来源/资产。
3. 使用授权资产验证真实权重回放、Qwen 生成、selector 与患者级评价；CPU 合成测试不能替代这些验收。
4. 检查点加载为兼容冻结资产使用 pickle 反序列化，只接受可信资产，详见 SECURITY.md。

## 本地环境

现有 `.venv` 的 Python 入口指向另一台机器，不能直接执行。本次验证使用本机 Python 3.12.14，PYTHONPATH 指向本仓库 src 与现有 `.venv/Lib/site-packages`，未修改全局 Git 安全目录或安装全局依赖。干净安装与 Linux CI 仍需在发布环境确认。

## 验证结果

- `python -m pytest`：32 passed，248.16 秒；包含冻结源码哈希、实际神经选择的合成测试和输入边界测试。
- `python scripts/check_setup.py --code-only`：通过。
- `python -m compileall -q src scripts tests`：通过。
- `.env`、本地配置、run 输出和权重样例的 Git 忽略检查：通过。

运行耗时主要包含本地依赖首次加载；诊断栈曾停在 scikit-learn 扩展导入，随后正常完成。本次没有提交、推送或修改远程仓库。
