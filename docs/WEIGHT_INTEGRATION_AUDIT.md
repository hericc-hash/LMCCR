# 权重整合续审（2026-09-20）

后续更新见 [补充包单病例实测](SINGLE_CASE_RUNTIME_GAPS_RESULT.md)。本文件保留第一份权重包的审计历史；其中四个 selector、原始 Planner 配置及 realizer 词典缺项现已补齐。

结论：本地权重整合与结构验证通过；完整 MRI → 报告复现尚未通过，当前不能宣称开箱即用复现论文结果。权重发布决定仍待所有者完成数据评估，本次没有暂存、提交或上传权重。

## 已验证

- 最新完整回归测试：43 项通过；`compileall` 通过。修复后的安装器对真实压缩包再次校验和重复安装通过。

- 安装包的 23 个有效载荷全部通过自带 SHA-256 清单验证；安装记录覆盖 25 个文件（包括两份清单）。清单一致性不等于独立来源认证或隐私审查。
- HR320、三个视觉专家、视觉证据编码器、APREB、CCMRT、最终 Planner、辅助 Planner 和已提供的两个 selector 模型均通过严格 state_dict 加载。
- Qwen3 本地 tokenizer 可用，398 个基础模型张量的索引/分片完整，三个 LoRA 各 288 个张量与基础模型形状匹配。尚未执行 Qwen 生成。
- 数据集目录与 manifest 核对为 Development 388、Internal 100、Independent 49。HR320 已在 8 个 Development 标注中心切片执行；这不是自动选片评估，也不是完整患者级验证。
- Git 候选文件检查未发现权重、DICOM、JSONL、压缩包或密钥扩展名文件，也没有大于 1 MB 的候选文件。weights、data、outputs 中仅说明文档与空目录标记进入候选集合；本机配置和结果受忽略规则保护。这不是完整隐私或秘密检测。

## 续审修复与核对

- 安装器新增对保留安装记录目录、Windows 设备名称和尾随点/空格路径的拒绝；安装记录的解析路径也必须位于 weights 内，避免符号链接逃逸。重复安装不覆盖内容不同的已有资产。
- TextListwise 文件名 seed 为 20260931、检查点内为 20261931 符合训练脚本 `scripts/selector/03_train_text_listwise_reranker.py` 的 `seed + 1000` 约定，不是错配。
- Planner 的三个阈值符合 `selection/stage23_exact.py::binary_from_probs`：每个任务阈值重复五次，并添加独立前凸阈值，组成 16 维阈值。该函数默认前凸阈值为 0.5；这不证明缺失的原始运行配置使用了默认值。
- 部分视觉专家的非权重参数来自当前构造函数默认值。严格加载只能证明模型参数兼容，不能证明预处理、ROI 或运行配置与训练实验完全相同。

## 仍阻止完整复现的事项

1. selector 缺失 seeds 20260932、20260933 的 FactNet 和 TextListwise，共四个检查点。
2. Planner 缺失原始 `config_used.json`、`01_stage23b_data.pt`、`07_internal_factual_probs.pt`。后两项用于冻结概率回放，不能用合成数据冒充。
3. 原始 MRI 到视觉专家输入、Raw-E 和 47 维坐标状态的完整导出及去标识病例 ID 对齐尚未接通。
4. 原始 Direct 草稿运行配置/资产与冻结 realizer lexicalizer 尚未提供。
5. P3 机制回放另需冻结源码、pair/feature banks、经验方向选择、证据与 Jacobian 缓存；这些属于机制回放，不能混同普通事实推理的要求。
6. 源码许可证尚未确定；发布权重前仍须单独审核再分发许可、检查点元数据/优化器状态及数据和性能结果。当前安装保留原始字节，不是经过清理的发布导出。

复现命令与范围说明见 [LOCAL_REPRODUCTION.md](LOCAL_REPRODUCTION.md)。本机详细结果在被 Git 忽略的 `outputs/weight_validation/weight_validation.json`，明确记录 `full_pipeline_pass: false`。
