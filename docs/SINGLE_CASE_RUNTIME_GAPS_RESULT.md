# 补充包整合与单病例实测（2026-09-20）

**结论：前半段真实影像推理通过；完整 MRI → 最终报告尚未通过。** 本次失败原因是仍缺冻结运行资产，并非已经证实本机算力不足。没有生成或伪造最终报告，没有上传权重。

## 已补齐并验证

- 补充包 `LMCCR_FULL_PIPELINE_RUNTIME_GAPS_20260920.zip` 的 1,434 个文件通过 SHA-256 校验；五份顶层收集清单不在其哈希清单内。
- 安装缺失的四个 selector 权重，现有六个 selector 模型均严格加载通过。
- 安装原始 Planner `config_used.json`；实际配置与仓库模型契约匹配。
- 原冻结 Reader50 配置确认前凸阈值为 0.5，三个任务阈值与 Planner 检查点逐项一致。
- 安装 `CANONICAL_LEXICALIZER.json`、Direct 选择记录，以及冻结选片、映射、Direct/realizer/selector 运行源码和配置。
- 补充包未包含全部 `models.stage1_19i49_*` 依赖，已从本机原 private_repro_v3 源码补齐；没有覆盖新包自带文件。166 个安装文件的来源及哈希保存在 `weights/runtime/INSTALLATION.json`。
- 所有导入资产保留在被 Git 忽略的 weights 下。历史源码未作为新的公共 legacy 目录加入仓库。
- 回归测试 48 项通过，最小统计导出器另有 2 项测试通过，共 50 项；编译检查通过。

## 单病例真实执行结果

仅选取一个 Development 病例，CPU、4 线程、batch size 1、workers 0。没有读取 annotations、参考报告或诊断标签。

| 阶段 | 结果 | 实际执行范围 |
|---|---|---|
| 自动选片 | PASS | 使用冻结的矢状 T2 评分与几何中间层规则 |
| HR320 | PASS | 严格权重校验、真实像素推理、五个下游节段坐标 |
| 狭窄轴位映射 | PASS | 从选定矢状位坐标映射到轴位切片 |
| 神经轴位映射 | PASS | 轴位切片与相关 ROI 几何映射 |
| 三个视觉专家 | PASS | 每个专家输出 `[5,768]` 特征；15 个任务/节段有效标记均为 1，输出数值有限 |
| 视觉证据组装与坐标归一化 | BLOCKED | 缺冻结几何模板、规则尺度与坐标归一化统计 |
| Planner 病例推理 | 未执行 | 缺完整事实输入；严格权重/配置加载已单独通过 |
| Direct 草稿 | 未执行 | 上游输入缺失，且独立 prefix/LoRA 权重未提供 |
| Qwen 候选、embedding、selector 最终选择 | 未执行 | 没有可用的完整上游运行输入 |

第一次尝试发现原脚本用检查点父目录名判断训练 run，迁移到 `weights/coordinate_prior` 后触发错误。现已在严格验证 HR320 哈希之后恢复原 run 标识，同时保留实际本地路径；使用同一病例重新运行通过。没有更换病例。

旧 Dataset API 要求标签列，适配层沿用原运行源码生成的占位表，仅用于满足接口；模型 forward 使用明确的影像/坐标输入白名单，保存的专家输出剔除了标签字段。占位表不代表真实阴性标签，也不用于评价。

最终本地报告：`outputs/single_case_runtime_gaps_v2/single_case_result.json`。首轮失败记录保留在 `outputs/single_case_runtime_gaps/`，不覆盖历史结果。

## 仍需补的最小资产

### A. 冻结前凸几何常量

原服务器来源：

```text
/root/autodl-tmp/src/outputs/stage1_16a_visual_evidence_cache_v1/train_visual_evidence_cache.pt
```

只需要以下字段，不需要整份患者缓存：

```text
geometry_feature_names
metadata.lordosis_rule.flat_feature
metadata.lordosis_rule.flat_threshold
metadata.lordosis_rule.template_threshold
metadata.lordosis_rule.flat_scale
metadata.lordosis_rule.template_scale
metadata.lordosis_rule.normal_template
```

### B. 冻结坐标归一化常量

原服务器来源：

```text
/root/autodl-tmp/src/outputs/stage1_19_independent49_confirmatory_v1/06_external_segment_dataset/external_segment_ar_dataset.pt
```

只需要 `coordinate_feature_names`、`coordinate_normalizer_mean`、`coordinate_normalizer_std`。必须复制原冻结值，不能在单病例或测试队列上重新拟合。

已提供 `scripts/export_frozen_runtime_statistics.py`，可在原机器从这两个可信源文件导出最小 `.pt` 与哈希溯源记录。导出文件随后放入本地 `weights/runtime_assets/`。这两个最小导出文件与完整历史缓存哈希不同，因此应使用导出溯源记录验证，不能冒充原缓存文件的整文件哈希。

### C. Direct 分支选中的完整 epoch_2

原服务器目录：

```text
/root/autodl-tmp/src/outputs/qwen3_4b_direct_to_llm_ablation_v1/epoch_2/
  direct_prefix.pt
  lora/adapter_model.safetensors
  lora/adapter_config.json
```

建议导出完整目录并附 SHA-256 清单，安装目标为 `weights/direct/epoch_2/`。包内 `SELECTION.json` 确认选中 epoch 2，但没有附上这些权重；Qwen base 和三个 realizer adapter 均不能替代 Direct 自己的 prefix/LoRA。

## 还需要完成的工程工作

- 接收并验证上述 A/B/C 后，继续完成证据组装、归一化、Planner 和 Direct → realizer → selector 串联。
- 将原 Direct 脚本的 CUDA 强依赖适配为可选择设备，并实际验证本机 Qwen 生成。当前测试尚未到达这一阶段，不能承诺 CPU 耗时或最终输出。
- 运行入口中的自动选片、HR320、映射与三个专家已经接通；`test_single_case_pipeline.py` 当前是分阶段诊断入口，不是已完成的全流程产品入口。
- 历史回放不作为新患者推理替代。本机另发现 `LMCCR_HISTORICAL_REPLAY_ASSETS_20260920.zip`，只核查了其中两份张量资产的哈希与字段结构：没有发现所缺原始归一化常量，本次未拿其中患者特征替换实际影像输入。

## 运行命令

```powershell
.venv/Scripts/python.exe scripts/install_runtime_gaps.py --archive C:/path/LMCCR_FULL_PIPELINE_RUNTIME_GAPS_20260920.zip --supplemental-models /path/to/original/models
.venv/Scripts/python.exe scripts/validate_weight_bundle.py
.venv/Scripts/python.exe scripts/test_single_case_pipeline.py --case-id YOUR_CASE_ID --output outputs/new_single_case_attempt
```

诊断入口要求新的输出目录，保留先前运行记录；全流程未通过时返回非零退出码。模型兼容性验证与真实单病例执行是两份独立报告。
