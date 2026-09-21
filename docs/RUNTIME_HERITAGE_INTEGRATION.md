# 最后三组运行资产安装记录

日期：2026-09-20。来源：`LMCCR_RUNTIME_HERITAGE_PATCH_20260920.zip`。

按用户要求，本次仅整合资产、路径及读取格式；没有运行模型加载、单元测试、单病例推理或全流程验证。之前的测试结果不覆盖本次新增文件或适配代码。

## 已安装

```text
weights/runtime_assets/lordosis_geometry_constants.pt
weights/runtime_assets/coordinate_normalizer.pt
weights/direct/epoch_2/direct_prefix.pt
weights/direct/epoch_2/lora/adapter_model.safetensors
weights/direct/epoch_2/lora/adapter_config.json
```

包内 9 个带清单哈希的文件与 SHA-256 一致，包含全部五个运行资产。另有三份未被该清单覆盖的清单/导出状态文件，保留原样。源文件说明和安装记录位于 `weights/_packages/LMCCR_RUNTIME_HERITAGE_PATCH_20260920/`，均受 Git 忽略规则保护，没有上传。

## 工程适配

- `scripts/install_runtime_heritage.py`：可重复的安装入口；检查路径、哈希和已有文件冲突，不加载模型。
- 本地与示例资产配置新增几何常量、归一化常量和 Direct epoch_2 路径。
- 权重验证器和单病例诊断入口改为识别本包的两个最小统计文件名。
- `runtime/statistics.py::load_frozen_statistics`：读取最小导出文件，将顶层 `lordosis_rule` 在内存中适配为旧接口需要的 `metadata.lordosis_rule`；不改写原文件，不重新拟合统计。此适配尚未执行或测试。

## 当前边界

此前确认缺少的三组资产现已落盘，不能据此认定全流程通过。后续仍需完成证据组装、坐标归一化、Planner、Direct、realizer 和 selector 的串联及设备适配，再开展单病例验收。现有 `test_single_case_pipeline.py` 仍是前半段诊断入口，不是完整推理入口。

上次单病例报告与验证 JSON 保留原始内容，作为当时缺少资产的历史记录；没有将旧报告改写为新资产验证通过。

再次安装命令：

```powershell
.venv/Scripts/python.exe scripts/install_runtime_heritage.py --archive C:/path/LMCCR_RUNTIME_HERITAGE_PATCH_20260920.zip
```
