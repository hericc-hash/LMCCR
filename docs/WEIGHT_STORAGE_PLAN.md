# 权重普通 Git / LFS 存储建议

核查日期：2026-09-21。按当前工作区实际文件大小分析；未修改 Git 跟踪规则或上传文件。以下分类是存储建议，不替代权重内容和再分发许可审核。

## 结论

排除安装包、历史源码快照和重复补件后，运行目录包含 22 个权重文件，共 **936.02 MiB**。

建议小于 10 MiB 的 11 个稳定权重/常量文件走普通 Git（合计 **15.34 MiB**）；其余 11 个文件走 LFS（合计 **920.68 MiB**）。10 MiB 是本工程的维护建议，不是 GitHub 强制限制。若以后权重更新频繁，也可以将所有模型二进制统一交给 LFS。

GitHub 普通 Git 对超过 50 MiB 的文件发出警告，超过 100 MiB 的文件拒绝接收；网页上传单文件上限为 25 MiB。[官方文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

## 可以普通 Git 管理

以下路径相对于仓库根目录；总计 11 个文件。

| 文件 | 大小 | 建议 |
|---|---:|---|
| `weights/apreb/apreb.pt` | 2.003 MiB | 普通 Git |
| `weights/ccmrt/ccmrt.pt` | 3.963 MiB | 普通 Git |
| `weights/clinical_planner/05_stage2_3B_selected_candidate.pt` | 2.514 MiB | 普通 Git |
| `weights/selector/fact_seed20260931.pt` | 0.182 MiB | 普通 Git |
| `weights/selector/fact_seed20260932.pt` | 0.182 MiB | 普通 Git |
| `weights/selector/fact_seed20260933.pt` | 0.182 MiB | 普通 Git |
| `weights/selector/text_listwise_seed20260931.pt` | 2.101 MiB | 普通 Git |
| `weights/selector/text_listwise_seed20260932.pt` | 2.101 MiB | 普通 Git |
| `weights/selector/text_listwise_seed20260933.pt` | 2.101 MiB | 普通 Git |
| `weights/runtime_assets/coordinate_normalizer.pt` | 4,501 字节 | 普通 Git，作为运行常量 |
| `weights/runtime_assets/lordosis_geometry_constants.pt` | 3,253 字节 | 普通 Git，作为运行常量 |

## 应使用 LFS 的文件

| 文件 | 大小 | 原因 |
|---|---:|---|
| `weights/visual_experts/disc/best_common.pt` | 132.568 MiB | 超过普通 Git 100 MiB 限制 |
| `weights/visual_experts/stenosis/best_fused_common.pt` | 183.117 MiB | 超过普通 Git 100 MiB 限制 |
| `weights/visual_experts/nerve/best_fused_common.pt` | 266.132 MiB | 超过普通 Git 100 MiB 限制 |
| `weights/coordinate_prior/hr320_v7.pt` | 58.917 MiB | 超过 50 MiB 警告线，推荐 LFS |
| `weights/direct_auxiliary_planner/best_stage1_18c_explicit_clinical_mediator.pt` | 60.543 MiB | 超过 50 MiB 警告线，推荐 LFS |
| `weights/report_realizer/language_adapter/adapter_model.safetensors` | 45.037 MiB | 普通 Git 技术上允许；推荐避免大二进制历史膨胀 |
| `weights/report_realizer/dual_channel_adapter/adapter_model.safetensors` | 45.037 MiB | 同上 |
| `weights/report_realizer/scaffold_adapter/adapter_model.safetensors` | 45.037 MiB | 同上 |
| `weights/direct/epoch_2/lora/adapter_model.safetensors` | 45.037 MiB | 同上 |
| `weights/direct/epoch_2/direct_prefix.pt` | 23.178 MiB | 中型二进制，推荐与 Direct LoRA 一起管理 |
| `weights/visual_evidence/lumbar_visual_evidence.pt` | 16.078 MiB | 中型二进制，推荐与视觉权重一起管理 |

前三个文件若要作为仓库跟踪文件发布，必须使用 LFS；也可以改用 GitHub Release 等独立下载方式，而不将二进制纳入版本树。其余八个文件使用 LFS 是维护建议。

所有这些单文件都低于 GitHub Free/Pro 的 LFS 单文件 2 GB 上限。LFS 的每次权重更新按完整新文件计入存储，下载消耗仓库所有者带宽，因此不宜频繁提交训练中间检查点。[LFS 上限](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) · [计费方式](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)

## 不要直接把整个 weights 目录加入 Git

- `weights/runtime_gaps/08_selector_missing_ensemble_weights/` 中的四个补件，经 SHA256 比对，与 `weights/selector/` 下对应文件相同。仅保留规范目录的一份发布，不重复跟踪。
- `weights/_packages/` 是安装记录；`weights/runtime_gaps/` 和 `weights/runtime/` 包含补件、来源快照等内容，不应按“权重目录”整体上传。需要公开的源码和配置应先归入正式源码/配置目录。
- `weights/third_party/Qwen3_4B_Instruct_2507/` 当前没有模型文件。建议仓库保留上游模型地址和精确 revision，由使用者从 [Qwen 官方模型仓库](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 下载，避免重复托管整个基座模型。
- `.json`、`.yaml`、模型配置和 SHA256 清单一般适合普通 Git。先去除服务器绝对路径；`CANONICAL_LEXICALIZER.json` 含文本及诊断内容，不应仅因文件小就自动发布。

## 当前跟踪状态

当前 `.gitignore` 排除了 `weights/**/*`、`*.pt`、`*.safetensors` 等，仓库根目录没有 `.gitattributes`。因此即使安装 LFS 或执行 `git lfs track`，被忽略的权重也不会自动加入仓库。

实施时应先在 `.gitignore` 末尾精确放行已选定的文件，再为上表 LFS 文件添加 `.gitattributes` 规则。不要全局取消 weights 的忽略，也不要混入病例数据、生成报告或重复补件。本次仅分析，未更改这些规则。
