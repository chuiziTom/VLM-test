# VLM-test

**视觉语言模型（VLM）空间感知、几何推理与视频时序理解的本地实验仓库。**

本项目围绕一个问题展开：**模型能否从“看见并定位目标”，进一步做到可靠的空间关系理解、全局比较、多步推理，以及基于视频的时序判断与关键帧选择？**

当前包括两类实验：**标准化数据集评测**（CV-Bench、SPAR-Bench-Tiny）与**探索性定性测试**（书架/停车场几何理解、Qwen 视频空间理解）。所有定性观察均不等同于官方 Benchmark 结论。

> 更新：2026-11-13。当前纳入 Qwen3.5-4B、Xiaomi-Lens-0、Tencent WeVisOne-VL-4B；其中腾讯模型主要参与定位/分割的定性实验，**不在 CV-Bench、SPAR-Bench 的两模型正式对比成绩中**。

## 1. 模型与实验概览

| 模型 | 模型标识 | 本地环境 | 实验覆盖 |
|---|---|---|---|
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `vlm-test` | CV-Bench、SPAR-Bench-Tiny、书架/停车场、视频空间理解与关键帧 |
| Xiaomi-Lens-0 | `SeerRay-Lab/Xiaomi-Lens-0` | `xiaomi-lens` | CV-Bench、SPAR-Bench-Tiny、书架/停车场、BBox/Mask |
| WeVisOne-VL-4B | `tencent/WeVisOne-VL-4B` | `wevisone-vl` | 停车场自由提示词测试、官方原生检测/指代/分割诊断 |

**从哪里开始：** 看正式指标 → [CV-Bench 综合报告](reports/cv_bench_overall_report.md)、[SPAR-Bench-Tiny 综合报告](reports/spar_bench_tiny_comparison.md)；看图像案例 → [`evaluation/几何理解简单测试/`](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95)；看视频关键帧 → [Qwen 视频 Notebook](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/qwen3.5-4B_%E8%A7%86%E9%A2%91%E7%A9%BA%E9%97%B4%E7%90%86%E8%A7%A3%E4%B8%8E%E5%85%B3%E9%94%AE%E5%B8%A7%E6%B5%8B%E8%AF%95.ipynb)。

## 2. 标准化数据集评测

### 2.1 CV-Bench（2,638 题）

已完成 Xiaomi 与 Qwen 的 **6 组全量实验**：`legacy_8192` 的 Thinking Off/On（两模型各 2 组），以及 `direct_answer_64_v1`（两模型各 1 组）。

| 模型 | Legacy / Thinking Off（8192） | Legacy / Thinking On（8192） | Direct Answer / Off（64） |
|---|---:|---:|---:|
| Xiaomi-Lens-0 | 87.76% | 87.29% | 87.04% |
| Qwen3.5-4B | 86.76% | 85.34% | 85.88% |

以上为**本地 CV-Bench 官方加权准确率**，不是模型发布方榜单。结果详情见 [历史 Off/On 对比](reports/legacy_8192/cvbench_qwen35_xiaomi_comparison.md)、[Direct Answer 对比](reports/direct_answer_64_v1/cvbench_direct_answer_comparison.md)；实验清单见 [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) 与 [experiment_manifest.json](experiment_manifest.json)。

注意：Legacy 与 Direct Answer **不仅改变生成长度，也改变了 Prompt 和答案解析**，不能把两列差异直接归因于 Thinking 或 Token 数。

### 2.2 SPAR-Bench-Tiny（RGB；1,000 题，20 类任务）

覆盖单视角与多视角空间任务；Xiaomi/Qwen 各运行 Thinking Off、Thinking On、Direct Answer 三种协议，共 **6 组**，报告记录了独立复算与校验。

| 协议 | Xiaomi Official / Strict | Qwen Official / Strict |
|---|---:|---:|
| Thinking Off / 8192 | 26.23 / 24.92 | 36.18 / 25.27 |
| Thinking On / 8192 | 30.70 / 30.12 | 27.01 / 26.94 |
| Direct Answer Off / 64 | 39.25 / 39.12 | 43.45 / 41.17 |

单位为百分数。`Official` 使用复现的官方评分口径；`Strict` 额外要求严格答案格式。特别是 Thinking Off 的 Qwen，两种口径相差明显，**不能只看 Official 总分评价空间能力**。Thinking On 也存在较多生成截断/未闭合情况。

- [SPAR 综合对比](reports/spar_bench_tiny_comparison.md) · [结果核验](reports/spar_bench_tiny_verification.md) · [评测代码使用说明](evaluation/spar_bench/README.md)
- 当前 GitHub `main` 上已上传评测代码与汇总报告；`experiments/spar_bench/tiny_rgb/` 原始大规模预测目录**未见于当前仓库**，本地归档与 GitHub 上传状态需区分。

## 3. 图像几何理解：书架与停车场

### 3.1 基础书架测试

包含二维网格定位、按顺序计数到指定书籍、颜色/文字识别，以及提示词语言和 Thinking 模式影响。

- [早期书架测试与人工核对报告](reports/geometry_understanding_simple_test.md)
- [Qwen 基础测试 Notebook](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/qwen3.5-4B.ipynb) · [Xiaomi 基础测试 Notebook](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/xiaomi-lens-0.ipynb)

### 3.2 复杂停车场测试（两张图片，四类任务）

使用 [停车场图1](datasets/%E5%87%A0%E4%BD%95%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/%E5%81%9C%E8%BD%A6%E5%9C%BA%E5%9B%BE1.png)、[停车场图2](datasets/%E5%87%A0%E4%BD%95%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/%E5%81%9C%E8%BD%A6%E5%9C%BA%E5%9B%BE2.jpg)，以尽可能一致的任务措辞与图像中心参考系观察：

1. **单目标定位**：找到指定红车并描述图像中心相对方位。
2. **多目标定位**：找出全部蓝色车辆，检查漏检与误检。
3. **全局极值比较**：选出全图位置最低的蓝车。
4. **多步空间关系**：从 A 出发“向图像左侧移动一列、再向上移动一行”寻找 B。

**原始 Notebook：**

| 模型 / 协议 | 实验文件 |
|---|---|
| Xiaomi-Lens-0（自由提示词） | [xiaomi-lens-0 测试v2.0.ipynb](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/xiaomi-lens-0%20%E6%B5%8B%E8%AF%95v2.0.ipynb) |
| Qwen3.5-4B（自由提示词） | [qwen3.5-4B_停车场几何测试v2.0.ipynb](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/qwen3.5-4B_%E5%81%9C%E8%BD%A6%E5%9C%BA%E5%87%A0%E4%BD%95%E6%B5%8B%E8%AF%95v2.0.ipynb) |
| WeVisOne-VL-4B（自由提示词） | [WeVisOne-VL-4B_停车场几何测试v2.0.ipynb](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/WeVisOne-VL-4B_%E5%81%9C%E8%BD%A6%E5%9C%BA%E5%87%A0%E4%BD%95%E6%B5%8B%E8%AF%95v2.0.ipynb) |
| WeVisOne-VL-4B（**官方原生任务独立诊断**） | [WeVisOne-VL-4B_停车场_官方原生任务独立诊断.ipynb](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/WeVisOne-VL-4B_%E5%81%9C%E8%BD%A6%E5%9C%BA_%E5%AE%98%E6%96%B9%E5%8E%9F%E7%94%9F%E4%BB%BB%E5%8A%A1%E7%8B%AC%E7%AB%8B%E8%AF%8A%E6%96%AD.ipynb) |

**目前的定性观察：** 单目标定位通常优于密集场景全局检索；漏检会影响后续极值选择；BBox/Mask 的存在并不保证文字空间关系正确。腾讯原生 `detection/referring/segmentation` 接口改善了输出格式和 Mask 生成，但没有消除测试中的全局漏检与空间变换错误。

**解释限制：** Xiaomi/Qwen 在个别任务中出现重复 BBox、Mask Token 或 Thinking 未结束的生成退化；这些结果应单独记作**当前生成配置下未完成/不稳定**，不能都算作“完成推理但回答错误”。三者分割接口、输出格式及 Thinking 支持并不一致；腾讯原生任务诊断**不用于三模型同条件准确率比较**。

## 4. Qwen3.5-4B 视频空间理解与关键帧选择

实验入口：[qwen3.5-4B_视频空间理解与关键帧测试.ipynb](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/qwen3.5-4B_%E8%A7%86%E9%A2%91%E7%A9%BA%E9%97%B4%E7%90%86%E8%A7%A3%E4%B8%8E%E5%85%B3%E9%94%AE%E5%B8%A7%E6%B5%8B%E8%AF%95.ipynb)。

测试素材：[2 秒合成车辆运动视频（MP4）](evaluation/%E5%87%A0%E4%BD%95%E7%90%86%E8%A7%A3%E7%AE%80%E5%8D%95%E6%B5%8B%E8%AF%95/qwen_video_outputs/qwen_spatial_demo_2s.mp4)（12 FPS，共 24 帧）；红车从蓝车左侧向右侧移动，蓝车静止，便于核对真值。

实验区分三件事：

- **人工均匀采样 8 帧后输入模型**：验证运动方向、初末关系和从候选帧选关键帧。
- **直接输入 MP4**：确认处理器产生 `pixel_values_videos`、`video_grid_thw`，而非逐帧独立调用。
- **关闭自动采样，输入短视频全部 24 帧后让模型按任务选时间戳**：检查关键事件发生时刻、从原视频提取对应关键帧。

**观察：** Qwen 能正确描述明显的“红车向右、蓝车静止”和初末左右关系变化；但对具体的越过时刻、距离最近时刻以及关键帧时间戳定位出现错误。模型输出时间戳/编号后，实际图片由 Python 从源视频提取，**不是模型直接生成图片像素**。

这只是单段 2 秒合成视频的功能验证，尚不能证明复杂真实视频或无人机 VLN 场景中的稳定关键帧选择能力。

## 5. 项目目录

```text
VLM-test/
├── models/
│   ├── Xiaomi-Lens-0/               # 小米官方源码副本、推理与环境说明
│   └── WeVisOne-VL/                 # 腾讯官方源码副本、教程与解码器
├── datasets/
│   ├── CV-Bench/
│   ├── SPAR-Bench-Tiny/             # GitHub 上主要为下载说明
│   ├── VSI-Bench/
│   └── 几何简单测试/               # 书架和停车场图片
├── evaluation/
│   ├── common/                      # CV-Bench 公共模块
│   ├── cv-bench/                    # Qwen / Xiaomi / Direct Answer 脚本
│   ├── spar_bench/                  # SPAR 数据、模型适配、评分与核验
│   └── 几何理解简单测试/           # 三模型图像、Qwen 视频、腾讯原生诊断
├── experiments/
│   └── cv_bench/                    # GitHub 已归档的 CV-Bench 逐样本结果
├── reports/                         # CV / SPAR / 基础几何测试报告
├── analysis/                        # CV-Bench 独立复算工具
├── docs/                            # 说明文档（部分仍采用旧路径示例）
├── environment-vlm-test.yml
├── requirements-vlm-test.txt
├── environment-wevisone-vl.yml
├── EXPERIMENT_INDEX.md
└── README.md
```

## 6. 环境与运行入口

三个模型的依赖差异较大，**不要合并环境**：

| 环境 | Python | PyTorch | Transformers | 说明 |
|---|---|---|---|---|
| `vlm-test` | 3.11 | 2.10.0+cu128 | 5.20.2 | Qwen3.5-4B |
| `xiaomi-lens` | 3.11 | 2.10.0+cu128 | 5.4.0 | Xiaomi-Lens-0，包含自定义 Mask 解码 |
| `wevisone-vl` | 3.12 | 2.11.0+cu130 | 5.12.1 | WeVisOne-VL-4B，官方 `WeVisOneVL` 接口 |

- Qwen：[environment-vlm-test.yml](environment-vlm-test.yml) + [requirements-vlm-test.txt](requirements-vlm-test.txt)
- Xiaomi：[environment-xiaomi-lens.yml](models/Xiaomi-Lens-0/environment-xiaomi-lens.yml) + [ENV_SETUP.md](models/Xiaomi-Lens-0/ENV_SETUP.md)
- Tencent：[environment-wevisone-vl.yml](environment-wevisone-vl.yml) + [ENV_SETUP_WeVisOne-VL.md](ENV_SETUP_WeVisOne-VL.md)

从仓库根目录可先做环境检查：

```bash
micromamba run -n vlm-test python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
micromamba run -n xiaomi-lens python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
micromamba run -n wevisone-vl python -c "import wevisone_vl; print(wevisone_vl.__file__)"
```

**Notebook 使用：** 在对应环境中选择 Jupyter Kernel，按单元格运行。停车场图片使用相对项目根目录的数据路径；腾讯原生诊断请使用 `wevisone-vl` 环境。模型权重通常需要单独从 Hugging Face 下载或使用本地缓存，不在此仓库内。

**CV-Bench 重跑注意：** 当前脚本已位于 `evaluation/cv-bench/`，但部分代码和旧说明仍按迁移前 `evaluation/qwen35_4b/` 等路径计算项目根目录，直接照旧命令重跑可能发生路径/导入错误；原有实验归档与报告不受影响。需要重新执行时，先核对脚本 `ROOT`、模型适配器路径和本地 `dataset -> datasets` 兼容链接。这里不对尚未修正的入口声称开箱即用。

SPAR-Bench 的评测参数和运行命令以 [evaluation/spar_bench/README.md](evaluation/spar_bench/README.md) 为准；依赖本地准备好的完整数据集。

## 7. 复现实验与结论边界

- **原始预测优先**：判断模型是否真的找到目标，以原始文字、BBox/Mask 以及保存的图像输出为依据，不能只看绘图效果。
- **区分失败类型**：视觉漏检、定位错误、空间推理错误、输出格式不匹配、重复生成/截断应分别标记。
- **不要混合评分口径**：CV-Bench 与 SPAR-Bench、Official 与 Strict、正式 Benchmark 与停车场/视频定性测试均不是同一种分数。
- **视觉能力不等于导航能力**：当前工作主要测空间感知、单图/短视频时序理解；尚未搭建完整的无人机 VLN 闭环导航 Benchmark。
- **优先保留可追溯记录**：模型版本、环境、Prompt、Thinking、生成参数、输入图片/视频、原始回答和可视化应一起保存。

后续方向：从**单图目标感知 → 多帧空间关系变化 → 指令相关关键帧 → 无人机 VLN 导航决策**逐步推进。现阶段结果支持继续研究，但不足以对三模型整体空间智能水平给出普遍排名。
