# Qwen3.5-4B 模型评测报告

> **整理日期：2026-11-02** · **已完成：CV-Bench 静态图片评测（3 组 × 2,638 题）**  
> 本文作为 **Qwen3.5-4B 的独立长期实验报告**，未来其他 Benchmark 可以继续按章节追加；官方模型资料与本地实测指标分开记录。  
> **独立 Markdown：** 所有图片、文档、代码及原始结果引用均为完整 HTTPS 链接，现有仓库资源固定到提交 `3f4618e6b0f7`，可以直接复制本文到其他平台阅读（在线资源需联网）。

## 1. 模型简介与官方资料

**Qwen3.5-4B** 是通用 4B 级多模态视觉语言模型，支持文本和图像输入，主要用于图片理解、视觉问答、OCR、数学图像理解、空间理解等场景。其语言主干使用混合结构（Gated DeltaNet 线性注意力与 Gated Attention），并通过原生视觉语言训练获得多模态能力。

| 项目 | 资料 |
|---|---|
| Hugging Face | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| 官方模型介绍 / 配置 / 推理代码 | [Qwen3.5-4B Model Card](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Qwen3.5 系列技术介绍 | [Qwen3.5: Towards Native Multimodal Agents（Qwen Team）](https://qwen.ai/blog?id=qwen3.5) |
| 官方项目入口 | [Qwen 官方 GitHub](https://github.com/QwenLM) |
| 官方展示与示例 | [Qwen3.5 官方博客](https://qwen.ai/blog?id=qwen3.5) / [Hugging Face 图像推理示例](https://huggingface.co/Qwen/Qwen3.5-4B) |

**论文说明：** 本文将模型卡所关联的 Qwen3.5 技术介绍作为官方技术资料，不将其他尺寸模型的独立论文误列为“Qwen3.5-4B 专属论文”。

**官方公开的任务（非本地复测）：** 模型卡覆盖通用视觉问答、OCR、数学与图表识别、空间智能等，并列出 CountBench、RefCOCO、EmbSpatialBench、RefSpatialBench 等相关评测。**本项目未完整复测上述所有官方 Benchmark**；下面的分数只来自本地 CV-Bench。

**官方宣传 / 演示：** [Qwen3.5 官方博客（含演示资料）](https://qwen.ai/blog?id=qwen3.5)，以及 [Qwen3.5-4B Hugging Face 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)。不引用未经核验的宣传图 URL。

## 2. 本地部署环境

| 项目 | 实验配置 |
|---|---|
| 系统 | Ubuntu 24.04.5 LTS / Linux 7.0.0-38-generic |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition（96 GB 级；PyTorch 可见约 94.96 GiB） |
| NVIDIA 驱动 | 580.178.04（驱动报告支持 CUDA 13.0） |
| Python / PyTorch CUDA | 3.11.18 / 2.10.0+cu128（运行时 CUDA 12.8） |
| Micromamba 环境 | `vlm-test` |
| Transformers | `5.20.2` |
| 模型类 | `Qwen3_5ForConditionalGeneration` + `AutoProcessor` |
| Attention / 精度 | `sdpa` / BF16 |
| Batch Size / 解码 | 1 / `do_sample=False` |
| 模型版本 | `Qwen/Qwen3.5-4B`；权重缓存快照 `851bf6e8…` |

**部署方法：** 使用 Hugging Face Transformers 的 `AutoProcessor` 构造图片与文本输入，由 `Qwen3_5ForConditionalGeneration.from_pretrained` 加载模型，在 GPU 运行 `generate()`。本项目使用 Transformers 5.20.2 与 `sdpa`，没有强行让 Qwen 与 Xiaomi 共用不同的模型类或 Processor。

可参考：[官方 Model Card / Quickstart](https://huggingface.co/Qwen/Qwen3.5-4B) · [本地模型加载适配器](https://github.com/chuiziTom/VLM-test/blob/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/evaluation/qwen35_4b/model_qwen35.py) · [环境安装说明](https://github.com/chuiziTom/VLM-test/blob/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/docs/environment_setup.md) · [历史评测入口](https://github.com/chuiziTom/VLM-test/blob/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/evaluation/qwen35_4b/run_qwen35_cvbench.py) · [Direct Answer 入口](https://github.com/chuiziTom/VLM-test/blob/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/evaluation/direct_answer/run_qwen35_direct_answer.py)。

## 3. 已完成任务：CV-Bench 静态图片

**CV-Bench：** [Hugging Face Dataset](https://huggingface.co/datasets/nyu-visionx/CV-Bench) · [Cambrian-1 论文](https://arxiv.org/abs/2406.16860)。

本次输入是**单张 RGB 图片 + 选择题问题和候选选项**，通过 [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) 的 `test_2d.parquet` 与 `test_3d.parquet` 全量测试模型：

| 子任务 | 测试内容 | 来源 | 题数 |
|---|---|---|---:|
| 2D Count | 物体计数 | ADE20K、COCO | 788 |
| 2D Relation | 左右上下等二维关系 | ADE20K、COCO | 650 |
| 3D Depth | 哪个物体更靠近相机 | Omni3D | 600 |
| 3D Distance | 哪个物体更靠近指定目标 | Omni3D | 600 |
| **总计** | **2D 1,438 / 3D 1,200** | | **2,638** |

![CV-Bench 2D 题目样本](https://raw.githubusercontent.com/chuiziTom/VLM-test/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/datasets/CV-Bench/samples/2d_overview.jpg)

![CV-Bench 3D 题目样本](https://raw.githubusercontent.com/chuiziTom/VLM-test/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/datasets/CV-Bench/samples/3d_overview.jpg)

样例问题如“What is the number of ...?”、“Which object is closer to the camera?”。**没有测试连续帧或视频，也没有用本次成绩推断 Qwen 的视频时序理解能力。**

### 3.1 评测流程和协议

```text
CV-Bench Parquet → 读图、问题和选项
   → Qwen Chat Template + AutoProcessor
   → enable_thinking=True/False
   → model.generate（贪心）
   → 解码新生成 Token，严格解析最终选项
   → 对照 Ground Truth
   → 统计 Accuracy / Token / 推理时间 / 显存 / Invalid
```

| 实验协议 | Thinking | max_new_tokens | 格式 |
|---|---|---:|---|
| Legacy 长回答 | Off | 8192 | 允许解释；最终 `FINAL_ANSWER: (X)` |
| Legacy 长回答 | On | 8192 | 先输出 `<think>…</think>`，只评分思考之后的最终答案 |
| Direct Answer v1 | Off | 64 | 整段必须为单行 `FINAL_ANSWER: (X)` |

两个模型评测共享的 CV-Bench 加载、选项解析及官方评分代码位于 [evaluation/common](https://github.com/chuiziTom/VLM-test/tree/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/evaluation/common)。Legacy On/Off 只切换 `enable_thinking`；Direct Answer 在关闭显式 Thinking 的同时强制短格式输出。无效输出、格式错误、截断均按错题计入分母。

**官方加权评分：**

```text
Accuracy_2D = (Accuracy_ADE20K + Accuracy_COCO) / 2
Accuracy_3D = Accuracy_Omni3D
CV-Bench = (Accuracy_2D + Accuracy_3D) / 2
```

此公式按数据来源加权，不完全等于 2,638 条样本的普通平均准确率。`max_new_tokens` 是上限，不代表固定生成数。

## 4. 本地实测结果

### 4.1 三组实验准确率（均为 2,638 / 2,638）

| 指标 | Legacy Off / 8192 | Legacy On / 8192 | Direct Off / 64 |
|---|---:|---:|---:|
| **CV-Bench 官方总分** | **86.76%** | 85.34% | 85.88% |
| 普通全样本 Accuracy | 86.73%（2288/2638） | 85.14%（2246/2638） | 85.63%（2259/2638） |
| 2D 官方加权 | 82.27% | 79.09% | 80.26% |
| 3D Omni3D | 91.25% | **91.58%** | 91.50% |
| Count | **72.59%** | 66.50% | 69.42% |
| Relation | 95.54% | **95.85%** | 94.46% |
| Depth | 93.00% | 93.17% | **94.83%** |
| Distance | 89.50% | **90.00%** | 88.17% |

按图像来源细分：

| 来源 | Legacy Off | Legacy On | Direct Off |
|---|---:|---:|---:|
| ADE20K（633） | **76.46%** | 73.46% | 76.30% |
| COCO（805） | **88.07%** | 84.72% | 84.22% |
| Omni3D（1200） | 91.25% | **91.58%** | 91.50% |

### 4.2 推理速度、Token 与回答有效性

| 指标 | Legacy Off / 8192 | Legacy On / 8192 | Direct Off / 64 |
|---|---:|---:|---:|
| **平均端到端耗时** | 10.64 s | 37.13 s | **0.334 s** |
| P95 端到端耗时 | 35.71 s | 214.31 s | **0.401 s** |
| 平均生成 Token | 118.56 | 1,365.24 | **9.98** |
| 平均 Thinking Token | 0 | 1,301.77 | 0 |
| Tokens/s（包含 Prefill） | 11.22 | 36.79 | 31.30 |
| PyTorch 全局峰值 allocated | 9.22 GiB | 9.22 GiB | 9.22 GiB |
| Invalid 数 | 1（0.04%） | **172（6.52%）** | 24（0.91%） |
| 截断数 | 0 | **172** | 0 |

注意：截断数通常包含在 Invalid 中，不可直接相加；计时测得的是共享 GPU 条件下的运行表现。Tokens/s 包含 Prefill，且不同生成长度会影响它的解释，不能只看这个数判断单 Token Decode 性能。

### 4.3 核心发现

**（1）Thinking On 的主要问题是长推理没有结束。**

在 Legacy 相同 Prompt、8192 Token 上限下，Qwen Thinking On 官方总分从 **86.76% 降至 85.34%**（−1.42 pp），平均生成 Token 从 **118.56 增至 1,365.24**，平均端到端耗时从 10.64 s 增至 37.13 s（3.49 倍）。其中 **172/2638 条（6.52%）**到达上限仍未输出可解析的最终答案，全部计为错误。

按任务统计，这 172 条无效回答中：
- Count：118 条；
- Relation：14 条；
- Depth：19 条；
- Distance：21 条。

它们占用了大量生成预算，也使完整准确率下降。**仅统计合法回答子集**时的普通 Accuracy 为 **91.08%**（诊断指标），但剔除这些失败样本会引入选择偏差，因此**绝不能替代 85.34% 的官方加权成绩**。

**（2）Direct Answer 明显缩短输出，但要关注格式稳定性。**

Direct Answer / Off / 64 将平均输出从 118.56 Token 降至 **9.98 Token**，端到端时延从 10.64 s 降至 **0.334 s**（约 −96.9%）。官方分由 86.76% 变成 85.88%（−0.88 pp）。与 Legacy Off 不同，这一实验同时修改了 Prompt、Token 上限和解析方式，不能视为 Thinking 单变量对照。

**（3）24 条 Direct Answer 格式违规值得单独诊断。**

Direct Answer 有 **24 条 Invalid（0.91%）**，其中 22 条是 Count 任务，另外 2 条属于 Distance；它们都不是截断，而是生成了额外文本，例如选项后又带上计数值。按统一严格协议必须计错。原有分析报告指出，其中有 17 条字母本身与正确答案相同，但**不能因事后观察而改变正式分数**。

**（4）任务差异：** Thinking On 使 Qwen 的 3D 指标从 91.25% 略升至 91.58%，但 2D 计数大幅下降（72.59% → 66.50%）。Depth 在 Direct Answer 中达到 94.83%，说明简单直接回答对部分空间判断任务可能更合适，但还需要控制实验验证。

## 5. 原始结果、诊断与后续扩展

| 数据与报告 | 完整 HTTPS 链接 |
|---|---|
| Legacy Off 原始实验 | [Qwen Thinking Off（metrics + predictions）](https://github.com/chuiziTom/VLM-test/tree/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/experiments/cv_bench/legacy_8192/qwen35_4b/thinking_off) |
| Legacy On 原始实验 | [Qwen Thinking On（metrics + predictions）](https://github.com/chuiziTom/VLM-test/tree/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/experiments/cv_bench/legacy_8192/qwen35_4b/thinking_on) |
| Direct Off 原始实验 | [Qwen Direct Answer（metrics + predictions）](https://github.com/chuiziTom/VLM-test/tree/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/experiments/cv_bench/direct_answer_64_v1/qwen35_4b) |
| Legacy 综合核验报告 | [Thinking Off/On 横向分析](https://github.com/chuiziTom/VLM-test/blob/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/reports/legacy_8192/cvbench_qwen35_xiaomi_comparison.md) |
| Direct Answer 详细报告 | [Qwen Direct Answer 单模型报告](https://github.com/chuiziTom/VLM-test/blob/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/reports/direct_answer_64_v1/qwen35_4b_direct_answer_report.md) |
| 独立复核工具 | [Legacy 复核](https://github.com/chuiziTom/VLM-test/blob/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/analysis/legacy_8192/verify_results.py) / [Direct 复核](https://github.com/chuiziTom/VLM-test/blob/3f4618e6b0f7a849df21d160519a3a0cde6b8ecf/analysis/direct_answer_64_v1/verify_and_compare.py) |
| 总报告及另一模型 | [CV-Bench 双模型总报告](https://github.com/chuiziTom/VLM-test/blob/main/reports/cv_bench_overall_report.md) / [Xiaomi-Lens-0 报告](https://github.com/chuiziTom/VLM-test/blob/main/reports/xiaomi_lens_0_evaluation.md) |

**本地核验命令（只读既有记录，不重新运行模型）：**

```bash
cd /home/20261024_1/Workspace/qyh/VLM-test
micromamba run -n vlm-test python -B analysis/legacy_8192/verify_results.py
micromamba run -n vlm-test python -B analysis/direct_answer_64_v1/verify_and_compare.py
```

### 5.1 实测案例补充区

后续可在此补充典型图片、真实模型输出、Ground Truth、错误原因和日志截图。所有新增截图应使用完整 HTTPS 链接，不使用本机绝对磁盘路径，也不从思考文本中猜测最终答案。

### 5.2 后续新实验的追加规范

未来增加其他 Benchmark（例如 SPAR-Bench、VSI-Bench），在下方继续新增 `## 6. <Benchmark / 协议名称>（日期）` 章节，按同一顺序写：

1. 数据集与任务范围（静态图 / 多视角 / 视频）；
2. 模型权重版本与部署环境；
3. Prompt、Thinking 开关和 Token 限制；
4. 评分口径、结果表、推理时间、Invalid/截断；
5. 案例图 / 日志与结果链接；
6. 局限与结论。

**不要覆盖历史 CV-Bench 成绩，也不要混用不同 Prompt / 解析器的实验。**

---

*本报告仅整理已完成的本地实验，不是 Qwen 官方 Benchmark 的复现报告；未重新执行模型推理。*
