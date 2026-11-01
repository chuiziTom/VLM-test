# Qwen3.5-4B CV-Bench Direct Answer 评测报告

- 实验规范名：`direct_answer_think_off_max_new_tokens_64_v1`
- 模型：`Qwen/Qwen3.5-4B`（本地快照 `851bf6e8…`）
- Micromamba 环境：`vlm-test`
- 运行时间：2026-11-01
- 结果目录：`eval_cvbench_direct_answer/results/direct_answer_think_off_max_new_tokens_64_v1/qwen35_4b/`

---

## 1. 实验目的

在不输出显式思考过程的前提下，考察 Qwen3.5-4B 在 CV-Bench 上的空间理解能力，
并与同一协议下的 Xiaomi-Lens-0 对照。本实验只回答一个问题：

> 关闭显式 Thinking、强制模型直接输出选项后，准确率、生成 Token 和推理耗时分别是多少？

本实验**不是**“完全无推理”实验：模型不输出分析文字，不代表内部没有进行计算。

---

## 2. 测试环境

| 项目 | 值 |
|---|---|
| 操作系统 | Linux 7.0.0-38-generic（x86_64） |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU 显存 | 94.96 GiB |
| NVIDIA Driver | 580.178.04 |
| CUDA Runtime | 12.8（PyTorch `2.10.0+cu128`） |
| Python | 3.11.18 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.20.2 |
| Micromamba 环境 | `vlm-test` |
| 模型精度 | `bfloat16` |
| Attention | `sdpa` |
| Batch Size | 1 |
| 生成参数 | `do_sample=False`（贪心）、`max_new_tokens=64`、`seed=0` |

**GPU 共享说明**：评测期间整卡上另有 `graphnav` 进程（约 14 GiB）。
`experiment.json` 记录 `gpu_shared: true`，耗时只能视为“共享 GPU 条件下的测量值”。

---

## 3. 模型与权重

| 项目 | 值 |
|---|---|
| Hugging Face ID | `Qwen/Qwen3.5-4B` |
| 本地快照 | `…/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| 模型类 | `Qwen3_5ForConditionalGeneration` |
| Processor | `AutoProcessor`（tokenizer + 图像处理器） |
| 加载方式 | `from_pretrained(..., dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0")` |
| 模型加载耗时 | 3.62 s |
| 加载后 allocated / reserved | 8.46 / 8.47 GiB |

模型只加载一次，随后顺序评测全部 2638 条样本。

---

## 4. 数据集

本地路径：`dataset/CV-Bench/`，使用 `test_2d.parquet` 与 `test_3d.parquet`。

| 维度 | task | source | 数量 |
|---|---|---|---:|
| 2D | Count | ADE20K | 342 |
| 2D | Count | COCO | 446 |
| 2D | Relation | ADE20K | 291 |
| 2D | Relation | COCO | 359 |
| 3D | Depth | Omni3D | 600 |
| 3D | Distance | Omni3D | 600 |
| **合计** | | | **2638** |

2D = 1438，3D = 1200。样本 uid 与 Xiaomi 实验完全一致，可逐题对齐。

---

## 5. Direct Answer Prompt（v1）

与 Xiaomi 使用**逐字符相同**的指令文件：
`eval_cvbench_direct_answer/prompts/direct_answer_v1.txt`

```text
Answer the multiple-choice question directly.

Do not provide explanations, reasoning, or analysis.
Output exactly one line in the following format:

FINAL_ANSWER: (X)

where X is one of the provided option letters.

Do not output anything else.
```

实际输入：

```text
instruction = CV-Bench 原始 prompt（问题 + 选项） + "\n\n" + 上述指令
```

- Prompt 文件 SHA-1：`89db68802fb5edef5d4a97aafa041e003da75f4c`
- instruction 后缀 SHA-1：`3405f00036da52b4c0db835067a193d88d85ad1e`

---

## 6. Thinking Off 与 Chat Template

使用 Qwen checkpoint 自带 chat template 的官方布尔参数：

```python
prompt_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
```

渲染后校验后缀：

```python
expected_suffix = "<think>\n\n</think>\n\n"
mode_verified = prompt_text.endswith(expected_suffix)
```

本次 2638 条样本的 `mode_verified` 全部为 `True`。

---

## 7. 输入与生成流程

```text
PIL 图片 + instruction
        ↓
apply_chat_template(enable_thinking=False)
        ↓
Processor：文本 tokenize + 图像切 patch 归一化
        ↓
input_ids / pixel_values → GPU
        ↓
model.generate(max_new_tokens=64, do_sample=False)
        ↓
只解码新生成 token（保留特殊 token，供解析器剥离）
        ↓
整段严格解析 FINAL_ANSWER: (X)
        ↓
与 Ground Truth 比较
```

---

## 8. 严格答案解析

规则（`common/direct_parser.py`）：

1. 只解码新生成 token；
2. 剥离 `<|im_end|>`、`<|endoftext|>` 等结构性特殊 token；
3. 去除首尾空白后，整段必须**恰好**是 `FINAL_ANSWER: (X)`；
4. 字母必须属于本题真实选项集合；
5. 其他情况一律 `invalid`，并保留在 Accuracy 分母中。

---

## 9. 官方评分规则

```text
Accuracy_2D = (Accuracy_ADE20K + Accuracy_COCO) / 2
Accuracy_3D = Accuracy_Omni3D
CV-Bench Accuracy = (Accuracy_2D + Accuracy_3D) / 2
```

---

## 10. 整体准确率

| 指标 | 数值 |
|---|---:|
| **官方 CV-Bench 总分** | **85.88%** |
| 全部样本普通 Accuracy | 85.63%（2259/2638） |
| 有效回答率 | 99.09%（2614/2638） |
| Invalid | 24（0.91%） |
| 格式违规 | 24（0.91%） |
| 截断 | 0（0.00%） |
| 推理失败 | 0（0.00%） |

### 24 条 Invalid 的具体形态

全部为 `extra_text`，两类：

| 形态 | 数量 | 示例 |
|---|---:|---|
| 选项字母 + 计数值，缺少 `FINAL_ANSWER` 前缀 | 22 | `(A) 1<|im_end|>` |
| 有 `FINAL_ANSWER` 但后面还有额外文字 | 2 | `FINAL_ANSWER: (B) do…<|im_end|>` |

其中 22 条以 `(X)` 开头的样本里，有 17 条的字母恰好等于 Ground Truth。
按严格协议，它们全部记为错误（不从分母删除）。

**诊断性说明（不改变正式成绩）**：如果放宽为“接受行首 `(X)`”，官方总分约为
86.41%，比严格口径高 0.53 pp。本实验不采用该放宽规则，因为它会违反“只接受
统一格式、禁止从额外文本中提取答案”的协议。

---

## 11. 2D / 3D 准确率

| 维度 | 正确 / 总数 | Accuracy |
|---|---|---:|
| 2D（官方加权） | — | **80.26%** |
| 3D（Omni3D） | 1098 / 1200 | **91.50%** |

2D 普通平均：1161 / 1438 = 80.74%。

---

## 12. 细分任务

| 任务 | 正确 / 总数 | Accuracy |
|---|---|---:|
| Object Counting（Count） | 547 / 788 | 69.42% |
| Spatial Relationship（Relation） | 614 / 650 | 94.46% |
| Depth Order（Depth） | 569 / 600 | 94.83% |
| Relative Distance（Distance） | 529 / 600 | 88.17% |

---

## 13. 数据来源

| 来源 | 正确 / 总数 | Accuracy |
|---|---|---:|
| ADE20K | 483 / 633 | 76.30% |
| COCO | 678 / 805 | 84.22% |
| Omni3D | 1098 / 1200 | 91.50% |

各组合：

| 组合 | 正确 / 总数 | Accuracy |
|---|---|---:|
| 2D/Count/ADE20K | 213 / 342 | 62.28% |
| 2D/Count/COCO | 334 / 446 | 74.89% |
| 2D/Relation/ADE20K | 270 / 291 | 92.78% |
| 2D/Relation/COCO | 344 / 359 | 95.82% |
| 3D/Depth/Omni3D | 569 / 600 | 94.83% |
| 3D/Distance/Omni3D | 529 / 600 | 88.17% |

---

## 14. 性能指标

### 14.1 时间

| 指标 | 数值 |
|---|---:|
| 模型加载 | 3.62 s |
| 预热 | 1.23 s |
| 平均图片解码 | 0.0028 s |
| 平均 Processor 预处理 | 0.0120 s |
| 平均生成 | 0.3187 s |
| 平均端到端 | **0.3343 s** |
| 中位端到端 | 0.3184 s |
| P95 端到端 | **0.4013 s** |
| 最长单条 | 0.6132 s |
| 总评测墙钟 | 882.1 s（14.7 min） |
| 吞吐（samples/s） | 2.99 |
| Tokens/s（含 Prefill） | 31.30 |

### 14.2 Token

| 指标 | 数值 |
|---|---:|
| 平均生成 Token | 9.98 |
| 中位生成 Token | 10 |
| P95 生成 Token | 10 |
| 最大生成 Token | 11 |
| 总生成 Token | 26316 |
| 平均 Prompt Token | 772.24 |

正常情况下输出 10 个 token（答案行 + `<|im_end|>` + `<|endoftext|>`）。
没有样本达到 64 token 上限。

### 14.3 显存

| 指标 | 数值 |
|---|---:|
| 加载后 allocated | 8.46 GiB |
| 加载后 reserved | 8.47 GiB |
| 单样本峰值（平均 / 最大） | 8.66 / 9.22 GiB |
| 整体峰值 allocated / reserved | 9.22 / 9.30 GiB |

---

## 15. 与历史 8192 实验的参考对照

> **重要**：历史实验使用不同的 Prompt、`max_new_tokens=8192` 和逐行解析规则，
> **不是严格对照组**，下表仅供观察趋势。

| 指标 | 历史 Thinking Off（8192） | 本次 Direct Answer（64） | 变化 |
|---|---:|---:|---:|
| 官方 CV-Bench | 86.76% | 85.88% | −0.88 pp |
| 2D | 82.27% | 80.26% | −2.00 pp |
| 3D | 91.25% | 91.50% | +0.25 pp |
| Count | 72.59% | 69.42% | −3.17 pp |
| Relation | 95.54% | 94.46% | −1.08 pp |
| Depth | 93.00% | 94.83% | **+1.83 pp** |
| Distance | 89.50% | 88.17% | −1.33 pp |
| 平均生成 Token | 118.56 | 9.98 | −108.58 |
| 平均 e2e | 10.64 s | 0.33 s | −10.31 s |
| P95 e2e | 35.71 s | 0.40 s | −35.31 s |

观察：Depth 在直接回答下明显提升，Count 下降最多；总体只下降 0.88 pp，
但生成量和耗时下降约两个数量级。

---

## 16. 可能存在的问题

1. **格式约束带来的额外损失**：24 条 Invalid 中有 17 条字母本身正确，
   严格协议下仍计错；这是协议要求，不是解析 bug。
2. **不是“无推理”**：只是不输出显式推理。
3. **GPU 共享**：评测期间有 `graphnav` 进程运行，绝对耗时只能作参考。
4. **Attention 与 Transformers 版本**：本模型用 `sdpa` + Transformers 5.20.2，
   与 Xiaomi 的 `flash_attention_2` + 5.4.0 不同，速度不能严格对齐。
5. **单一数据集**：CV-Bench 只覆盖四类任务。
6. **与历史实验不可严格对比**：Prompt、生成长度和解析规则都不同。

---

## 17. 下一步建议

1. 分析 24 条格式违规：模型倾向输出 `(A) 1`（选项 + 计数），可在不改变协议的前提下单独统计这类行为。
2. 对 Count 与 Distance 的 Direct Answer 错误做案例分析。
3. 在 GPU 独占条件下复测耗时与吞吐。
4. 扩展到 SPAR-Bench、VSI-Bench 等三维空间理解数据集。

---

## 18. 结论

- Qwen3.5-4B 在 Direct Answer 协议下取得 **85.88%** 的官方 CV-Bench 总分
  （2D 80.26%，3D 91.50%）。
- 有效回答率 99.09%，24 条格式违规全部计错；无截断、无推理失败。
- 平均仅生成 9.98 个 token，平均端到端 0.334 s；相比历史 8192 的 Thinking Off，
  耗时下降约 31.8×，准确率仅下降 0.88 pp。
- 与 Xiaomi-Lens-0 同协议对比，Qwen 在 Count、Relation、Depth 上略低，
  在 Distance 上略高（详见综合对比报告）。
