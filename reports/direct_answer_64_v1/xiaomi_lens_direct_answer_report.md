# Xiaomi-Lens-0 CV-Bench Direct Answer 评测报告

- 实验规范名：`direct_answer_think_off_max_new_tokens_64_v1`
- 模型：`SeerRay-Lab/Xiaomi-Lens-0`（本地快照 `68f57cdd…`）
- Micromamba 环境：`xiaomi-lens`
- 运行时间：2026-11-01
- 结果目录：`experiments/cv_bench/direct_answer_64_v1/xiaomi_lens/`

---

## 1. 实验目的

在不输出显式思考过程的前提下，考察 Xiaomi-Lens-0 在 CV-Bench 上的空间理解能力，
并与同一协议下的 Qwen3.5-4B 对照。本实验只回答一个问题：

> 关闭显式 Thinking、强制模型直接输出选项后，准确率、生成 Token 和推理耗时分别是多少？

本实验**不是**“完全无推理”实验：模型不输出分析文字，不代表内部没有进行计算。
结论应表述为“关闭显式 Thinking 输出后的直接答案评测”。

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
| Transformers | 5.4.0 |
| Micromamba 环境 | `xiaomi-lens` |
| 模型精度 | `bfloat16` |
| Attention | `flash_attention_2` |
| Batch Size | 1 |
| 生成参数 | `do_sample=False`（贪心）、`max_new_tokens=64`、`seed=0` |

**GPU 共享说明**：评测期间整卡上另有 `graphnav` 进程（约 14 GiB）。
`experiment.json` 记录 `gpu_shared: true`，因此耗时只能视为“共享 GPU 条件下的测量值”。

---

## 3. 模型与权重

| 项目 | 值 |
|---|---|
| Hugging Face ID | `SeerRay-Lab/Xiaomi-Lens-0` |
| 本地快照 | `…/models--SeerRay-Lab--Xiaomi-Lens-0/snapshots/68f57cdde1a19f751e78b18af04efa8ca8bd7fce` |
| 模型类 | `XiaomiLensQwen3_5ForConditionalGeneration` |
| 加载方式 | `from_pretrained(..., dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda:0")` |
| 模型加载耗时 | 4.99 s |
| 加载后 allocated / reserved | 8.61 / 8.63 GiB |

模型只加载一次，随后顺序评测全部 2638 条样本。

---

## 4. 数据集

本地路径：`datasets/CV-Bench/`，使用 `test_2d.parquet` 与 `test_3d.parquet`，
未重新下载、未修改。

| 维度 | task | source | 数量 |
|---|---|---|---:|
| 2D | Count | ADE20K | 342 |
| 2D | Count | COCO | 446 |
| 2D | Relation | ADE20K | 291 |
| 2D | Relation | COCO | 359 |
| 3D | Depth | Omni3D | 600 |
| 3D | Distance | Omni3D | 600 |
| **合计** | | | **2638** |

2D = 1438，3D = 1200。样本 uid 形如 `2D:1234`，与 Qwen 实验完全一致。

---

## 5. Direct Answer Prompt（v1）

统一指令文件：`evaluation/direct_answer/prompts/direct_answer_v1.txt`

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

- 保留原始题目、选项内容与顺序；
- 不向模型提供 Ground Truth；
- Prompt 文件 SHA-1：`89db68802fb5edef5d4a97aafa041e003da75f4c`
- 完整 instruction 后缀 SHA-1：`3405f00036da52b4c0db835067a193d88d85ad1e`

---

## 6. Thinking Off 的实现

不是只靠 Prompt 里的“Do not provide reasoning”，而是同时使用模型自带的
chat template 布尔参数：

```python
prompt_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
```

渲染后检查 prompt 后缀必须是 `<think>\n\n</think>\n\n`：

```python
expected_suffix = "<think>\n\n</think>\n\n"
mode_verified = prompt_text.endswith(expected_suffix)
```

本次 2638 条样本的 `mode_verified` 全部为 `True`，否则程序会直接报错停止。

---

## 7. 输入流程

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
只解码新生成 token
        ↓
整段严格解析 FINAL_ANSWER: (X)
        ↓
与 Ground Truth 比较
```

计时全部在 `torch.cuda.synchronize()` 之后读取。

---

## 8. 严格答案解析

规则（`common/direct_parser.py`）：

1. 只解码新生成 token；
2. 剥离 `<|im_end|>`、`<|endoftext|>`、`<|mask|>` 等结构性特殊 token；
3. 去除首尾空白后，整段必须**恰好**是 `FINAL_ANSWER: (X)`；
4. 字母必须属于本题真实选项集合（2–6 个选项，A–F）；
5. 其他情况一律 `invalid`，并保留在 Accuracy 分母中。

不做全文搜索、不做选项文字模糊匹配、不根据 Ground Truth 反推。

**本次实际结果**：0 条 Invalid、0 条格式违规、0 条截断，`finish_reason` 全部为 `eos`。

---

## 9. 官方评分规则

```text
Accuracy_2D = (Accuracy_ADE20K + Accuracy_COCO) / 2
Accuracy_3D = Accuracy_Omni3D
CV-Bench Accuracy = (Accuracy_2D + Accuracy_3D) / 2
```

同时报告“全部样本普通 Accuracy”，两者因来源样本量不等而略有差异。

---

## 10. 整体准确率

| 指标 | 数值 |
|---|---:|
| **官方 CV-Bench 总分** | **87.04%** |
| 全部样本普通 Accuracy | 86.92%（2293/2638） |
| 有效回答率 | 100.00%（2638/2638） |
| Invalid | 0（0.00%） |
| 格式违规 | 0（0.00%） |
| 截断 | 0（0.00%） |
| 推理失败 | 0（0.00%） |

---

## 11. 2D / 3D 准确率

| 维度 | 正确 / 总数 | Accuracy |
|---|---|---:|
| 2D（官方加权） | — | **82.17%** |
| 3D（Omni3D） | 1103 / 1200 | **91.92%** |

2D 普通平均：1190 / 1438 = 82.75%。

---

## 12. 细分任务

| 任务 | 正确 / 总数 | Accuracy |
|---|---|---:|
| Object Counting（Count） | 571 / 788 | 72.46% |
| Spatial Relationship（Relation） | 619 / 650 | 95.23% |
| Depth Order（Depth） | 577 / 600 | **96.17%** |
| Relative Distance（Distance） | 526 / 600 | 87.67% |

---

## 13. 数据来源

| 来源 | 正确 / 总数 | Accuracy |
|---|---|---:|
| ADE20K | 489 / 633 | 77.25% |
| COCO | 701 / 805 | 87.08% |
| Omni3D | 1103 / 1200 | 91.92% |

各组合：

| 组合 | 正确 / 总数 | Accuracy |
|---|---|---:|
| 2D/Count/ADE20K | 217 / 342 | 63.45% |
| 2D/Count/COCO | 354 / 446 | 79.37% |
| 2D/Relation/ADE20K | 272 / 291 | 93.47% |
| 2D/Relation/COCO | 347 / 359 | 96.66% |
| 3D/Depth/Omni3D | 577 / 600 | 96.17% |
| 3D/Distance/Omni3D | 526 / 600 | 87.67% |

---

## 14. 性能指标

### 14.1 时间

| 指标 | 数值 |
|---|---:|
| 模型加载 | 4.99 s |
| 预热 | 1.39 s |
| 平均图片解码 | 0.0036 s |
| 平均 Processor 预处理 | 0.0231 s |
| 平均生成 | 0.2717 s |
| 平均端到端 | **0.2993 s** |
| 中位端到端 | 0.2767 s |
| P95 端到端 | **0.4039 s** |
| 最长单条 | 0.6482 s |
| 总评测墙钟 | 789.7 s（13.2 min） |
| 吞吐（samples/s） | 3.34 |
| Tokens/s（含 Prefill） | 29.45 |

`tokens/s = 生成 token 数 / generate 耗时`，**包含 Prefill**，不是纯 Decode 速度。

### 14.2 Token

| 指标 | 数值 |
|---|---:|
| 平均生成 Token | 8.00 |
| 中位生成 Token | 8 |
| P95 生成 Token | 8 |
| 最大生成 Token | 8 |
| 总生成 Token | 21104 |
| 平均 Prompt Token | 772.24 |

所有样本都恰好生成 8 个 token（答案行 + 结束标记），没有任何截断。

### 14.3 显存

| 指标 | 数值 |
|---|---:|
| 加载后 allocated | 8.61 GiB |
| 加载后 reserved | 8.63 GiB |
| 单样本峰值（平均 / 最大） | 8.81 / 9.36 GiB |
| 整体峰值 allocated / reserved | 9.36 / 9.45 GiB |

---

## 15. 与历史 8192 实验的参考对照

> **重要**：历史实验使用不同的 Prompt（允许分析）、`max_new_tokens=8192` 和
> 逐行解析规则，**不是严格对照组**，下表仅供观察趋势。

| 指标 | 历史 Thinking Off（8192） | 本次 Direct Answer（64） | 变化 |
|---|---:|---:|---:|
| 官方 CV-Bench | 87.76% | 87.04% | −0.72 pp |
| 2D | 83.44% | 82.17% | −1.27 pp |
| 3D | 92.08% | 91.92% | −0.16 pp |
| Count | 74.49% | 72.46% | −2.03 pp |
| Relation | 95.85% | 95.23% | −0.62 pp |
| Depth | 94.83% | 96.17% | **+1.33 pp** |
| Distance | 89.33% | 87.67% | −1.67 pp |
| 平均生成 Token | 137.24 | 8.00 | −129.24 |
| 平均 e2e | 4.98 s | 0.30 s | −4.68 s |
| P95 e2e | 12.42 s | 0.40 s | −12.02 s |

观察：

- 输出格式约束把生成量从约 137 token 压缩到 8 token，端到端耗时下降约 94%；
- 总体准确率只下降 0.72 pp，其中 Depth 反而提升；
- Count 与 Distance 的下降最明显，说明这两类任务可能更依赖逐步分析过程。

---

## 16. 可能存在的问题

1. **受限于固定格式**：本次严格要求一行答案，若模型原本需要中间步骤，准确率可能被低估。
2. **不是“无推理”**：只是不输出显式推理，内部计算仍然发生。
3. **GPU 共享**：评测期间有 `graphnav` 进程运行，绝对耗时只能作参考。
4. **模型可能自发输出 `<bbox>` / `<mask>`**：本次因输出极短，实际未出现；解析器会剥离结构性特殊 token。
5. **单一数据集**：CV-Bench 只覆盖四类任务，不能代表全部空间理解能力。
6. **与历史实验不可严格对比**：Prompt、生成长度和解析规则都不同。

---

## 17. 下一步建议

1. 对 Count 与 Distance 的 Direct Answer 错误做案例分析，判断是“缺少中间步骤”还是“类别/计数定义”问题。
2. 在同一 Direct Answer 协议下增加“允许一句简短理由”的对照，量化格式约束带来的损失。
3. 在 GPU 独占条件下复测耗时与吞吐，得到可靠的性能基线。
4. 扩展到 SPAR-Bench、VSI-Bench 等三维空间理解数据集，检验结论是否可迁移。

---

## 18. 结论

- Xiaomi-Lens-0 在 Direct Answer 协议下取得 **87.04%** 的官方 CV-Bench 总分
  （2D 82.17%，3D 91.92%）。
- **2638 条样本全部输出合法格式，无 Invalid、无截断**；平均仅生成 8 个 token。
- 相比历史 8192 的 Thinking Off，准确率仅下降 0.72 pp，但平均端到端耗时从
  4.98 s 降到 0.30 s（约 16.6×），是明显的效率收益。
- Depth Order 在直接回答下反而提升（94.83% → 96.17%），
  Count 与 Relative Distance 下降较多，提示这两类任务更需要显式分析。
