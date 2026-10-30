# Qwen3.5-4B × CV-Bench 评测

本目录用**和 Xiaomi-Lens-0 完全相同的评测协议**，评测 `Qwen/Qwen3.5-4B` 在 CV-Bench
测试集上的空间理解能力，并对比 Thinking Off / Thinking On 两种模式。

> 优先级：**评分正确性 > 两模型可比性 > 可复现性 > 代码可读性 > 推理性能**
>
> 无效输出不会被排除，而是按错误计入准确率分母。

Xiaomi 的评测目录 `Xiaomi-Lens-0/eval_xiaomi_cvbench/` 在整个过程中**只读**，没有被
修改、删除或写入任何文件，也没有干扰它的运行进程。

---

## 0. 为什么单独建一个目录

Xiaomi 的评测代码设计时就把"模型相关"和"模型无关"分开了。因此这里**没有重写评测框架**：

| 文件 | 来源 | 作用 |
| --- | --- | --- |
| `cvbench_dataset.py` | 逐字节复制 | 读取 parquet、切分 type/task/source、懒加载图片 |
| `answer_parser.py` | 逐字节复制 | 严格答案解析、Thinking 切分 |
| `metrics.py` | 逐字节复制 | 官方加权 Accuracy、耗时/Token/显存统计、同题对比 |
| `test_answer_parser.py` | 逐字节复制 | 17 个解析器单元测试 |
| `model_qwen35.py` | **新写** | Qwen3.5-4B 的加载、Prompt、生成 |
| `run_qwen35_cvbench.py` | 基于 runner 改写 | 命令行、断点续跑、结果保存、校验 |
| `make_report.py` | **新写** | 中文报告 + 与 Xiaomi 的横向对比 |
| `test_metrics.py` | **新写** | 数据集、官方分数、Invalid 计分、续跑工具链测试 |

复制而不是 `import`，有两个原因：

1. 直接 import 会在 Xiaomi 目录里生成 `__pycache__`，属于向只读目录写入；
2. 复制能保证两边的评分逻辑是同一份代码，不会因为对方后续改动而悄悄分叉。

**验证方式**：`sha1sum` 对比，四个通用文件的哈希与 Xiaomi 原文件完全一致（见第 9 节）。

---

## 1. 数据集

本地路径：`/home/20261024_1/Workspace/qyh/VLM-test/dataset/CV-Bench`
（图片字节直接存在 `test_2d.parquet` / `test_3d.parquet` 的 `image` 列里）

| 维度 | task | source | 数量 |
| --- | --- | --- | ---: |
| 2D | Count | ADE20K | 342 |
| 2D | Count | COCO | 446 |
| 2D | Relation | ADE20K | 291 |
| 2D | Relation | COCO | 359 |
| 3D | Depth | Omni3D | 600 |
| 3D | Distance | Omni3D | 600 |
| **合计** | | | **2638** |

- 2D 合计 1438，3D 合计 1200，与官方 2638 条一致。
- 选项数量不固定：3D 全部是 2 选项；2D 有 3/4/5/6 选项。程序按 `len(choices)` 动态
  生成合法字母集合（A…F），不会假设只有 A–D。
- 样本 uid 为 `type:idx`（例如 `2D:1234`），两个模型用同一套 uid，可以直接对齐。

---

## 2. 图片和问题是怎么送进 Qwen 的

```text
PIL Image ─┐
           ├─► apply_chat_template ─► 文本模板 ─┐
问题文本 ──┘                                     ├─► Processor ─► input_ids + pixel_values ─► GPU
                                                 │
```

1. **Chat Template**（`chat_template.jinja`）把"消息列表"变成模型认识的纯文本。图片占位
   会展开成 `<|vision_start|><|image_pad|><|vision_end|>`。
2. **Processor** 做两件事：分词器把文本变成 token id；图像处理器把 PIL 图片切成
   16×16 的 patch 并归一化，得到 `pixel_values`。`<|image_pad|>` 在 `input_ids` 里会被
   视觉 token 替换。
3. 生成的 `input_ids` / `pixel_values` 一起 `.to(device)` 送进 GPU。

图片**不写进** JSONL，只保存问题、选项和时间统计。

---

## 3. Thinking 是怎么开关的

控制方式与 Xiaomi 完全相同——用 checkpoint 自带 chat template 的布尔参数：

| 模式 | 调用 | 渲染后的 prompt 结尾 |
| --- | --- | --- |
| Thinking Off | `enable_thinking=False` | `<think>\n\n</think>\n\n` |
| Thinking On | `enable_thinking=True` | `<think>\n` |

渲染完成后程序会**检查 prompt 后缀**是否真的是该模式的结果（`mode_verified`）。如果模板
将来变了、开关失效，会直接抛错停止，而不是静默跑错实验。

**Adaptive Thinking：不支持。** 模板只有布尔开关，没有 adaptive/auto 参数，也没有
Thinking Budget。`--mode adaptive` 会打印说明并退出（退出码 2），不会偷偷改成 On 或 Off。

---

## 4. generate() 的输出是怎么解码的

```python
output_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=False, use_cache=True)
completion_ids = output_ids[:, prompt_length:]          # 只取新生成的 token
raw_text = tokenizer.decode(completion_ids[0], skip_special_tokens=False)
```

- `prompt_length` 是输入长度，减掉它才能拿到"模型自己写的那部分"。
- `skip_special_tokens=False`：保留 `<|im_end|>` 等特殊 token，方便判断结束原因，
  Parser 会在匹配前把它们剥掉。
- `do_sample=False`：贪心解码，同样的输入永远得到同样的输出，保证可复现。
- `finish_reason`：最后一个 token 是 eos → `eos`；生成数达到上限 → `length`（截断）。

---

## 5. 为什么要把 Thinking 和最终回答分开

Thinking On 时，模型会先自己推理，推理过程里**会提到多个选项**：

```text
<think>
Option A is unlikely.
Option B may be correct.
Option C matches the image.
</think>

FINAL_ANSWER: (C)
```

如果直接在全文里搜"A/B/C/D"，很可能抓到 `Option A` 里的 A，得出错误结论。
所以程序按 `</think>`（token id `248069`，单个 token）切开，**只在之后的最终回答区
解析答案**。

Thinking Token 数 = `</think>` 之前的 token 数；最终回答 Token 数 = 之后的 token 数。
Thinking Off 时模板已经插入了空的思考区，completion 里没有 `</think>`，Thinking Token 记为 0。

---

## 6. 严格答案匹配的工作方式

Prompt 末尾追加（与 Xiaomi **逐字符相同**）：

```text
After answering the question, put the final choice on a separate line
using exactly this format: FINAL_ANSWER: (X)
where X is one of the provided option letters.
```

解析规则（`answer_parser.py`）：

1. 先按 `</think>` 切出最终回答区（Thinking On 必须有 `</think>`，否则无效）。
2. 在回答区里**逐行**匹配整行 `FINAL_ANSWER: (X)`，允许行首行尾空白。
3. 重复的相同答案可以接受；出现两个**不同**选项 → `conflicting_final_answers`。
4. 字母必须在本题真实存在的选项集合内，否则 `invalid_option`。
5. 解析不出来 → `invalid`，**仍然计入分母**（算答错）。

| invalid_reason | 含义 |
| --- | --- |
| `missing_think_close` | Thinking On 但没有 `</think>` |
| `truncated_before_final_answer` | 同上，且达到 `max_new_tokens` |
| `truncated_no_final_answer` | 回答区没有答案，且被截断 |
| `malformed_final_answer` | 出现 `FINAL_ANSWER` 但格式不对（小写、加粗等） |
| `empty_final_answer` | `</think>` 之后为空 |
| `no_final_answer` | 有文字但没有答案标记 |
| `conflicting_final_answers` | 多个互相冲突的选项 |
| `invalid_option` | 字母不属于本题 |
| `inference_error` | 推理异常 |

**不做**：全文搜索第一个/最后一个字母、模糊匹配选项文字、从 Thinking 中间取答案、
根据标准答案反推预测。

---

## 7. 官方分数怎么算

CV-Bench 官方不是"全样本简单平均"，而是按数据来源分别算再平均：

```text
Accuracy_2D  = (Accuracy_ADE20K + Accuracy_COCO) / 2
Accuracy_3D  = Accuracy_Omni3D
CV-Bench     = (Accuracy_2D + Accuracy_3D) / 2
```

报告里同时给出"全部样本普通 Accuracy"，两者可能不同（因为 ADE20K 633 条、COCO 805 条、
Omni3D 1200 条，样本数不等）。**官方总分才是可对外比较的成绩。**

如果只跑了子集（例如 `--limit 20`），缺少某个来源时对应官方分数记为 `N/A`，不会用
小样本分数冒充完整成绩。`metrics.json` 里 `subset: true` 表示这是子集结果。

### 各细分任务怎么统计

`metrics.py` 按 `type` / `task` / `source` / `type_task` / `type_task_source` 五个维度
分组，每组记录：

```text
正确数 / 总样本数、Accuracy (%)、Invalid 数、Invalid Rate
```

例如 `groups.by_task.Count = {n, correct, accuracy, invalid}`。报告里的
"物体计数 123/788 (15.6%)" 就是从这里来的。

---

## 8. 耗时和显存是怎么量的

**计时**（全部在 `torch.cuda.synchronize()` 之后读取，避免 CUDA 异步导致低估）：

| 字段 | 含义 |
| --- | --- |
| `preprocess_seconds` | Processor 处理图片和文本 + 搬到 GPU |
| `image_load_seconds` | 从 parquet 解出图片字节 |
| `generate_seconds` | `model.generate()` 的耗时 |
| `e2e_seconds` | 单条样本端到端耗时 |
| `model_load_seconds` | 加载权重到显存 |
| `warmup_seconds` | 预热（不计入成绩） |

`Tokens/s = 生成 token 数 / generate 耗时`，**包含 Prefill**，不是纯 Decode 速度。

**显存**（PyTorch CUDA Memory API，单位 GiB）：

| 字段 | 含义 |
| --- | --- |
| `allocated_gib` | 当前活跃 Tensor 占用 |
| `reserved_gib` | PyTorch 缓存池保留量 |
| `peak_allocated_gib` | 统计窗口内的历史峰值 |
| `gpu_used_gib` | **整张卡**的使用量（包含别人的进程） |

每个样本生成前都会 `reset_peak_memory_stats()`，所以单样本峰值 = 模型权重 + 该样本激活，
不是"增量显存"。`gpu_used_gib` 不能和 PyTorch 峰值混为一谈。

---

## 9. 与 Xiaomi 结果怎么对比、哪些能公平比较

运行 `make_report.py` 会自动读取 Xiaomi 的 `results/`（**只读**）并生成对比表。

### 可以公平比较

- **准确率**：解析器、指标代码、Prompt 格式指令、生成参数（贪心、同样的
  `max_new_tokens`）全部一致。协议一致性会在报告里逐项核对并打勾。

### 不能直接比较

- **耗时 / 吞吐 / 显存**：两个模型参数量不同、Attention 实现不同，而且这台机器上
  同时跑着其他用户的 DRL 任务，显存和算力都在被争用。
  `experiment.json` 里的 `gpu_shared` 会记录当时是否共享 GPU。
- 想严格比较推理效率，需要在 GPU 独占时用**同一份样本**单独复测，不必重跑全部准确率。

### Xiaomi 还没跑完时

报告会显式标注 `status: partial` 和已完成样本数，并把 Xiaomi 的数字标记为"部分结果"，
**不会**把它当成正式成绩，也不会据此断言谁更强。

### 文件指纹（证明共用同一套协议）

```text
63f3f0296112209a3318396df68d2bffb247d285  answer_parser.py
91f10b3c7c51387897292026f9802d1668812ca4  cvbench_dataset.py
b2e0a27d3b41dcdf403a97bd63ac7579d4fcc9e0  metrics.py
8a1d4e82ac5753ac8a31223fa5a6d3370b6b4d3a  test_answer_parser.py
```

这 4 个文件与 `Xiaomi-Lens-0/eval_xiaomi_cvbench/` 中的同名文件**哈希完全相同**。

---

## 10. 代码结构

```text
eval_qwen35_cvbench/
├── cvbench_dataset.py      # [复制] parquet 读取、样本标准化、图片懒加载
├── answer_parser.py        # [复制] 严格答案解析
├── metrics.py              # [复制] Accuracy / 耗时 / Token / 显存 / 同题对比
├── test_answer_parser.py   # [复制] 17 个解析器测试
├── model_qwen35.py         # [新写] Qwen3.5-4B 适配器
├── run_qwen35_cvbench.py   # [改写] 命令行、续跑、保存、校验
├── make_report.py          # [新写] 中文报告 + 横向对比
├── test_metrics.py         # [新写] 19 个指标/数据集/续跑测试
├── README.md
└── results/
    ├── thinking_off/{predictions.jsonl, metrics.json, summary.csv, experiment.json}
    ├── thinking_on/{...}
    └── comparison_report.md
```

---

## 11. 常用命令

环境：`micromamba run -n vlm-test python ...`（Python 3.11 / PyTorch 2.10 / Transformers 5.20）

### 单元测试（不需要 GPU）

```bash
micromamba run -n vlm-test python eval_qwen35_cvbench/test_answer_parser.py
micromamba run -n vlm-test python eval_qwen35_cvbench/test_metrics.py
```

### 单样本测试

```bash
micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py \
    --mode off --limit 1 --verbose --run-name single
```

### 分层小样本（覆盖全部 6 个 type/task/source）

```bash
micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py \
    --mode all --limit 16 --stratified --run-name smoke16
```

### 完整测试集

```bash
# 两种模式依次跑完，模型只加载一次
micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py \
    --mode all --resume

# 或者分开跑
micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py --mode off --resume
micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py --mode on  --resume
```

### 断点续跑

用**完全相同的参数**再加 `--resume`。程序按 uid 跳过已完成样本；`experiment.json`
的 fingerprint 不一致会拒绝续跑，防止不同配置混在一起；崩溃留下的半行 JSON 会自动修掉。
CUDA OOM 的样本不会写成"已完成"，续跑时会重新评测。

### 校验结果

```bash
micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py \
    --mode all --validate
```

检查：样本覆盖、重复 UID、metrics.json 与逐样本重算是否一致。

### 生成报告

```bash
micromamba run -n vlm-test python eval_qwen35_cvbench/make_report.py
cat eval_qwen35_cvbench/results/comparison_report.md
```

### 只看某个小样本目录的报告

```bash
micromamba run -n vlm-test python eval_qwen35_cvbench/make_report.py --run-name smoke16
```

---

## 12. 共享 GPU 注意事项

本机 GPU 同时跑着其他用户的 DRL 任务，空闲显存会在 20–60 GB 之间波动。程序的处理：

1. **加载前**检查空闲显存，低于 `--require-free-gib`（默认 12 GiB）就退出，不去 OOM。
2. **每个样本前**检查 `torch.cuda.mem_get_info`，低于 `--min-free-gib` 则等待。
3. **OOM** 时 `empty_cache()` 并退避重试；仍失败则安全停止，不写该样本。
4. 全程**不终止、不影响**其他进程。

> 注意：`torch.cuda.mem_get_info()` 报告的空闲显存可能明显小于 `nvidia-smi` 的
> `memory.free`（因为 CUDA 上下文预留）。阈值不要设太高，否则会每个样本前空等。

## 13. 输出字段

`predictions.jsonl` 每条：

- `uid` / `idx` / `type` / `task` / `source` / `source_dataset`
- `question` / `choices` / `ground_truth`
- `model_name`（Qwen 特有，Xiaomi 记录里没有）
- `thinking_mode` / `enable_thinking` / `mode_verified`
- `raw_generation`（只含新生成 token，保留特殊 token）
- `thinking_text` / `final_answer_text`
- `prediction` / `correct` / `invalid` / `invalid_reason`
- `truncated` / `finish_reason`
- `prompt_tokens` / `generated_tokens` / `thinking_tokens` / `answer_tokens` / `mask_tokens`
- `image_load_seconds` / `preprocess_seconds` / `generate_seconds` / `e2e_seconds`
- `vram`（allocated / reserved / peak / 整卡使用量）
- `error` / `retryable`

字段名与 Xiaomi 一致；`mask_tokens` 保留只是为了对齐表结构（Qwen 无分割掩码输出，恒为 0）。

`metrics.json` 记录模型与数据集版本、Prompt 模板、生成参数、全部 Accuracy、
耗时/Token/显存统计、invalid/截断原因、完成状态，以及 `gpu_shared`。
