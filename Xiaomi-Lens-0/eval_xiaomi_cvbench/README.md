# Xiaomi-Lens-0 × CV-Bench 评测

本目录是一套**可复用、可断点续跑、评分严格**的 CV-Bench 评测程序，用来比较
Xiaomi-Lens-0 在 **Thinking Off / Thinking On** 两种模式下的空间理解能力、耗时、
Token 与显存开销。

> 优先级：**评测正确性 > 实验可复现性 > 代码可读性 > 推理性能**。
> 无效输出不会被排除，而是按错误计入准确率分母。

---

## 1. 数据集

本地路径：`/home/20261024_1/Workspace/qyh/VLM-test/dataset/CV-Bench`

图片字节直接存放在 `test_2d.parquet` / `test_3d.parquet` 的 `image` 列中，`img/`
目录只有 9 张预览图，因此程序从 parquet 按 row group 懒加载图片，不预先导出。

| 维度 | task | source | 数量 |
| --- | --- | --- | ---: |
| 2D | Count | ADE20K | 342 |
| 2D | Count | COCO | 446 |
| 2D | Relation | ADE20K | 291 |
| 2D | Relation | COCO | 359 |
| 3D | Depth | Omni3D | 600 |
| 3D | Distance | Omni3D | 600 |
| **合计** | | | **2638** |

- 2D 合计 1438，3D 合计 1200，与官方 Dataset Card 的 2638 条一致。
- 选项数量不固定：2 选项目最多（3D 全部为 2 选项），2D 还有 4/5/6 选项。

---

## 2. Thinking 模式调查

结论以 checkpoint 的 `chat_template.jinja` 与 `generation_config.json` 为准。

| 模式 | 是否原生支持 | 控制方式 | 验证方式 |
| --- | --- | --- | --- |
| Thinking Off | 是 | `apply_chat_template(..., enable_thinking=False)`，模板注入空的 `<think>\n\n</think>\n\n` | 渲染后的 prompt 必须以该后缀结尾；每个样本记录 `mode_verified` |
| Thinking On | 是 | `apply_chat_template(..., enable_thinking=True)`，模板以 `<think>\n` 结尾 | 同上 |
| Adaptive / Auto | **否** | 模板只有布尔参数，没有 adaptive 开关或 Thinking Budget | 不实现第三组；程序对 `--mode adaptive` 明确提示并退出码 2 |

**没有把固定 Token 预算或外部规则称为“原生自适应思考”。** 如果将来要做外部路由，
它是另一套实验（例如按任务类型决定是否开启 Thinking），需要单独命名与报告。

模型输出的 Thinking 区与最终回答用 `</think>`（token id `248069`）精确切分：
Thinking Token 数 = `</think>` 之前的 token 数，最终回答 Token 数 = 之后的 token 数。
Thinking Off 时模板已经把思考区关闭，因此 completion 中没有 `</think>`，Thinking
Token 记为 0。

模型在回答中可能自发输出 `<bbox>...</bbox><mask><|mask|></mask>`。CV-Bench 是文本
选择题，程序**不执行** Mask 解码/可视化，只统计 `<|mask|>` 数量；这些 token 不会
被当作选项。

---

## 3. 评测流程

```text
读取 CV-Bench parquet（图像 + 问题 + 选项）
    ↓
按 idx 排序，可选 --split / --limit / --stratified
    ↓
构建统一 Prompt：数据集原始 prompt + FINAL_ANSWER 格式指令
    ↓
apply_chat_template(enable_thinking=...) → 验证模板后缀
    ↓
Processor 处理图像与文本 → 移到 GPU
    ↓
model.generate(max_new_tokens=8192, do_sample=False)
    ↓
只解码新生成的 token（skip_special_tokens=False）
    ↓
遇到 </think> 切分 Thinking / 最终回答
    ↓
只在最终回答区严格匹配 FINAL_ANSWER: (X)
    ↓
与 Ground Truth 比较（无效 = 错误，仍进分母）
    ↓
逐条写 predictions.jsonl（可断点续跑）
    ↓
汇总 Accuracy / 耗时 / Token / 显存
```

### 3.1 Prompt

保留数据集原始 `prompt`（问题与选项顺序不变），在末尾追加：

```text
After answering the question, put the final choice on a separate line
using exactly this format: FINAL_ANSWER: (X)
where X is one of the provided option letters.
```

三种可运行模式使用完全相同的 Prompt 和格式指令，只有 `enable_thinking` 不同。
标准答案从不传给模型。

### 3.2 严格答案解析

规则（`answer_parser.py`）：

1. 只在 **`</think>` 之后**的最终回答区查找答案。
2. 只接受**整行**严格匹配 `FINAL_ANSWER: (X)`（允许行首/行尾空白、冒号后空格）。
3. `X` 必须是当前题目真实存在的选项字母。
4. 多个相同答案可接受；多个不同答案判为 `conflicting_final_answers`。
5. 以下情况记录为 invalid，且**不会被排除出分母**：

| invalid_reason | 含义 |
| --- | --- |
| `missing_think_close` | Thinking On 但没有 `</think>` |
| `truncated_before_final_answer` | 同上，且达到 `max_new_tokens` |
| `truncated_no_final_answer` | 最终回答区没有答案，且被截断 |
| `malformed_final_answer` | 出现 `FINAL_ANSWER` 但格式不符合 |
| `empty_final_answer` | `</think>` 后为空 |
| `no_final_answer` | 有回答文字但没有答案标记 |
| `conflicting_final_answers` | 多个互相冲突的选项 |
| `invalid_option` | 选项字母不属于本题 |
| `inference_error` | 推理异常（记录异常信息） |

**禁止**：全文本搜索第一个/最后一个 A/B/C/D、模糊匹配选项文字、从 Thinking 中间
过程取答案、根据 Ground Truth 猜测。

正确例子：

```text
Ground Truth: (C)

模型输出：
The image contains three objects...
</think>
FINAL_ANSWER: (C)

→ 解析 C，Correct
```

错误例子（不能解析成 C）：

```text
Thinking:
Option A is unlikely.
Option B may be correct.
Option C is more plausible.

最终回答：
FINAL_ANSWER: (B)

→ 最终预测是 B，因为只解析 </think> 之后的答案行
```

### 3.3 官方 Accuracy

按 Dataset Card：

```text
Accuracy_2D = (Accuracy_ADE20K + Accuracy_COCO) / 2
Accuracy_3D = Accuracy_Omni3D
CV-Bench Accuracy = (Accuracy_2D + Accuracy_3D) / 2
```

报告同时给出“全部样本普通平均 Accuracy”。两者可能不同，官方总分才是可对外比较的
成绩。子集运行会在 `metrics.json` 中标记 `subset: true`；任一官方分组缺失时官方
分数为 `N/A`，不会用小样本分数冒充完整测试集成绩。

---

## 4. Thinking 模式的公平对比

- 同一 checkpoint、同一数据集、同一批 `idx`、同一图片预处理、同一 Prompt 与格式指令。
- 同一严格解析函数、同一 Accuracy 公式。
- bfloat16，Batch Size = 1，`do_sample=False`（贪心，确定性）。
- 两种模式**共用同一个模型实例**，只切换经过验证的 `enable_thinking`。
- 两种模式统一使用 `max_new_tokens=8192`。CV-Bench 问题较短，实测 20 条样本的
  最长输出约 1000 token；如果正式运行出现截断，会单独报告截断率，不做“补答”。
- 若需要修改生成长度，必须固定新配置并重跑受影响的整组实验；不同配置的结果放在
  不同 `--run-name` 目录，程序会因 fingerprint 不一致而拒绝混用。

---

## 5. 性能指标说明

- **Prefill vs Decode**：Prefill 一次性处理图片和 prompt，Decode 逐个生成 token。
  本报告的 `Tokens/s` = 生成 token 数 / 整个 `generate()` 耗时，**包含 Prefill**，
  不是纯 Decode 速度。
- **CUDA 计时**：所有耗时在 `torch.cuda.synchronize()` 之后读取，避免异步执行低估计时。
- **当前显存 vs 峰值显存**：
  - `allocated`：当前 PyTorch Tensor 实际占用；
  - `reserved`：PyTorch 缓存池占用；
  - `peak_allocated`：自上次 `reset_peak_memory_stats()` 以来的峰值；
  - `gpu_used`：整张 GPU 的使用量（包含其他进程）。
  本程序在每个样本生成前重置峰值统计，因此单样本峰值 = 模型权重 + 该样本的激活；
  它不是“增量显存”。其他 GPU 进程的使用量会单独记录，不能与 PyTorch 峰值混为一谈。
- **Thinking 成本**：Thinking On 会生成额外的推理 token，直接增加 Decode 时间和
  显存中的 KV Cache 占用；但如果回答本身变短，总时长不一定按 Thinking Token 比例增长。
- **Adaptive vs Thinking Budget**：Adaptive 是模型自己决定“是否思考”；Thinking
  Budget 是外部限制思考 token 数；外部路由器是程序规则。三者不是一回事。本模型
  只原生支持布尔开关。

---

## 6. 代码结构

```text
eval_xiaomi_cvbench/
├── cvbench_dataset.py     # parquet 读取、样本标准化、图片懒加载、分布统计
├── xiaomi_lens_adapter.py # 模型专用：加载、chat template、generate、计时/显存
├── answer_parser.py       # 严格答案解析与 Thinking 切分（与模型无关）
├── metrics.py             # 官方/分组 Accuracy、耗时、Token、显存、同题对比
├── run_evaluation.py      # CLI、断点续跑、结果保存、validate
├── make_report.py         # 生成中文 comparison_report.md
├── test_answer_parser.py  # 17 个解析器单元测试
├── README.md
└── results/
    ├── thinking_off/
    │   ├── predictions.jsonl
    │   ├── metrics.json
    │   ├── summary.csv
    │   └── experiment.json
    ├── thinking_on/
    │   └── ...
    └── comparison_report.md
```

未来评测 Qwen3.5 等其他模型时，主要替换 `xiaomi_lens_adapter.py`：实现同样的
`load()` / `build_inputs()` / `generate()` 返回结构即可，数据、解析、指标与报告
模块无需重写。

---

## 7. 使用方法

环境：`micromamba run -n xiaomi-lens python ...`（Python 3.11 / PyTorch 2.10 /
Transformers 5.4.0）。为读取本地 parquet 安装了 `pyarrow`，未升级或安装其他大型框架。

### 7.1 单元测试（不需要 GPU）

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/test_answer_parser.py
```

### 7.2 单样本

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
    --mode off --limit 1 --verbose --run-name single
```

### 7.3 小规模分层测试（覆盖全部 6 个 type/task/source）

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
    --mode all --limit 20 --stratified --run-name smoke20
```

### 7.4 Thinking Off 全量

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
    --mode off --resume
```

### 7.5 Thinking On 全量

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
    --mode on --resume
```

### 7.6 两种模式依次全量（模型只加载一次）

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
    --mode all --resume
```

### 7.7 Adaptive（明确不支持）

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py --mode adaptive
# 打印不支持说明，退出码 2；不会静默改成 Off/On
```

### 7.8 断点续跑

使用相同参数并加 `--resume`。程序按稳定 `uid`（`type:idx`）跳过已完成样本；
`experiment.json` 的 fingerprint 不一致时会拒绝续跑，防止不同模型/模式/生成参数
的结果混在一起。中断产生的半行 JSON 会在下次启动时自动修掉。

### 7.9 结果校验

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
    --mode all --validate
```

检查：样本覆盖是否与数据集一致、是否有重复 UID、metrics.json 与逐样本重算是否一致。

### 7.10 生成对比报告

```bash
micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/make_report.py
cat eval_xiaomi_cvbench/results/comparison_report.md
```

---

## 8. 运行注意事项（共享 GPU）

本机 GPU 同时运行其他用户的 DRL/CityNav 进程，显存占用会在 35–95 GB 之间波动。
这会带来两个影响：

1. **绝对耗时不可当作独占 GPU 的基准**；Off/On 在同一环境下顺序运行，相对比较
   仍有参考价值，但报告里会明确标注。
2. **可能 CUDA OOM**。程序的处理方式：
   - 每个样本前检查空闲显存（默认低于 6 GiB 时最多等待 10 分钟）；
   - OOM 时 `empty_cache()` 并退避重试（默认 4 次）；
   - 仍然 OOM 时**安全停止**，不把该样本写成“已完成”；
   - 下次 `--resume` 会丢弃所有 OOM 记录并重新评测这些样本。

建议在 GPU 至少有 12 GiB 空闲时启动全量运行。显存不足时不要终止其他用户的进程。

> 注意：本机 `torch.cuda.mem_get_info()` 报告的空闲显存可能远小于 `nvidia-smi`
> 的 `memory.free`（实测 4.6 GiB vs 30 GiB）。程序用 `mem_get_info` 判断是否需要
> 等待，默认阈值为 2 GiB；如果阈值设得过高，会在每个样本前空等。

## 9. 结果文件

`predictions.jsonl` 每条样本包含：

- `uid` / `idx` / `type` / `task` / `source`
- `question` / `choices` / `ground_truth`
- `thinking_mode` / `enable_thinking` / `mode_verified`
- `raw_generation`（只含新生成 token，保留特殊 Token）
- `thinking_text` / `final_answer_text`
- `prediction` / `correct` / `invalid` / `invalid_reason`
- `truncated` / `finish_reason`
- `prompt_tokens` / `generated_tokens` / `thinking_tokens` / `answer_tokens` / `mask_tokens`
- `image_load_seconds` / `preprocess_seconds` / `generate_seconds` / `e2e_seconds`
- `vram`（allocated / reserved / peak / 整卡使用量）
- `error`（推理异常信息）

不写入图片二进制。

`metrics.json` 保存模型与数据集版本、Prompt 模板、生成参数、全部 Accuracy、
耗时/Token/显存统计、invalid/截断原因、完成状态。

---

## 10. 当前实验状态

完整测试集的运行命令、进度与最终结果见
`results/comparison_report.md`。报告中的每个数字都来自实际运行；若运行尚未完成，
报告中会明确标记 `status: partial` 与已完成样本数，不会把部分结果说成完整成绩。
