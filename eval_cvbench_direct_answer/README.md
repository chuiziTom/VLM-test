# CV-Bench Direct Answer 统一评测（v1）

本目录实现一组**独立于历史评测**的实验：

- 模型：Xiaomi-Lens-0、Qwen3.5-4B
- 模式：**Thinking Off + Direct Answer**（只输出一行答案，不输出分析）
- `max_new_tokens = 64`
- 数据集：CV-Bench 测试集（2638 条，与历史实验相同）

历史 8192 实验（`legacy_max_new_tokens_8192`）的代码和结果**未被修改**，
两套实验的规范名称与实际路径见项目根目录的 `EXPERIMENT_INDEX.md`。

---

## 1. 目录结构

```text
eval_cvbench_direct_answer/
├── prompts/
│   └── direct_answer_v1.txt          # 统一 Direct Answer 指令
├── common/
│   ├── direct_parser.py              # 整段严格解析
│   ├── module_loader.py              # 按路径只读导入原模块
│   └── runner_core.py                # 评测主循环、断点续跑、指标保存
├── run_xiaomi_direct_answer.py       # Xiaomi 入口（xiaomi-lens 环境）
├── run_qwen35_direct_answer.py       # Qwen 入口（vlm-test 环境）
├── tests/
│   └── test_direct_parser.py         # 17 个解析器单元测试
├── results/
│   ├── direct_answer_think_off_max_new_tokens_64_v1/
│   │   ├── xiaomi_lens/{predictions.jsonl, metrics.json, summary.csv, experiment.json}
│   │   └── qwen35_4b/{...}
│   └── smoke_direct_answer_v1/       # 16 条冒烟，不算正式成绩
├── reports/
│   ├── xiaomi_lens_direct_answer_report.md
│   ├── qwen35_4b_direct_answer_report.md
│   └── cvbench_direct_answer_comparison.md
└── logs/
```

---

## 2. 复用了什么、没有改什么

| 组件 | 来源 | 使用方式 |
|---|---|---|
| CV-Bench 数据加载、图片解码、uid | `eval_qwen35_cvbench/cvbench_dataset.py` / `Xiaomi-Lens-0/eval_xiaomi_cvbench/cvbench_dataset.py` | 按文件路径导入（两份 SHA-1 相同） |
| 官方加权 Accuracy、耗时/Token/显存统计 | 各自的 `metrics.py` | 按文件路径导入 |
| 模型加载、chat template、生成 | `model_qwen35.py` / `xiaomi_lens_adapter.py` | 直接使用原适配器 |
| 答案解析 | **新写** `common/direct_parser.py` | 本次实验要求整段严格匹配 |

**未修改**任何原有 Python 文件、预测文件、指标文件或数据集。

---

## 3. Direct Answer Prompt（v1）

文件：`prompts/direct_answer_v1.txt`，完整内容：

```text
Answer the multiple-choice question directly.

Do not provide explanations, reasoning, or analysis.
Output exactly one line in the following format:

FINAL_ANSWER: (X)

where X is one of the provided option letters.

Do not output anything else.
```

实际送入模型的 instruction 为：

```python
instruction = record.prompt + "\n\n" + DIRECT_ANSWER_V1
```

即保留 CV-Bench 原始题目与选项，只在末尾追加统一指令。两个模型逐字符相同。

---

## 4. Thinking Off 如何生效

不是只在 Prompt 里写“不要推理”，而是使用 checkpoint 自带 chat template 的
官方布尔参数：

```python
apply_chat_template(..., enable_thinking=False)
```

- Qwen3.5-4B：`model_qwen35.py` 中的 `Qwen35Adapter.build_inputs(..., mode="off")`
- Xiaomi-Lens-0：`xiaomi_lens_adapter.py` 中的 `XiaomiLensAdapter.build_inputs(..., mode="off")`

渲染后会校验 prompt 后缀必须为 `<think>\n\n</think>\n\n`（`mode_verified`），
否则直接报错停止，不会把未验证的配置标记为 Thinking Off。

---

## 5. 严格答案解析

`common/direct_parser.py` 的规则：

1. 只解码模型新生成的 token；
2. 去掉 `<|im_end|>`、`<|endoftext|>`、`<|mask|>` 等结构性特殊 token；
3. 去掉首尾空白后，整段必须**恰好**匹配 `FINAL_ANSWER: (X)`；
4. 字母必须属于本题真实选项集合（A–F 动态确定）；
5. 其他情况全部记为 `invalid`，且**保留在 Accuracy 分母中**。

无效原因：

| `invalid_reason` | 含义 |
|---|---|
| `extra_text` | 除答案行外还有分析文字 |
| `multiple_final_answers` | 出现多行答案 |
| `malformed_final_answer` | 出现 `FINAL_ANSWER` 但格式不对 |
| `invalid_option` | 字母不在本题选项中 |
| `empty_output` | 没有输出 |
| `truncated_*` | 达到 64 token 上限且没有合法答案 |
| `inference_error` | 推理异常 |

`format_violation` 单独标记格式类问题，便于统计格式违规率。

---

## 6. 运行命令

冒烟（16 条分层样本，输出到独立目录）：

```bash
# Xiaomi
micromamba run -n xiaomi-lens python -B \
    eval_cvbench_direct_answer/run_xiaomi_direct_answer.py \
    --limit 16 --stratified \
    --output-dir eval_cvbench_direct_answer/results/smoke_direct_answer_v1 --show 16

# Qwen
micromamba run -n vlm-test python -B \
    eval_cvbench_direct_answer/run_qwen35_direct_answer.py \
    --limit 16 --stratified \
    --output-dir eval_cvbench_direct_answer/results/smoke_direct_answer_v1 --show 16
```

全量（顺序运行，模型各自只加载一次）：

```bash
micromamba run -n xiaomi-lens python -B \
    eval_cvbench_direct_answer/run_xiaomi_direct_answer.py --resume

micromamba run -n vlm-test python -B \
    eval_cvbench_direct_answer/run_qwen35_direct_answer.py --resume
```

校验：

```bash
micromamba run -n xiaomi-lens python -B \
    eval_cvbench_direct_answer/run_xiaomi_direct_answer.py --validate
micromamba run -n vlm-test python -B \
    eval_cvbench_direct_answer/run_qwen35_direct_answer.py --validate
```

单元测试：

```bash
micromamba run -n vlm-test python -B \
    eval_cvbench_direct_answer/tests/test_direct_parser.py
```

---

## 7. 输出字段

`predictions.jsonl` 在历史字段基础上增加：

- `protocol` = `"direct_answer"`
- `prompt_version` = `"v1"`
- `normalized_text`：清洗后的完整输出
- `format_violation`：是否格式违规
- 其余 `uid / type / task / source / question / choices / ground_truth /
  prediction / correct / invalid / invalid_reason / truncated /
  generated_tokens / *_seconds / vram` 与历史一致，便于复用 `metrics.py`。

`metrics.json` 额外包含 `direct_answer` 段：Prompt 路径与 SHA-1、解析器版本、
格式违规数与违规率、无效原因分布。

---

## 8. 与历史实验的关系

历史实验（8192）与本次实验（64）在以下方面不同：

- Prompt 追加内容不同（历史允许分析，本次禁止）；
- `max_new_tokens` 不同；
- 解析规则不同（历史逐行扫描最终回答区，本次整段严格匹配）。

因此**不能把两次实验当作严格对照组**，跨实验比较只能作为参考，
报告与 `EXPERIMENT_INDEX.md` 均需明确标注。
