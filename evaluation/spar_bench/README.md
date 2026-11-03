# SPAR-Bench-Tiny 评测（RGB）

本目录用 Xiaomi-Lens-0 与 Qwen3.5-4B 评测 **SPAR-Bench-Tiny RGB**（1,000 条，
20 种任务 × 50 条）。旧 CV-Bench 代码保持不变；这里的所有文件都是新增的。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `inspect_dataset.py` | 只读本地 Parquet，生成数据集统计与样本拼图 |
| `sparbench_dataset.py` | 数据读取、图片解码、按任务分层取样 |
| `sparbench_prompts.py` | 复现官方 Prompt + Direct Answer Prompt |
| `sparbench_parser.py` | Thinking 切分 + Strict 答案解析 |
| `sparbench_metrics.py` | 官方评分函数（逐行复现）与汇总 |
| `model_base.py` | 多图输入、生成、计时、显存统计的公共基类 |
| `xiaomi_lens_adapter.py` | Xiaomi-Lens-0 模型加载 |
| `qwen35_adapter.py` | Qwen3.5-4B 模型加载 |
| `run_evaluation.py` | 单模型单协议评测入口（支持断点续跑） |
| `recompute_metrics.py` | 从 `predictions.jsonl` 重算指标 |
| `make_reports.py` | 生成单模型报告与综合对比报告 |
| `reference/` | 官方 `utils.py` / yaml 原件，供对拍测试 |
| `tests/` | 与官方实现的逐函数对拍测试 |

## 2. 环境

```bash
# Xiaomi-Lens-0
micromamba run -n xiaomi-lens python evaluation/spar_bench/run_evaluation.py ...

# Qwen3.5-4B
micromamba run -n vlm-test python evaluation/spar_bench/run_evaluation.py ...
```

两个环境都使用现有依赖，不重新创建环境、不升级 PyTorch / Transformers。

## 3. 三种协议

| 协议 | enable_thinking | max_new_tokens | Prompt |
|---|---|---:|---|
| `think_off_8192` | False | 8192 | `official_v1` |
| `think_on_8192` | True | 8192 | `official_v1` |
| `direct_off_64` | False | 64 | `direct_answer_v1` |

`official_v1` 逐字复现官方 `sparbench_doc_to_text`；Thinking On/Off 的 Prompt
完全相同，唯一变量是 chat template 的 `enable_thinking`。`direct_answer_v1`
是独立 Prompt 版本，不能当作 Thinking Off 的单变量消融。

## 4. 快速开始

```bash
# 数据集统计
micromamba run -n vlm-test python evaluation/spar_bench/inspect_dataset.py

# 20 任务冒烟测试
micromamba run -n vlm-test python evaluation/spar_bench/run_evaluation.py \
    --model qwen35 --protocol think_off_8192 --limit 20 --stratified \
    --run-name smoke

# 全量（可重复执行，--resume 继续）
micromamba run -n vlm-test python evaluation/spar_bench/run_evaluation.py \
    --model qwen35 --protocol think_off_8192 --resume

# 重算指标（不占 GPU）
micromamba run -n vlm-test python evaluation/spar_bench/recompute_metrics.py \
    --all experiments/spar_bench/tiny_rgb

# 生成报告
micromamba run -n vlm-test python evaluation/spar_bench/make_reports.py
```

## 5. 产物结构

```text
experiments/spar_bench/tiny_rgb/
├── dataset_statistics.json
├── task_distribution.csv
├── xiaomi_lens_0/
│   ├── think_off_8192/{predictions.jsonl,metrics.json,summary.csv,experiment.json}
│   ├── think_on_8192/...
│   └── direct_off_64/...
└── qwen3_5_4b/
    └── ... 同上
```

`predictions.jsonl` 每条保存：id / task / img_type / format_type / 问题 / GT /
图片数 / Prompt / 原始输出 / Thinking / 最终回答 / 官方与严格解析结果 /
token 数 / 各阶段耗时 / 显存峰值 / 错误信息。

## 6. 评分口径

- **MCA**：官方 `exact_match`（含子串与首字符回退）→ Accuracy。
- **NA**：官方 `process_na` 抽数 + `mean_relative_accuracy(.5,.95,.05)` → MRA。
- **VCI**：官方 `compute_vci_metric`，5 个方向差值各算一次 MRA 再取均值。
- **Overall**：20 个任务分数的算术平均；Low / Medium / High 同样先算任务分数。
- **Strict**（补充）：最终回答必须是严格格式，解析失败记 0 并标 invalid。
  不做事后模糊匹配，也不从 Thinking 中间过程猜答案。

Thinking On 时官方与严格评分都作用在 `</think>` 之后的最终回答；若 Thinking
未闭合且达到 token 上限，最终回答为空，记 0 并标记 invalid / truncated。

## 7. 正确性保障

- `tests/test_sparbench_metrics.py` 把官方 `utils.py` 作为参考模块加载，
  在随机输入上逐函数对拍 `exact_match`、`process_na`、`mean_relative_accuracy`、
  `compute_vci_metric` 与官方 Prompt 构建。
- 每条样本渲染 Prompt 后检查 Thinking 后缀；处理器后再检查
  `<|vision_start|>` 数量与 `image_grid_thw` 行数，必须等于原始图片数。
- 每条预测立即写入 `predictions.jsonl` 并 `fsync`；JSONL 半行会被自动修复。
- `experiment.json` 保存配置指纹；配置不同时拒绝写入同一目录。
- OOM 会重试；重试耗尽的样本标记为 `retryable`，`--resume` 时自动重跑，
  不会被当成已完成样本。
