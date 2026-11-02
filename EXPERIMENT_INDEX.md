# CV-Bench 实验索引

本文件是 CV-Bench 全部实验的规范名称与实际路径映射。2026-11-01 项目重构后，
所有实验结果统一归档在 `experiments/cv_bench/` 下。

---

## 1. 命名规范

```text
<protocol>_<thinking>_<length>[_<prompt_version>]
```

| 字段 | 取值 |
|---|---|
| `protocol` | `legacy`（历史长回答评测） / `direct_answer`（强制直接输出答案） |
| `thinking` | `think_off` / `think_on` |
| `length` | `max_new_tokens` 的实际取值 |
| `prompt_version` | Direct Answer Prompt 版本；历史实验无此字段 |

- 历史实验：`legacy_max_new_tokens_8192`
- Direct Answer：`direct_answer_think_off_max_new_tokens_64_v1`

---

## 2. 历史 8192 实验（4 组，完整）

| 规范名称 | 模型 | 环境 | Thinking | Max New Tokens | 结果路径 | 状态 |
|---|---|---|---|---:|---|---|
| `legacy_max_new_tokens_8192` | Xiaomi-Lens-0 | `xiaomi-lens` | Off | 8192 | `experiments/cv_bench/legacy_8192/xiaomi_lens/thinking_off/` | completed 2638/2638 |
| `legacy_max_new_tokens_8192` | Xiaomi-Lens-0 | `xiaomi-lens` | On | 8192 | `experiments/cv_bench/legacy_8192/xiaomi_lens/thinking_on/` | completed 2638/2638 |
| `legacy_max_new_tokens_8192` | Qwen3.5-4B | `vlm-test` | Off | 8192 | `experiments/cv_bench/legacy_8192/qwen35_4b/thinking_off/` | completed 2638/2638 |
| `legacy_max_new_tokens_8192` | Qwen3.5-4B | `vlm-test` | On | 8192 | `experiments/cv_bench/legacy_8192/qwen35_4b/thinking_on/` | completed 2638/2638 |

每个目录包含：`predictions.jsonl`、`metrics.json`、`summary.csv`、`experiment.json`。

配置核实：四个运行目录的 `experiment.json` / `metrics.json` 均记录
`"max_new_tokens": 8192`、`"do_sample": false`、`"split": "all"`、
`"limit": null`、`"selected_samples": 2638`；
历史 Prompt 的 `prompt_template_sha1 = cd11a2a3d0ec89fdacf1b321c824f708b1c503b5`。

额外的历史文件（同样位于 `legacy_8192/` 下）：

| 文件 | 说明 |
|---|---|
| `xiaomi_lens/comparison.json` / `comparison_report.md` | Xiaomi Off/On 自动对比 |
| `xiaomi_lens/speed_bench.json` | 早期速度测试记录 |
| `qwen35_4b/comparison.json` | Qwen Off/On 同题对比 |

> 说明：这些结果目录中的 `experiment.json` 记录的是**运行时**的绝对路径
> （例如旧的 `dataset/` 目录）。项目重构后没有改写这些配置，以保持原始实验记录
> 的真实性；`dataset` 现为指向 `datasets/` 的兼容符号链接，所以旧路径仍然可读。

---

## 3. Direct Answer v1 实验（2 组，完整）

| 规范名称 | 模型 | 环境 | Thinking | Max New Tokens | Prompt | 结果路径 | 状态 |
|---|---|---|---|---:|---|---|---|
| `direct_answer_think_off_max_new_tokens_64_v1` | Xiaomi-Lens-0 | `xiaomi-lens` | Off | 64 | `direct_answer_v1` | `experiments/cv_bench/direct_answer_64_v1/xiaomi_lens/` | completed 2638/2638 |
| `direct_answer_think_off_max_new_tokens_64_v1` | Qwen3.5-4B | `vlm-test` | Off | 64 | `direct_answer_v1` | `experiments/cv_bench/direct_answer_64_v1/qwen35_4b/` | completed 2638/2638 |

Prompt 文件：`evaluation/direct_answer/prompts/direct_answer_v1.txt`
（SHA-1 记录在各运行目录的 `metrics.json → direct_answer.prompt_sha1`）。

---

## 4. 小规模冒烟测试（非正式成绩）

| 路径 | 说明 |
|---|---|
| `experiments/cv_bench/smoke/xiaomi_lens/smoke/` | Xiaomi 单样本测试 |
| `experiments/cv_bench/smoke/xiaomi_lens/smoke20/` | Xiaomi 20 条分层冒烟 |
| `experiments/cv_bench/smoke/qwen35_4b/single/` | Qwen 单样本测试 |
| `experiments/cv_bench/smoke/qwen35_4b/smoke16/` | Qwen 16 条分层冒烟 |
| `experiments/cv_bench/smoke/direct_answer/xiaomi_lens/` | Xiaomi Direct Answer 16 条冒烟 |
| `experiments/cv_bench/smoke/direct_answer/qwen35_4b/` | Qwen Direct Answer 16 条冒烟 |

---

## 5. 正式报告

| 报告 | 路径 |
|---|---|
| 历史 Off/On 综合对比 | `reports/legacy_8192/cvbench_qwen35_xiaomi_comparison.md` |
| Xiaomi Direct Answer 单模型 | `reports/direct_answer_64_v1/xiaomi_lens_direct_answer_report.md` |
| Qwen Direct Answer 单模型 | `reports/direct_answer_64_v1/qwen35_4b_direct_answer_report.md` |
| Direct Answer 两模型对比 | `reports/direct_answer_64_v1/cvbench_direct_answer_comparison.md` |

独立核验脚本与派生数据：

| 内容 | 路径 |
|---|---|
| 历史 8192 结果复算 | `analysis/legacy_8192/verify_results.py` + `verification.json` |
| Direct Answer 结果复算 | `analysis/direct_answer_64_v1/verify_and_compare.py` + `verification.json` |

---

## 6. 复现与校验

```bash
# 历史 8192：校验四组结果（不加载模型）
micromamba run -n vlm-test   python -B evaluation/qwen35_4b/run_qwen35_cvbench.py --mode all --validate
micromamba run -n xiaomi-lens python -B evaluation/xiaomi_lens/run_evaluation.py   --mode all --validate

# Direct Answer：校验两组结果
micromamba run -n vlm-test   python -B evaluation/direct_answer/run_qwen35_direct_answer.py --validate
micromamba run -n xiaomi-lens python -B evaluation/direct_answer/run_xiaomi_direct_answer.py --validate

# 独立复算
micromamba run -n vlm-test python -B analysis/legacy_8192/verify_results.py
micromamba run -n vlm-test python -B analysis/direct_answer_64_v1/verify_and_compare.py
```

**续跑说明**：

- 历史 8192 结果仍可用原命令加 `--resume`，配置指纹中的数据集字符串未变。
- Direct Answer 历史结果的 `experiment.json` 记录了重构前的代码目录，
  若用 `--resume`，指纹校验会提示配置不一致。这些运行已经完成；
  需要核验时请使用 `--validate`，不要为了续跑而改写原始 `experiment.json`。

---

## 7. 跨实验比较的限制

历史 8192 与 Direct Answer 的 Prompt、`max_new_tokens`、解析规则都不同，
**不能当作严格对照组**；报告中的跨实验数字均标注为参考趋势。
