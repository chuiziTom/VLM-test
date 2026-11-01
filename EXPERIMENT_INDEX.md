# CV-Bench 实验索引

本文件为 CV-Bench 相关实验建立统一的**规范名称 ↔ 实际路径**映射。

> 原则：**不移动、不重命名历史结果目录**。原有脚本、README、报告都直接引用
> `results/thinking_off/` 与 `results/thinking_on/`；移动它们会破坏断点续跑和报告
> 复现。因此这里只建立索引，历史文件保持在原位置。

---

## 1. 命名规范

```text
<protocol>_<thinking>_<length>[ _<prompt_version> ]
```

| 字段 | 取值 |
|---|---|
| `protocol` | `legacy`（历史长回答评测） / `direct_answer`（强制直接输出答案） |
| `thinking` | `think_off` / `think_on` |
| `length` | `max_new_tokens` 的实际取值 |
| `prompt_version` | Direct Answer Prompt 的版本，历史实验无此字段 |

规范名称：

- 历史实验：`legacy_max_new_tokens_8192`
- 本次实验：`direct_answer_think_off_max_new_tokens_64_v1`

---

## 2. 历史实验（只读，保持原位置）

| 规范名称 | 模型 | 环境 | Thinking | Max New Tokens | Prompt 版本 | 实际结果路径 | 状态 |
|---|---|---|---|---:|---|---|---|
| `legacy_max_new_tokens_8192` | Xiaomi-Lens-0 | `xiaomi-lens` | Off | 8192 | legacy（`FINAL_ANSWER` 格式指令） | `Xiaomi-Lens-0/eval_xiaomi_cvbench/results/thinking_off/` | completed 2638/2638 |
| `legacy_max_new_tokens_8192` | Xiaomi-Lens-0 | `xiaomi-lens` | On | 8192 | legacy | `Xiaomi-Lens-0/eval_xiaomi_cvbench/results/thinking_on/` | completed 2638/2638 |
| `legacy_max_new_tokens_8192` | Qwen3.5-4B | `vlm-test` | Off | 8192 | legacy | `eval_qwen35_cvbench/results/thinking_off/` | completed 2638/2638 |
| `legacy_max_new_tokens_8192` | Qwen3.5-4B | `vlm-test` | On | 8192 | legacy | `eval_qwen35_cvbench/results/thinking_on/` | completed 2638/2638 |

**配置核实来源**：四个运行目录中的 `experiment.json` / `metrics.json` 均记录
`"max_new_tokens": 8192`、`"do_sample": false`、`"split": "all"`、
`"limit": null`、`"selected_samples": 2638`。历史 Prompt 的
`prompt_template_sha1 = cd11a2a3d0ec89fdacf1b321c824f708b1c503b5`。

历史实验的小规模试跑（不属于正式成绩）：

| 路径 | 说明 |
|---|---|
| `eval_qwen35_cvbench/results/single/` | Qwen 单样本测试 |
| `eval_qwen35_cvbench/results/smoke16/` | Qwen 16 条分层冒烟 |
| `Xiaomi-Lens-0/eval_xiaomi_cvbench/results/smoke/` | Xiaomi 单样本测试 |
| `Xiaomi-Lens-0/eval_xiaomi_cvbench/results/smoke20/` | Xiaomi 20 条分层冒烟 |

---

## 3. 本次实验（Direct Answer v1）

| 规范名称 | 模型 | 环境 | Thinking | Max New Tokens | Prompt 版本 | 实际结果路径 | 状态 |
|---|---|---|---|---:|---|---|---|
| `direct_answer_think_off_max_new_tokens_64_v1` | Xiaomi-Lens-0 | `xiaomi-lens` | Off | 64 | `direct_answer_v1` | `eval_cvbench_direct_answer/results/direct_answer_think_off_max_new_tokens_64_v1/xiaomi_lens/` | 见运行后 `experiment.json` |
| `direct_answer_think_off_max_new_tokens_64_v1` | Qwen3.5-4B | `vlm-test` | Off | 64 | `direct_answer_v1` | `eval_cvbench_direct_answer/results/direct_answer_think_off_max_new_tokens_64_v1/qwen35_4b/` | 见运行后 `experiment.json` |

Prompt：`eval_cvbench_direct_answer/prompts/direct_answer_v1.txt`
（SHA-1 记录在各运行目录的 `metrics.json → direct_answer.prompt_sha1`）。

小样本冒烟（不属于正式成绩）：

| 路径 | 说明 |
|---|---|
| `eval_cvbench_direct_answer/results/smoke_direct_answer_v1/xiaomi_lens/` | Xiaomi 16 条分层冒烟 |
| `eval_cvbench_direct_answer/results/smoke_direct_answer_v1/qwen35_4b/` | Qwen 16 条分层冒烟 |

---

## 4. 路径引用保护说明

- 历史结果目录**未被移动或重命名**，原脚本仍可直接用
  `--run-name` / `--output-dir` 指向它们。
- 本次新结果只写入 `eval_cvbench_direct_answer/`，不会与历史 `thinking_off` /
  `thinking_on` 混在一起。
- 历史与本次实验的 Prompt、`max_new_tokens`、解析规则均不同，**不能当作严格
  的对照组**；报告中的跨实验比较必须明确标注这一点。
