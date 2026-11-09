# SPAR-Bench-Tiny 结果核验

> 全部检查直接读取 `predictions.jsonl` / `experiment.json` / `metrics.json`，并用评分代码重新计算。

| 模型 | 协议 | 样本 | 任务 | Official | Strict | Invalid | 截断 | 错误 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| xiaomi_lens_0 | `think_off_8192` | 1000 | 20 | 26.23 | 24.92 | 425 | 78 | 0 | ✅ 通过 |
| xiaomi_lens_0 | `think_on_8192` | 1000 | 20 | 30.70 | 30.12 | 453 | 378 | 0 | ✅ 通过 |
| xiaomi_lens_0 | `direct_off_64` | 1000 | 20 | 39.25 | 39.12 | 55 | 55 | 0 | ✅ 通过 |
| qwen3_5_4b | `think_off_8192` | 1000 | 20 | 36.18 | 25.27 | 427 | 9 | 0 | ✅ 通过 |
| qwen3_5_4b | `think_on_8192` | 1000 | 20 | 27.01 | 26.94 | 549 | 542 | 0 | ✅ 通过 |
| qwen3_5_4b | `direct_off_64` | 1000 | 20 | 43.45 | 41.17 | 47 | 1 | 0 | ✅ 通过 |

## 明细

### xiaomi_lens_0 / `think_off_8192`

- ✅ **样本数**：1000 / 1000
- ✅ **ID 唯一**：1000 个唯一 ID
- ✅ **任务覆盖**：20 种任务各 50 条
- ✅ **输入类型**：single_view=400, multi_view=600
- ✅ **图片完整性**：所有样本的 Prompt 图片数 = 原始图片数
- ✅ **Thinking 模式**：所有样本的 chat template 后缀校验通过
- ✅ **运行状态**：complete
- ✅ **指标可复算**：官方分数与落盘一致
- ✅ **metrics 一致**：n_samples=1000, overall=0.262340

### xiaomi_lens_0 / `think_on_8192`

- ✅ **样本数**：1000 / 1000
- ✅ **ID 唯一**：1000 个唯一 ID
- ✅ **任务覆盖**：20 种任务各 50 条
- ✅ **输入类型**：single_view=400, multi_view=600
- ✅ **图片完整性**：所有样本的 Prompt 图片数 = 原始图片数
- ✅ **Thinking 模式**：所有样本的 chat template 后缀校验通过
- ✅ **运行状态**：complete
- ✅ **指标可复算**：官方分数与落盘一致
- ✅ **metrics 一致**：n_samples=1000, overall=0.306960

### xiaomi_lens_0 / `direct_off_64`

- ✅ **样本数**：1000 / 1000
- ✅ **ID 唯一**：1000 个唯一 ID
- ✅ **任务覆盖**：20 种任务各 50 条
- ✅ **输入类型**：single_view=400, multi_view=600
- ✅ **图片完整性**：所有样本的 Prompt 图片数 = 原始图片数
- ✅ **Thinking 模式**：所有样本的 chat template 后缀校验通过
- ✅ **运行状态**：complete
- ✅ **指标可复算**：官方分数与落盘一致
- ✅ **metrics 一致**：n_samples=1000, overall=0.392480

### qwen3_5_4b / `think_off_8192`

- ✅ **样本数**：1000 / 1000
- ✅ **ID 唯一**：1000 个唯一 ID
- ✅ **任务覆盖**：20 种任务各 50 条
- ✅ **输入类型**：single_view=400, multi_view=600
- ✅ **图片完整性**：所有样本的 Prompt 图片数 = 原始图片数
- ✅ **Thinking 模式**：所有样本的 chat template 后缀校验通过
- ✅ **运行状态**：complete
- ✅ **指标可复算**：官方分数与落盘一致
- ✅ **metrics 一致**：n_samples=1000, overall=0.361780

### qwen3_5_4b / `think_on_8192`

- ✅ **样本数**：1000 / 1000
- ✅ **ID 唯一**：1000 个唯一 ID
- ✅ **任务覆盖**：20 种任务各 50 条
- ✅ **输入类型**：single_view=400, multi_view=600
- ✅ **图片完整性**：所有样本的 Prompt 图片数 = 原始图片数
- ✅ **Thinking 模式**：所有样本的 chat template 后缀校验通过
- ✅ **运行状态**：complete
- ✅ **指标可复算**：官方分数与落盘一致
- ✅ **metrics 一致**：n_samples=1000, overall=0.270060

### qwen3_5_4b / `direct_off_64`

- ✅ **样本数**：1000 / 1000
- ✅ **ID 唯一**：1000 个唯一 ID
- ✅ **任务覆盖**：20 种任务各 50 条
- ✅ **输入类型**：single_view=400, multi_view=600
- ✅ **图片完整性**：所有样本的 Prompt 图片数 = 原始图片数
- ✅ **Thinking 模式**：所有样本的 chat template 后缀校验通过
- ✅ **运行状态**：complete
- ✅ **指标可复算**：官方分数与落盘一致
- ✅ **metrics 一致**：n_samples=1000, overall=0.434460

