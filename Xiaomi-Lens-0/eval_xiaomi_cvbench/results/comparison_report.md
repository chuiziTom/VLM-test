# Xiaomi-Lens-0 CV-Bench 评测报告

生成时间：2026-10-30 05:18:30

## 实验配置

- 模型：`Xiaomi-Lens-0` （`SeerRay-Lab/Xiaomi-Lens-0`）
- Attention：`flash_attention_2`；dtype：bfloat16
- max_new_tokens：8192；do_sample：False（贪心，确定性）
- Prompt 模板：`After answering the question, put the final choice on a separate line using exactly this format: FINAL_ANSWER: (X)
where X is one of the provided option letters.`
- Seed：0；parser version：1.0
- 模型加载耗时：23.32 s；预热：12.77 s
- 整卡显存（结束时）：20.22 GiB 已使用（包含机器上的其他 GPU 进程，不能与 PyTorch 峰值混为一谈）
- 运行状态：completed，完成 2638/2638
- 注意：本机 GPU 同时运行其他用户的进程，绝对耗时受资源竞争影响；Off/On 在同一台机器上顺序运行，相对差异仍可参考，但不宜当作独占 GPU 的基准。

## 数据集分布

```json
{
  "total": 2638,
  "by_type": {
    "2D": 1438,
    "3D": 1200
  },
  "by_task": {
    "Count": 788,
    "Depth": 600,
    "Distance": 600,
    "Relation": 650
  },
  "by_source": {
    "ADE20K": 633,
    "COCO": 805,
    "Omni3D": 1200
  },
  "by_type_task_source": {
    "2D/Count/ADE20K": 342,
    "2D/Count/COCO": 446,
    "2D/Relation/ADE20K": 291,
    "2D/Relation/COCO": 359,
    "3D/Depth/Omni3D": 600,
    "3D/Distance/Omni3D": 600
  },
  "choice_counts": {
    "2": 1850,
    "4": 493,
    "5": 156,
    "6": 139
  },
  "answer_letters": {
    "A": 1112,
    "B": 1104,
    "C": 169,
    "D": 167,
    "E": 63,
    "F": 23
  }
}
```

## 主对比表

| 指标 | Thinking Off | Thinking On | On - Off |
| --- | ---: | ---: | ---: |
| CV-Bench 官方总分 | 87.76% | 87.29% | -0.48 pp |
| 2D Accuracy（官方加权） | 83.44% | 83.41% | -0.03 pp |
| 3D Accuracy（Omni3D） | 92.08% | 91.17% | -0.92 pp |
| 物体计数 (Count) | 74.49% | 73.86% | -0.63 pp |
| 二维空间关系 (Relation) | 95.85% | 96.15% | +0.31 pp |
| 深度顺序 (Depth) | 94.83% | 94.17% | -0.67 pp |
| 相对距离 (Distance) | 89.33% | 88.17% | -1.17 pp |
| ADE20K Accuracy | 77.57% | 78.99% | +1.42 pp |
| COCO Accuracy | 89.32% | 87.83% | -1.49 pp |
| Omni3D Accuracy | 92.08% | 91.17% | -0.92 pp |
| 全部样本普通 Accuracy | 87.76% | 87.23% | -0.53 pp |
| 平均端到端耗时 (s) | 4.98 | 19.58 | 14.60 |
| P95 端到端耗时 (s) | 12.42 | 65.45 | 53.02 |
| 平均耗时倍率 (On/Off) | 1.00x | 3.93x | +2.93x |
| 平均生成耗时 (s) | 4.94 | 19.51 | 14.58 |
| 平均生成 Token | 137.24 | 417.44 | 280.20 |
| 平均 Thinking Token | 0.00 | 302.26 | 302.26 |
| Tokens/s（含 Prefill） | 27.80 | 21.39 | -6.41 |
| 单样本峰值显存 (GiB) | 8.81 | 8.81 | 0.00 |
| Invalid Rate | 0.34% | 0.99% | 0.64% |
| 截断率 | 0.27% | 0.42% | 0.15% |
| 有效回答率 | 99.66% | 99.01% | -0.64% |

> `pp` = percentage points。Thinking Token 按 `</think>` 特殊 Token 精确切分；Tokens/s 包含 Prefill 与 Decode，不是纯 Decode 速度。

## 分模式明细

### Thinking Off

- 完成样本：2638（状态：completed）
- 官方 CV-Bench：87.76%
- 官方 2D：83.44%（ADE20K 77.57% / COCO 89.32%）
- 官方 3D：92.08%
- 全部样本普通 Accuracy：87.76%
- Invalid：9（0.34%）；截断：7（0.27%）
- 平均生成 Token：137.24；Thinking Token：0.00
- 平均生成耗时：4.94 s；P95：12.35 s
- Tokens/s（Prefill+Decode）：27.80
- 单样本峰值显存（平均/最大）：8.81 / 9.36 GiB
- Invalid 原因：`{"truncated_no_final_answer": 6, "malformed_final_answer": 3}`

### Thinking On

- 完成样本：2638（状态：completed）
- 官方 CV-Bench：87.29%
- 官方 2D：83.41%（ADE20K 78.99% / COCO 87.83%）
- 官方 3D：91.17%
- 全部样本普通 Accuracy：87.23%
- Invalid：26（0.99%）；截断：11（0.42%）
- 平均生成 Token：417.44；Thinking Token：302.26
- 平均生成耗时：19.51 s；P95：65.39 s
- Tokens/s（Prefill+Decode）：21.39
- 单样本峰值显存（平均/最大）：8.81 / 9.36 GiB
- Invalid 原因：`{"malformed_final_answer": 15, "truncated_before_final_answer": 11}`

## 同题对比（Off vs On）

- 共同样本：2638
- Off 错、On 对：81
- Off 对、On 错：95
- 预测发生变化：211

| 任务 | Off 正确/总数 | On 正确/总数 | Accuracy 变化 |
| --- | ---: | ---: | ---: |
| 物体计数 (Count) | 587/788 | 582/788 | -0.63 pp |
| 深度顺序 (Depth) | 569/600 | 565/600 | -0.67 pp |
| 相对距离 (Distance) | 536/600 | 529/600 | -1.17 pp |
| 二维空间关系 (Relation) | 623/650 | 625/650 | +0.31 pp |

## 数据结论

- 官方 CV-Bench：Off 87.76%，On 87.29%，Thinking 下降 0.48 pp。
- 2D：Off 83.44%，On 83.41%，变化 -0.03 pp。
- 3D：Off 92.08%，On 91.17%，变化 -0.92 pp。
- 受益任务：二维空间关系 (Relation)；下降任务：物体计数 (Count), 深度顺序 (Depth), 相对距离 (Distance)；持平任务：无。
- 平均端到端耗时：Off 4.98s，On 19.58s，On/Off = 3.93x。
- 平均生成 Token：Off 137.2，On 417.4，增加 280.2。
- Invalid Rate：Off 0.34%，On 0.99%；截断率：Off 0.27%，On 0.42%。
- 同题对比（2638 条）：Off 错/On 对 81 条，Off 对/On 错 95 条，预测改变 211 条。

## 深入解读

- 剔除 invalid 后的有效回答准确率：Off 88.06%，On 88.09%（变化 +0.04 pp）。
- Invalid 数量：Off 9（0.34%），On 26（0.99%）；截断：Off 7，On 11。
- On 的 Thinking Token 占全部输出 Token 的 72.4%（平均 302.3 / 417.4）。
- 输出 Token 倍率 On/Off = 3.04x。
- 端到端耗时倍率 On/Off = 3.93x；Tokens/s：Off 27.80，On 21.39。

| 2D 组合分组 | Off | On | 变化 |
| --- | ---: | ---: | ---: |
| 2D/Count/ADE20K | 216/342 (63.16%) | 224/342 (65.50%) | +2.34 pp |
| 2D/Count/COCO | 371/446 (83.18%) | 358/446 (80.27%) | -2.91 pp |
| 2D/Relation/ADE20K | 275/291 (94.50%) | 276/291 (94.85%) | +0.34 pp |
| 2D/Relation/COCO | 348/359 (96.94%) | 349/359 (97.21%) | +0.28 pp |
| 3D/Depth/Omni3D | 569/600 (94.83%) | 565/600 (94.17%) | -0.67 pp |
| 3D/Distance/Omni3D | 536/600 (89.33%) | 529/600 (88.17%) | -1.17 pp |

## 局限与注意事项

- 本机 GPU 同时运行其他用户的 DRL/CityNav 进程，绝对耗时受资源竞争影响；Off/On 是顺序运行而非独占 GPU 的微基准，耗时倍率应视为近似值。
- 达到 `max_new_tokens=8192` 的样本记为截断，严格解析下通常没有最终答案，按错误计入分母；没有做“补答”或从 Thinking 中间过程猜答案。
- 模型可能自发输出 `<bbox>` / `<|mask|>`；本评测只做文本选择题，不执行 Mask 解码或可视化，这些 token 只计入生成量。
- Invalid 与截断均保留在准确率分母中，因此官方分数低于“只统计有效回答”的分数。

## Adaptive Thinking

Xiaomi-Lens-0 **不支持原生 Adaptive Thinking**。该 checkpoint 的 chat template 只暴露布尔参数 `enable_thinking`，没有 adaptive/auto 开关，也没有 Thinking Budget 参数。因此本报告不包含第三组分数，也未用固定 Token 预算或外部规则冒充原生自适应模式。

## 结论

- 所有准确率均来自严格解析后的 `FINAL_ANSWER: (X)` 行；无效输出按错误计入分母。
- 官方总分使用 `(2D_ADE20K + 2D_COCO)/2` 与 `3D_Omni3D` 的加权公式，不是全样本简单平均。
- 若任一官方分组缺失，对应官方分数标记为不可计算（N/A）。
