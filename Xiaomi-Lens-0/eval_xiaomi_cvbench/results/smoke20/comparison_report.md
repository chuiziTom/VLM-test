# Xiaomi-Lens-0 CV-Bench 评测报告

生成时间：2026-10-29 11:18:48

## 实验配置

- 模型：`Xiaomi-Lens-0` （`SeerRay-Lab/Xiaomi-Lens-0`）
- Attention：`flash_attention_2`；dtype：bfloat16
- max_new_tokens：8192；do_sample：False（贪心，确定性）
- Prompt 模板：`After answering the question, put the final choice on a separate line using exactly this format: FINAL_ANSWER: (X)
where X is one of the provided option letters.`
- Seed：0；parser version：1.0
- 模型加载耗时：11.65 s；预热：9.78 s
- 整卡显存（结束时）：60.88 GiB 已使用（包含机器上的其他 GPU 进程，不能与 PyTorch 峰值混为一谈）
- 运行状态：completed，完成 20/20
- 注意：本机 GPU 同时运行其他用户的进程，绝对耗时受资源竞争影响；Off/On 在同一台机器上顺序运行，相对差异仍可参考，但不宜当作独占 GPU 的基准。

## 数据集分布

```json
{
  "total": 20,
  "by_type": {
    "2D": 14,
    "3D": 6
  },
  "by_task": {
    "Count": 8,
    "Depth": 3,
    "Distance": 3,
    "Relation": 6
  },
  "by_source": {
    "ADE20K": 7,
    "COCO": 7,
    "Omni3D": 6
  },
  "by_type_task_source": {
    "2D/Count/ADE20K": 4,
    "2D/Count/COCO": 4,
    "2D/Relation/ADE20K": 3,
    "2D/Relation/COCO": 3,
    "3D/Depth/Omni3D": 3,
    "3D/Distance/Omni3D": 3
  },
  "choice_counts": {
    "2": 12,
    "4": 5,
    "5": 1,
    "6": 2
  },
  "answer_letters": {
    "A": 10,
    "B": 4,
    "C": 2,
    "D": 3,
    "E": 1
  }
}
```

## 主对比表

| 指标 | Thinking Off | Thinking On | On - Off |
| --- | ---: | ---: | ---: |
| CV-Bench 官方总分 | 89.29% | 89.29% | +0.00 pp |
| 2D Accuracy（官方加权） | 78.57% | 78.57% | +0.00 pp |
| 3D Accuracy（Omni3D） | 100.00% | 100.00% | +0.00 pp |
| 物体计数 (Count) | 62.50% | 62.50% | +0.00 pp |
| 二维空间关系 (Relation) | 100.00% | 100.00% | +0.00 pp |
| 深度顺序 (Depth) | 100.00% | 100.00% | +0.00 pp |
| 相对距离 (Distance) | 100.00% | 100.00% | +0.00 pp |
| ADE20K Accuracy | 71.43% | 71.43% | +0.00 pp |
| COCO Accuracy | 85.71% | 85.71% | +0.00 pp |
| Omni3D Accuracy | 100.00% | 100.00% | +0.00 pp |
| 全部样本普通 Accuracy | 85.00% | 85.00% | +0.00 pp |
| 平均端到端耗时 (s) | 6.02 | 14.21 | 8.19 |
| P95 端到端耗时 (s) | 17.47 | 33.59 | 16.11 |
| 平均耗时倍率 (On/Off) | 1.00x | 2.36x | +1.36x |
| 平均生成耗时 (s) | 5.94 | 14.14 | 8.19 |
| 平均生成 Token | 140.75 | 372.65 | 231.90 |
| 平均 Thinking Token | 0.00 | 258.25 | 258.25 |
| Tokens/s（含 Prefill） | 23.68 | 26.36 | 2.68 |
| 单样本峰值显存 (GiB) | 8.79 | 8.79 | -0.00 |
| Invalid Rate | 0.00% | 0.00% | 0.00% |
| 截断率 | 0.00% | 0.00% | 0.00% |
| 有效回答率 | 100.00% | 100.00% | 0.00% |

> `pp` = percentage points。Thinking Token 按 `</think>` 特殊 Token 精确切分；Tokens/s 包含 Prefill 与 Decode，不是纯 Decode 速度。

## 分模式明细

### Thinking Off

- 完成样本：20（状态：completed）
- 官方 CV-Bench：89.29%
- 官方 2D：78.57%（ADE20K 71.43% / COCO 85.71%）
- 官方 3D：100.00%
- 全部样本普通 Accuracy：85.00%
- Invalid：0（0.00%）；截断：0（0.00%）
- 平均生成 Token：140.75；Thinking Token：0.00
- 平均生成耗时：5.94 s；P95：17.18 s
- Tokens/s（Prefill+Decode）：23.68
- 单样本峰值显存（平均/最大）：8.79 / 9.26 GiB
- **注意：这是子集结果，不是完整测试集成绩。**

### Thinking On

- 完成样本：20（状态：completed）
- 官方 CV-Bench：89.29%
- 官方 2D：78.57%（ADE20K 71.43% / COCO 85.71%）
- 官方 3D：100.00%
- 全部样本普通 Accuracy：85.00%
- Invalid：0（0.00%）；截断：0（0.00%）
- 平均生成 Token：372.65；Thinking Token：258.25
- 平均生成耗时：14.14 s；P95：33.51 s
- Tokens/s（Prefill+Decode）：26.36
- 单样本峰值显存（平均/最大）：8.79 / 9.26 GiB
- **注意：这是子集结果，不是完整测试集成绩。**

## 同题对比（Off vs On）

- 共同样本：20
- Off 错、On 对：0
- Off 对、On 错：0
- 预测发生变化：1

| 任务 | Off 正确/总数 | On 正确/总数 | Accuracy 变化 |
| --- | ---: | ---: | ---: |
| 物体计数 (Count) | 5/8 | 5/8 | +0.00 pp |
| 深度顺序 (Depth) | 3/3 | 3/3 | +0.00 pp |
| 相对距离 (Distance) | 3/3 | 3/3 | +0.00 pp |
| 二维空间关系 (Relation) | 6/6 | 6/6 | +0.00 pp |

## 数据结论

- 官方 CV-Bench：Off 89.29%，On 89.29%，Thinking 不变 0.00 pp。
- 2D：Off 78.57%，On 78.57%，变化 +0.00 pp。
- 3D：Off 100.00%，On 100.00%，变化 +0.00 pp。
- 受益任务：无；下降任务：无；持平任务：物体计数 (Count), 深度顺序 (Depth), 相对距离 (Distance), 二维空间关系 (Relation)。
- 平均端到端耗时：Off 6.02s，On 14.21s，On/Off = 2.36x。
- 平均生成 Token：Off 140.8，On 372.6，增加 231.9。
- Invalid Rate：Off 0.00%，On 0.00%；截断率：Off 0.00%，On 0.00%。
- 同题对比（20 条）：Off 错/On 对 0 条，Off 对/On 错 0 条，预测改变 1 条。

## Adaptive Thinking

Xiaomi-Lens-0 **不支持原生 Adaptive Thinking**。该 checkpoint 的 chat template 只暴露布尔参数 `enable_thinking`，没有 adaptive/auto 开关，也没有 Thinking Budget 参数。因此本报告不包含第三组分数，也未用固定 Token 预算或外部规则冒充原生自适应模式。

## 结论

- 所有准确率均来自严格解析后的 `FINAL_ANSWER: (X)` 行；无效输出按错误计入分母。
- 官方总分使用 `(2D_ADE20K + 2D_COCO)/2` 与 `3D_Omni3D` 的加权公式，不是全样本简单平均。
- 若任一官方分组缺失，对应官方分数标记为不可计算（N/A）。
