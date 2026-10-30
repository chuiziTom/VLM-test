# Qwen3.5-4B × CV-Bench 评测报告

生成时间：2026-10-29 14:47:37

## 实验配置

- 模型：`Qwen3.5-4B` （`/home/20261024_1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`）
- Attention：`sdpa`；dtype：bfloat16；CPU offload：`None`
- max_new_tokens：8192；do_sample：False（贪心，确定性）
- Prompt 模板：`After answering the question, put the final choice on a separate line using exactly this format: FINAL_ANSWER: (X)
where X is one of the provided option letters.`
- Seed：0；parser version：1.0
- 模型加载耗时：3.71 s；预热：0.00 s
- 运行状态：completed，完成 16/16
- 是否与他人共享 GPU：`True`

## 数据集分布

```json
{
  "total": 16,
  "by_type": {
    "2D": 12,
    "3D": 4
  },
  "by_task": {
    "Count": 6,
    "Depth": 2,
    "Distance": 2,
    "Relation": 6
  },
  "by_source": {
    "ADE20K": 6,
    "COCO": 6,
    "Omni3D": 4
  },
  "by_type_task_source": {
    "2D/Count/ADE20K": 3,
    "2D/Count/COCO": 3,
    "2D/Relation/ADE20K": 3,
    "2D/Relation/COCO": 3,
    "3D/Depth/Omni3D": 2,
    "3D/Distance/Omni3D": 2
  },
  "choice_counts": {
    "2": 10,
    "4": 3,
    "5": 1,
    "6": 2
  },
  "answer_letters": {
    "A": 8,
    "B": 3,
    "C": 2,
    "D": 2,
    "E": 1
  }
}
```

## A. Qwen3.5-4B：Thinking Off vs On

| 指标 | Thinking Off | Thinking On | On - Off |
| --- | ---: | ---: | ---: |
| CV-Bench 官方总分 | 91.67% | 95.83% | +4.17 pp |
| 2D Accuracy（官方加权） | 83.33% | 91.67% | +8.33 pp |
| 3D Accuracy（Omni3D） | 100.00% | 100.00% | +0.00 pp |
| 物体计数 (Count) | 66.67% | 83.33% | +16.67 pp |
| 二维空间关系 (Relation) | 100.00% | 100.00% | +0.00 pp |
| 深度顺序 (Depth) | 100.00% | 100.00% | +0.00 pp |
| 相对距离 (Distance) | 100.00% | 100.00% | +0.00 pp |
| ADE20K Accuracy | 100.00% | 100.00% | +0.00 pp |
| COCO Accuracy | 66.67% | 83.33% | +16.67 pp |
| Omni3D Accuracy | 100.00% | 100.00% | +0.00 pp |
| 全部样本普通 Accuracy | 87.50% | 93.75% | +6.25 pp |
| 平均端到端耗时 (s) | 3.66 | 16.52 | 12.86 |
| P95 端到端耗时 (s) | 11.81 | 48.39 | 36.59 |
| 平均耗时倍率 (On/Off) | 1.00x | 4.51x | +3.51x |
| 平均生成耗时 (s) | 3.64 | 16.51 | 12.86 |
| 平均生成 Token | 102.44 | 599.00 | 496.56 |
| 平均 Thinking Token | 0.00 | 540.56 | 540.56 |
| Tokens/s（含 Prefill） | 28.13 | 36.29 | 8.16 |
| 单样本峰值显存 (GiB) | 8.63 | 8.63 | 0.00 |
| Invalid Rate | 0.00% | 0.00% | 0.00% |
| 截断率 | 0.00% | 0.00% | 0.00% |
| 有效回答率 | 100.00% | 100.00% | 0.00% |

> `pp` = percentage points。Thinking Token 按 `</think>` 特殊 Token 精确切分；
> Tokens/s 包含 Prefill 与 Decode，不是纯 Decode 速度。

## 分模式明细

### Thinking Off

- 完成样本：16（状态：completed）
- 官方 CV-Bench：91.67%
- 官方 2D：83.33%（ADE20K 100.00% / COCO 66.67%）
- 官方 3D：100.00%
- 全部样本普通 Accuracy：87.50%（14/16）
- Invalid：0（0.00%）；截断：0（0.00%）；推理失败：0
- 平均生成 Token：102.44（P95 225.50）；Thinking Token：0.00
- 平均生成耗时：3.64 s；P95：11.77 s
- Tokens/s（Prefill+Decode）：28.13
- 单样本峰值显存（平均/最大）：8.63 / 9.12 GiB
- **注意：这是子集结果，不是完整测试集成绩。**

### Thinking On

- 完成样本：16（状态：completed）
- 官方 CV-Bench：95.83%
- 官方 2D：91.67%（ADE20K 100.00% / COCO 83.33%）
- 官方 3D：100.00%
- 全部样本普通 Accuracy：93.75%（15/16）
- Invalid：0（0.00%）；截断：0（0.00%）；推理失败：0
- 平均生成 Token：599.00（P95 1767.50）；Thinking Token：540.56
- 平均生成耗时：16.51 s；P95：48.38 s
- Tokens/s（Prefill+Decode）：36.29
- 单样本峰值显存（平均/最大）：8.63 / 9.12 GiB
- **注意：这是子集结果，不是完整测试集成绩。**

## 同题对比（Qwen Off vs On）

- 共同样本：16
- Off 错、On 对：1
- Off 对、On 错：0
- 预测发生变化：1

| 任务 | Off 正确/总数 | On 正确/总数 | Accuracy 变化 |
| --- | ---: | ---: | ---: |
| 物体计数 (Count) | 4/6 | 5/6 | +16.67 pp |
| 深度顺序 (Depth) | 2/2 | 2/2 | +0.00 pp |
| 相对距离 (Distance) | 2/2 | 2/2 | +0.00 pp |
| 二维空间关系 (Relation) | 6/6 | 6/6 | +0.00 pp |

## B. Qwen3.5-4B vs Xiaomi-Lens-0

在 `/home/20261024_1/Workspace/qyh/VLM-test/Xiaomi-Lens-0/eval_xiaomi_cvbench/results` 未找到 Xiaomi 的结果，跳过横向对比。

## Adaptive Thinking

Qwen3.5-4B **不支持原生 Adaptive Thinking**。其 chat template 只有布尔参数 `enable_thinking`，没有 adaptive/auto 开关，也没有 Thinking Budget 参数。因此本报告不含第三组分数，也没有用固定 Token 预算或外部路由冒充原生自适应模式。

## 可比较性说明

- 两个模型的准确率使用**逐字节相同**的解析器与指标代码（`answer_parser.py`、`metrics.py` 的 sha1 与 Xiaomi 目录一致），Prompt 格式指令也完全相同，因此准确率可以直接比较。
- 推理速度、吞吐与显存**不能**直接比较：两个模型架构不同（Qwen3.5-4B 与 Xiaomi-Lens-0 参数量不同），Attention 实现不同，而且都在与他人共享的 GPU 上运行。
- 所有准确率均来自严格解析后的 `FINAL_ANSWER: (X)` 行；无效输出按错误计入分母。
- 官方总分使用 `(2D_ADE20K + 2D_COCO)/2` 与 `3D_Omni3D` 的加权公式，不是全样本简单平均；任一官方分组缺失时该分数记为 N/A。
