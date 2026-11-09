# Xiaomi-Lens-0 vs Qwen3.5-4B — SPAR-Bench-Tiny 综合对比

> 数据：SPAR-Bench-Tiny（RGB，1,000 条，20 种任务）。两组模型使用相同的样本、相同的 Prompt 版本和相同的评分代码。
> 所有数字来自实际预测记录；未完成的实验标为 N/A。

## 1. 总体对比

| 协议 | 模型 | 样本数 | Official | Strict | Low | Medium | High |
|---|---|---:|---:|---:|---:|---:|---:|
| Thinking Off / 8192 | Xiaomi-Lens-0 | 1000 | 26.23 | 24.92 | 42.45 | 15.69 | 15.33 |
| Thinking Off / 8192 | Qwen3.5-4B | 1000 | 36.18 | 25.27 | 46.12 | 24.19 | 31.33 |
| Thinking On / 8192 | Xiaomi-Lens-0 | 1000 | 30.70 | 30.12 | 22.02 | 24.57 | 40.44 |
| Thinking On / 8192 | Qwen3.5-4B | 1000 | 27.01 | 26.94 | 12.10 | 25.11 | 40.89 |
| Direct Answer Off / 64 | Xiaomi-Lens-0 | 1000 | 39.25 | 39.12 | 40.70 | 29.12 | 41.33 |
| Direct Answer Off / 64 | Qwen3.5-4B | 1000 | 43.45 | 41.17 | 48.60 | 31.37 | 42.89 |

## 2. 20 种任务对比（Official）

| 任务 | 层级 | Xiaomi Off | Qwen Off | Xiaomi On | Qwen On | Xiaomi Direct | Qwen Direct |
|---|---|---:|---:|---:|---:|---:|---:|
| `depth_prediction_oc` | low | 60.60 | 66.60 | 30.80 | 20.40 | 61.00 | 69.40 |
| `depth_prediction_oo` | low | 23.80 | 24.40 | 2.20 | 1.80 | 22.60 | 30.00 |
| `distance_prediction_oc` | low | 57.20 | 57.00 | 49.80 | 37.40 | 59.20 | 65.80 |
| `distance_prediction_oo` | low | 37.00 | 45.60 | 19.20 | 4.80 | 34.00 | 49.60 |
| `depth_prediction_oc_mv` | low | 51.20 | 55.20 | 25.20 | 11.40 | 48.00 | 59.60 |
| `depth_prediction_oo_mv` | low | 17.00 | 28.00 | 5.20 | 0.00 | 18.60 | 24.00 |
| `distance_prediction_oo_mv` | low | 26.40 | 39.80 | 12.80 | 1.80 | 21.60 | 34.20 |
| `distance_prediction_oc_mv` | low | 66.40 | 52.40 | 31.00 | 19.20 | 60.60 | 56.20 |
| `view_change_infer` | medium | 7.08 | 2.56 | 13.72 | 5.32 | 7.36 | 10.12 |
| `position_matching` | medium | 36.00 | 62.00 | 54.00 | 60.00 | 52.00 | 60.00 |
| `camera_motion_infer` | medium | 4.00 | 8.00 | 6.00 | 10.00 | 28.00 | 24.00 |
| `obj_spatial_relation_oo` | high | 0.00 | 46.00 | 64.00 | 66.00 | 66.00 | 52.00 |
| `obj_spatial_relation_oc_mv` | high | 22.00 | 50.00 | 42.00 | 48.00 | 62.00 | 62.00 |
| `obj_spatial_relation_oo_mv` | high | 4.00 | 20.00 | 20.00 | 24.00 | 28.00 | 34.00 |
| `spatial_imagination_oc` | high | 0.00 | 10.00 | 20.00 | 26.00 | 22.00 | 30.00 |
| `spatial_imagination_oo` | high | 0.00 | 26.00 | 28.00 | 10.00 | 30.00 | 20.00 |
| `spatial_imagination_oc_mv` | high | 4.00 | 16.00 | 28.00 | 34.00 | 22.00 | 28.00 |
| `spatial_imagination_oo_mv` | high | 2.00 | 28.00 | 14.00 | 16.00 | 8.00 | 18.00 |
| `distance_infer_center_oo` | high | 62.00 | 76.00 | 80.00 | 74.00 | 68.00 | 72.00 |
| `distance_infer_center_oo_mv` | high | 44.00 | 10.00 | 68.00 | 70.00 | 66.00 | 70.00 |

## 3. MCA / NA / VCI 对比

| 协议 | 模型 | MCA | NA (MRA) | VCI |
|---|---|---:|---:|---:|
| Thinking Off / 8192 | Xiaomi-Lens-0 | 16.18 | 42.45 | 7.08 |
| Thinking Off / 8192 | Qwen3.5-4B | 32.00 | 46.12 | 2.56 |
| Thinking On / 8192 | Xiaomi-Lens-0 | 38.55 | 22.02 | 13.72 |
| Thinking On / 8192 | Qwen3.5-4B | 39.82 | 12.10 | 5.32 |
| Direct Answer Off / 64 | Xiaomi-Lens-0 | 41.09 | 40.70 | 7.36 |
| Direct Answer Off / 64 | Qwen3.5-4B | 42.73 | 48.60 | 10.12 |

## 4. 单视角 / 多视角

| 协议 | 模型 | Single-view | Multi-view |
|---|---|---:|---:|
| Thinking Off / 8192 | Xiaomi-Lens-0 | 30.07 | 23.67 |
| Thinking Off / 8192 | Qwen3.5-4B | 43.95 | 31.00 |
| Thinking On / 8192 | Xiaomi-Lens-0 | 36.75 | 26.66 |
| Thinking On / 8192 | Qwen3.5-4B | 30.05 | 24.98 |
| Direct Answer Off / 64 | Xiaomi-Lens-0 | 45.35 | 35.18 |
| Direct Answer Off / 64 | Qwen3.5-4B | 48.60 | 40.01 |

## 5. Thinking On / Off 的影响

只比较 `think_off_8192` 与 `think_on_8192`：两者 Prompt 完全相同，唯一变量是 `enable_thinking`。`direct_off_64` 使用另一版 Prompt，不参与此结论。

| 模型 | Off Official | On Official | 变化 | Off 平均生成 Tokens | On 平均生成 Tokens |
|---|---:|---:|---:|---:|---:|
| Xiaomi-Lens-0 | 26.23 | 30.70 | +4.46 pp | 960 | 4257 |
| Qwen3.5-4B | 36.18 | 27.01 | -9.17 pp | 245 | 6066 |

## 6. 推理速度与显存

| 协议 | 模型 | E2E 均值(s) | E2E P95(s) | Samples/s | Tokens/s | 峰值显存(GiB) |
|---|---|---:|---:|---:|---:|---:|
| Thinking Off / 8192 | Xiaomi-Lens-0 | 27.50 | 219.51 | 0.04 | 35.2 | 9.4 |
| Thinking Off / 8192 | Qwen3.5-4B | 6.67 | 15.66 | 0.15 | 37.8 | 9.3 |
| Thinking On / 8192 | Xiaomi-Lens-0 | 115.19 | 227.66 | 0.01 | 37.0 | 9.4 |
| Thinking On / 8192 | Qwen3.5-4B | 165.53 | 240.70 | 0.01 | 36.7 | 9.3 |
| Direct Answer Off / 64 | Xiaomi-Lens-0 | 0.55 | 2.18 | 1.83 | 24.4 | 9.4 |
| Direct Answer Off / 64 | Qwen3.5-4B | 0.49 | 0.71 | 2.03 | 22.2 | 9.3 |

## 7. 数据质量

| 模型 | 协议 | Invalid | 截断 | 错误 | OOM | Thinking 未闭合 |
|---|---|---:|---:|---:|---:|---:|
| Xiaomi-Lens-0 | Thinking Off / 8192 | 425 | 78 | 0 | 0 | 0 |
| Xiaomi-Lens-0 | Thinking On / 8192 | 453 | 378 | 0 | 0 | 378 |
| Xiaomi-Lens-0 | Direct Answer Off / 64 | 55 | 55 | 0 | 0 | 0 |
| Qwen3.5-4B | Thinking Off / 8192 | 427 | 9 | 0 | 0 | 0 |
| Qwen3.5-4B | Thinking On / 8192 | 549 | 542 | 0 | 0 | 541 |
| Qwen3.5-4B | Direct Answer Off / 64 | 47 | 1 | 0 | 0 | 0 |

## 8. 结论

### 8.1 总体差距

| 协议 | Xiaomi Official | Qwen Official | Qwen - Xiaomi | 领先 |
|---|---:|---:|---:|---|
| Thinking Off / 8192 | 26.23 | 36.18 | +9.94 pp | Qwen |
| Thinking On / 8192 | 30.70 | 27.01 | -3.69 pp | Xiaomi |
| Direct Answer Off / 64 | 39.25 | 43.45 | +4.20 pp | Qwen |

### 8.2 层级差距（Qwen - Xiaomi）

| 协议 | Low | Medium | High |
|---|---:|---:|---:|
| Thinking Off / 8192 | +3.68 | +8.49 | +16.00 |
| Thinking On / 8192 | -9.93 | +0.53 | +0.44 |
| Direct Answer Off / 64 | +7.90 | +2.25 | +1.56 |

### 8.3 任务级差距（|Qwen - Xiaomi| ≥ 10 pp）

| 协议 | 任务 | Xiaomi | Qwen | 差值 |
|---|---|---:|---:|---:|
| Thinking Off / 8192 | `depth_prediction_oo_mv` | 17.00 | 28.00 | +11.00 pp |
| Thinking Off / 8192 | `distance_prediction_oo_mv` | 26.40 | 39.80 | +13.40 pp |
| Thinking Off / 8192 | `distance_prediction_oc_mv` | 66.40 | 52.40 | -14.00 pp |
| Thinking Off / 8192 | `position_matching` | 36.00 | 62.00 | +26.00 pp |
| Thinking Off / 8192 | `obj_spatial_relation_oo` | 0.00 | 46.00 | +46.00 pp |
| Thinking Off / 8192 | `obj_spatial_relation_oc_mv` | 22.00 | 50.00 | +28.00 pp |
| Thinking Off / 8192 | `obj_spatial_relation_oo_mv` | 4.00 | 20.00 | +16.00 pp |
| Thinking Off / 8192 | `spatial_imagination_oc` | 0.00 | 10.00 | +10.00 pp |
| Thinking Off / 8192 | `spatial_imagination_oo` | 0.00 | 26.00 | +26.00 pp |
| Thinking Off / 8192 | `spatial_imagination_oc_mv` | 4.00 | 16.00 | +12.00 pp |
| Thinking Off / 8192 | `spatial_imagination_oo_mv` | 2.00 | 28.00 | +26.00 pp |
| Thinking Off / 8192 | `distance_infer_center_oo` | 62.00 | 76.00 | +14.00 pp |
| Thinking Off / 8192 | `distance_infer_center_oo_mv` | 44.00 | 10.00 | -34.00 pp |
| Thinking On / 8192 | `depth_prediction_oc` | 30.80 | 20.40 | -10.40 pp |
| Thinking On / 8192 | `distance_prediction_oc` | 49.80 | 37.40 | -12.40 pp |
| Thinking On / 8192 | `distance_prediction_oo` | 19.20 | 4.80 | -14.40 pp |
| Thinking On / 8192 | `depth_prediction_oc_mv` | 25.20 | 11.40 | -13.80 pp |
| Thinking On / 8192 | `distance_prediction_oo_mv` | 12.80 | 1.80 | -11.00 pp |
| Thinking On / 8192 | `distance_prediction_oc_mv` | 31.00 | 19.20 | -11.80 pp |
| Thinking On / 8192 | `spatial_imagination_oo` | 28.00 | 10.00 | -18.00 pp |
| Direct Answer Off / 64 | `distance_prediction_oo` | 34.00 | 49.60 | +15.60 pp |
| Direct Answer Off / 64 | `depth_prediction_oc_mv` | 48.00 | 59.60 | +11.60 pp |
| Direct Answer Off / 64 | `distance_prediction_oo_mv` | 21.60 | 34.20 | +12.60 pp |
| Direct Answer Off / 64 | `obj_spatial_relation_oo` | 66.00 | 52.00 | -14.00 pp |
| Direct Answer Off / 64 | `spatial_imagination_oo_mv` | 8.00 | 18.00 | +10.00 pp |

### 8.4 速度与显存

| 模型 | 协议 | E2E 均值(s) | Decode Tokens/s | 峰值显存(GiB) |
|---|---|---:|---:|---:|
| Xiaomi-Lens-0 | Thinking Off / 8192 | 27.50 | 35.4 | 9.4 |
| Qwen3.5-4B | Thinking Off / 8192 | 6.67 | 38.6 | 9.3 |
| Xiaomi-Lens-0 | Thinking On / 8192 | 115.19 | 37.1 | 9.4 |
| Qwen3.5-4B | Thinking On / 8192 | 165.53 | 36.7 | 9.3 |
| Xiaomi-Lens-0 | Direct Answer Off / 64 | 0.55 | 42.4 | 9.4 |
| Qwen3.5-4B | Direct Answer Off / 64 | 0.49 | 42.1 | 9.3 |

### 8.5 综合判断

- **Thinking Off**：Qwen 的 Official 领先 +9.94 pp；但 Strict 下为 +0.35 pp。 官方差距主要来自 `exact_match` 的格式宽容度，严格解析后两模型基本持平。
- **Thinking On**：Qwen - Xiaomi = -3.69 pp（Strict -3.18 pp）。两模型截断分别为 Xiaomi 378/1000、Qwen 542/1000；Low 层为 Xiaomi 22.02、Qwen 12.10，说明低层数值题被 8192 上限拖累最明显。
- **Direct Answer**：Qwen - Xiaomi = +4.20 pp（Strict +2.05 pp）。该协议是两模型各自的最好成绩：Xiaomi 39.25、Qwen 43.45；说明格式约束能显著提升两者的可评分性。
- **VCI 是共同短板**：全部 6 组在 2.56 ~ 13.72 之间（最高为 Xiaomi-Lens-0 / `think_on_8192`）。`view_change_infer` 需要精确输出 5 组移动/旋转参数，当前模型普遍格式不合格或数值偏差大。
- **最佳单项**：Qwen3.5-4B / `direct_off_64`，Official 43.45。
- **工程可用性**：Direct Answer 约 0.5s/样本；Thinking Off 下 Qwen 明显快于 Xiaomi；Thinking On 两组都达到 1~3 分钟/样本，且大量触顶 8192，部署成本远高于 Direct / Off。

## 9. 报告说明与局限

- 本次为 **SPAR-Bench-Tiny RGB-only**，不是完整 SPAR-Bench，也不包含 RGB-D。
- 所有成绩来自 `predictions.jsonl`，`metrics.json` 可由 `recompute_metrics.py` 复算。
- 官方 MCA `exact_match` 含模糊回退，报告同时给出 Strict 作为补充，二者不可混用。
- Multi-view 的 2~3 张图片全部按原始顺序输入，并通过 image token / grid 数量校验。

