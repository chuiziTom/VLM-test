# Qwen3.5-4B — SPAR-Bench-Tiny 评测报告

> 数据：SPAR-Bench-Tiny（RGB，1,000 条，20 种任务，50 条/任务）。模型：Qwen3.5-4B。

## 1. 模型与部署

| 项目 | 内容 |
|---|---|
| 模型 | Qwen3.5-4B |
| Hugging Face | https://huggingface.co/Qwen/Qwen3.5-4B |
| 运行环境 | vlm-test (micromamba) |
| 本地权重 | `/home/20261024_1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Attention | sdpa |
| dtype / batch | bfloat16 / 1 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| PyTorch / Transformers | 2.10.0+cu128 / 5.20.2 |
| 数据集 SHA-256 | `aa54325a05c494f3a4b47366be7cf11bd540e96f8aadfcba2c7b2f8aa63ccca8` |

## 2. 实验协议

| 协议 | Thinking | max_new_tokens | Prompt | 说明 |
|---|---|---:|---|---|
| `think_off_8192` | Off | 8192 | 官方 Prompt | 不开启显式思考 |
| `think_on_8192` | On | 8192 | 官方 Prompt | 与 Off 仅 `enable_thinking` 不同 |
| `direct_off_64` | Off | 64 | 官方 + 直接回答指令 | 独立 Prompt，要求只输出答案 |

三个协议使用相同的 1,000 条样本与相同顺序；Multi-view 的全部图片按原始顺序输入。

## 3. 运行状态

| 协议 | Thinking | max_new_tokens | Prompt 版本 | 状态 |
|---|---|---:|---|---|
| `think_off_8192` | off | 8192 | `official_v1` | 完成（1000 条） |
| `think_on_8192` | on | 8192 | `official_v1` | 完成（1000 条） |
| `direct_off_64` | off | 64 | `direct_answer_v1` | 完成（1000 条） |

## 4. 总体成绩

官方总分口径：先算 20 个任务各自的分数，再对任务分数取算术平均。Low / Medium / High 同样先算任务分数再取层级均值。

| 协议 | 样本数 | Official Overall | Strict Overall | Low | Medium | High |
|---|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 1000 | 36.18 | 25.27 | 46.12 | 24.19 | 31.33 |
| `think_on_8192` | 1000 | 27.01 | 26.94 | 12.10 | 25.11 | 40.89 |
| `direct_off_64` | 1000 | 43.45 | 41.17 | 48.60 | 31.37 | 42.89 |

## 5. MCA / NA / VCI

| 评分方式 | 指标 | 说明 |
|---|---|---|
| MCA | Accuracy | 官方 `exact_match`（含子串 / 首字符回退） |
| NA | MRA:.5:.95:.05 | 官方 `process_na` + `mean_relative_accuracy` |
| VCI | VCI metric | 官方 `compute_vci_metric`，5 个方向各算一次 MRA |

| 协议 | MCA Official | MCA Strict | NA Official | NA Strict | VCI Official | VCI Strict |
|---|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 32.00 | 16.36 | 46.12 | 40.58 | 2.56 | 0.72 |
| `think_on_8192` | 39.82 | 39.82 | 12.10 | 12.10 | 5.32 | 4.00 |
| `direct_off_64` | 42.73 | 39.09 | 48.60 | 48.30 | 10.12 | 7.04 |

## 6. 20 种任务得分（Official）

| 任务 | 层级 | 评分 | Thinking Off / 8192 Official | Thinking On / 8192 Official | Direct Answer Off / 64 Official |
|---|---|---|---:|---:|---:|
| `depth_prediction_oc` | low | NA | 66.60 | 20.40 | 69.40 |
| `depth_prediction_oo` | low | NA | 24.40 | 1.80 | 30.00 |
| `distance_prediction_oc` | low | NA | 57.00 | 37.40 | 65.80 |
| `distance_prediction_oo` | low | NA | 45.60 | 4.80 | 49.60 |
| `depth_prediction_oc_mv` | low | NA | 55.20 | 11.40 | 59.60 |
| `depth_prediction_oo_mv` | low | NA | 28.00 | 0.00 | 24.00 |
| `distance_prediction_oo_mv` | low | NA | 39.80 | 1.80 | 34.20 |
| `distance_prediction_oc_mv` | low | NA | 52.40 | 19.20 | 56.20 |
| `view_change_infer` | medium | VCI | 2.56 | 5.32 | 10.12 |
| `position_matching` | medium | MCA | 62.00 | 60.00 | 60.00 |
| `camera_motion_infer` | medium | MCA | 8.00 | 10.00 | 24.00 |
| `obj_spatial_relation_oo` | high | MCA | 46.00 | 66.00 | 52.00 |
| `obj_spatial_relation_oc_mv` | high | MCA | 50.00 | 48.00 | 62.00 |
| `obj_spatial_relation_oo_mv` | high | MCA | 20.00 | 24.00 | 34.00 |
| `spatial_imagination_oc` | high | MCA | 10.00 | 26.00 | 30.00 |
| `spatial_imagination_oo` | high | MCA | 26.00 | 10.00 | 20.00 |
| `spatial_imagination_oc_mv` | high | MCA | 16.00 | 34.00 | 28.00 |
| `spatial_imagination_oo_mv` | high | MCA | 28.00 | 16.00 | 18.00 |
| `distance_infer_center_oo` | high | MCA | 76.00 | 74.00 | 72.00 |
| `distance_infer_center_oo_mv` | high | MCA | 10.00 | 70.00 | 70.00 |

## 7. 20 种任务得分（Strict，补充）

Strict 只接受最终回答区域里的严格格式：选择题必须是单个选项字母，数值题必须是单个数值，VCI 必须完整给出 5 组参数；不做事后模糊匹配。

| 任务 | 层级 | Thinking Off / 8192 Strict | Thinking On / 8192 Strict | Direct Answer Off / 64 Strict | Invalid 数（三协议合计） |
|---|---|---:|---:|---:|---:|
| `depth_prediction_oc` | low | 66.60 | 20.40 | 69.40 | 35 |
| `depth_prediction_oo` | low | 24.40 | 1.80 | 28.00 | 49 |
| `distance_prediction_oc` | low | 57.00 | 37.40 | 65.80 | 14 |
| `distance_prediction_oo` | low | 45.60 | 4.80 | 49.60 | 44 |
| `depth_prediction_oc_mv` | low | 34.80 | 11.40 | 59.60 | 58 |
| `depth_prediction_oo_mv` | low | 19.20 | 0.00 | 23.60 | 69 |
| `distance_prediction_oo_mv` | low | 34.40 | 1.80 | 34.20 | 52 |
| `distance_prediction_oc_mv` | low | 42.60 | 19.20 | 56.20 | 39 |
| `view_change_infer` | medium | 0.72 | 4.00 | 7.04 | 86 |
| `position_matching` | medium | 56.00 | 60.00 | 60.00 | 18 |
| `camera_motion_infer` | medium | 8.00 | 10.00 | 24.00 | 64 |
| `obj_spatial_relation_oo` | high | 4.00 | 66.00 | 24.00 | 74 |
| `obj_spatial_relation_oc_mv` | high | 36.00 | 48.00 | 62.00 | 29 |
| `obj_spatial_relation_oo_mv` | high | 8.00 | 24.00 | 34.00 | 58 |
| `spatial_imagination_oc` | high | 2.00 | 26.00 | 22.00 | 52 |
| `spatial_imagination_oo` | high | 22.00 | 10.00 | 16.00 | 55 |
| `spatial_imagination_oc_mv` | high | 6.00 | 34.00 | 28.00 | 53 |
| `spatial_imagination_oo_mv` | high | 4.00 | 16.00 | 18.00 | 81 |
| `distance_infer_center_oo` | high | 26.00 | 74.00 | 72.00 | 41 |
| `distance_infer_center_oo_mv` | high | 8.00 | 70.00 | 70.00 | 52 |

## 8. 单视角 / 多视角

| 协议 | Single-view（400） | Multi-view（600） |
|---|---:|---:|
| `think_off_8192` | 43.95 | 31.00 |
| `think_on_8192` | 30.05 | 24.98 |
| `direct_off_64` | 48.60 | 40.01 |

Single-view 与 Multi-view 的任务集合不同，这一对比反映的是**任务组合差异**，不能单独解释为“多图输入能力”的因果效应。

## 9. 性能指标

**（1）延迟与吞吐**

| 协议 | 加载(s) | 总耗时(h) | E2E 均值(s) | E2E 中位(s) | E2E P95(s) | E2E 最大(s) | Prefill 均值(s) | Decode 均值(s) | Samples/s | Decode Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 6.7 | 1.86 | 6.67 | 0.58 | 15.66 | 213.78 | 0.136 | 6.34 | 0.150 | 38.6 |
| `think_on_8192` | 6.5 | 45.98 | 165.53 | 211.25 | 240.70 | 279.50 | 0.145 | 165.17 | 0.006 | 36.7 |
| `direct_off_64` | 7.0 | 0.14 | 0.49 | 0.43 | 0.71 | 2.29 | 0.139 | 0.16 | 2.025 | 42.1 |

**（2）Token 与显存**

| 协议 | 平均 Prompt | 平均生成 | 平均 Thinking | 平均 Answer | 总生成 Tokens | 最大生成 | 截断数 | 加载后 allocated(GiB) | 单样本峰值(GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 2560 | 245 | 0 | 245 | 244777 | 8192 | 9 | 8.46 | 9.26 |
| `think_on_8192` | 2558 | 6066 | 6062 | 3 | 6065541 | 8192 | 542 | 8.46 | 9.26 |
| `direct_off_64` | 2577 | 7 | 0 | 7 | 6544 | 64 | 1 | 8.46 | 9.26 |

说明：E2E 为单条样本端到端耗时（图片解码 + 预处理 + Prefill + Decode）；Decode Tokens/s 只统计 decode 阶段；峰值显存为 PyTorch `max_memory_allocated`，与整卡 `nvidia-smi` 使用量不同。


## 10. 数据质量与异常

| 协议 | Invalid | 截断 | 推理错误 | OOM | Thinking 未闭合 | 模式校验失败 | 图片数校验失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 427 | 9 | 0 | 0 | 0 | 0 | 0 |
| `think_on_8192` | 549 | 542 | 0 | 0 | 541 | 0 | 0 |
| `direct_off_64` | 47 | 1 | 0 | 0 | 0 | 0 | 0 |

### 失败案例示例

**Thinking Off / 8192**

- `distance_infer_center_oo` / id=201：reason=`mca_format_mismatch`，最终回答=`'B. desk (green point)'`，GT=`A`
- `spatial_imagination_oc` / id=326：reason=`truncated_no_final_answer`，最终回答=`'To determine the correct option, we need to analyze the spatial relationship between the objects in the 3D world coordin'`，GT=`C`
- `view_change_infer` / id=400：reason=`vci_field_count_mismatch`，最终回答=`'To determine the camera movement, we can compare the perspective and content of the two images.  -   **Camera Position:*'`，GT=`move_right:1.6,move_down:0,move_backward:0,rotate_up:15,rotate_left:75`

**Thinking On / 8192**

- `depth_prediction_oc` / id=0：reason=`truncated_before_final_answer`，最终回答=`''`，GT=`0.9`
- `view_change_infer` / id=411：reason=`vci_format_mismatch`，最终回答=`'move_left:1.0,move_down:0.1,move_back:0.5,rotate_down:10,rotate_left:30'`，GT=`move_right:0,move_up:0,move_forward:0,rotate_up:5,rotate_left:15`
- `view_change_infer` / id=435：reason=`vci_field_count_mismatch`，最终回答=`'move_forward:0.5,move_down:0.1,rotate_down:5,rotate_left:0'`，GT=`move_left:0.2,move_down:0,move_forward:0.4,rotate_down:10,rotate_right:10`

**Direct Answer Off / 64**

- `obj_spatial_relation_oo` / id=250：reason=`mca_format_mismatch`，最终回答=`'D. right, , farther'`，GT=`D`
- `view_change_infer` / id=401：reason=`vci_field_count_mismatch`，最终回答=`'move_forward:0.2,rotate_up:10,rotate_right:0'`，GT=`move_right:0.4,move_down:0,move_backward:0.2,rotate_up:10,rotate_left:0`
- `depth_prediction_oo_mv` / id=505：reason=`truncated_no_final_answer`，最终回答=`'Based on the provided images and the depth of the couch (red point) being 2.5 meters:  1.  **Ottoman (green point):** In'`，GT=`0.6`


## 11. 结论与下一步

- 三个协议中 **Direct Answer Off / 64** 的官方总分最高：Official 43.45 / Strict 41.17。
- 该协议层级得分：Low 48.60、Medium 31.37、High 42.89；MCA 42.73、NA 48.60、VCI 10.12。
- Thinking Off：Official 比 Strict 高 +10.91 pp。官方 `exact_match` 的首字符 / 子串回退会奖励部分解释性文本，Strict 不奖励；两者不能混用。
- Thinking Off 有 427 条严格解析失败、9 条截断；平均生成 245 tokens。
- Thinking On 相对 Off 的官方变化 -9.17 pp；On 截断 542/1000，平均生成 6066 tokens。
- Direct Answer：E2E 均值 0.49s，截断 1/1000，VCI 10.12。
- Direct Answer 使用独立 Prompt 版本，不能与 Thinking Off 做单变量归因。
- 下一步可在完整 SPAR-Bench 上复核 Tiny 结论，并补做 RGB-D 版本；VCI 是两模型共同的最弱项，值得单独做 prompt / 输出格式优化。

## 12. 测试局限

- 本次只评测 **SPAR-Bench-Tiny（RGB）**，不能代表完整 SPAR-Bench 或 RGB-D 结果。
- 每题 50 条，单任务分数置信区间较宽；任务间 1~2 个百分点的差异不宜过度解读。
- 官方分数基于官方评分函数；`exact_match` 含“预测是答案子串”与“首字符相等”回退逻辑，因此官方 MCA 分数可能高于严格解析分数。
- 官方 `exact_match` 对空串会因子串匹配返回 1；本实现的评分入口把“空最终回答”显式记 0 并标 invalid，避免把没答完算成正确。
- Thinking On 时官方 / 严格评分都作用在 `</think>` 之后的最终回答；若生成被截断且没有闭合 Thinking，最终回答为空，记 0 并标记 invalid。
- Direct Answer（64 tokens）使用独立的 Prompt 版本，不能把它与 Thinking Off 的差异解释为单一变量消融。
- 推理耗时包含 CUDA 同步与逐样本显存统计开销；不同时刻 GPU 上还有其他进程，绝对速度仅供参考，模型间对比应在同一时段内解读。

