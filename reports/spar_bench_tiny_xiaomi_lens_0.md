# Xiaomi-Lens-0 — SPAR-Bench-Tiny 评测报告

> 数据：SPAR-Bench-Tiny（RGB，1,000 条，20 种任务，50 条/任务）。模型：Xiaomi-Lens-0。

## 1. 模型与部署

| 项目 | 内容 |
|---|---|
| 模型 | Xiaomi-Lens-0 |
| Hugging Face | https://huggingface.co/SeerRay-Lab/Xiaomi-Lens-0 |
| 运行环境 | xiaomi-lens (micromamba) |
| 本地权重 | `/home/20261024_1/.cache/huggingface/hub/models--SeerRay-Lab--Xiaomi-Lens-0/snapshots/68f57cdde1a19f751e78b18af04efa8ca8bd7fce` |
| Attention | flash_attention_2 |
| dtype / batch | bfloat16 / 1 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| PyTorch / Transformers | 2.10.0+cu128 / 5.4.0 |
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
| `think_off_8192` | 1000 | 26.23 | 24.92 | 42.45 | 15.69 | 15.33 |
| `think_on_8192` | 1000 | 30.70 | 30.12 | 22.02 | 24.57 | 40.44 |
| `direct_off_64` | 1000 | 39.25 | 39.12 | 40.70 | 29.12 | 41.33 |

## 5. MCA / NA / VCI

| 评分方式 | 指标 | 说明 |
|---|---|---|
| MCA | Accuracy | 官方 `exact_match`（含子串 / 首字符回退） |
| NA | MRA:.5:.95:.05 | 官方 `process_na` + `mean_relative_accuracy` |
| VCI | VCI metric | 官方 `compute_vci_metric`，5 个方向各算一次 MRA |

| 协议 | MCA Official | MCA Strict | NA Official | NA Strict | VCI Official | VCI Strict |
|---|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 16.18 | 13.82 | 42.45 | 42.45 | 7.08 | 6.72 |
| `think_on_8192` | 38.55 | 38.55 | 22.02 | 22.02 | 13.72 | 2.20 |
| `direct_off_64` | 41.09 | 40.91 | 40.70 | 40.62 | 7.36 | 7.36 |

## 6. 20 种任务得分（Official）

| 任务 | 层级 | 评分 | Thinking Off / 8192 Official | Thinking On / 8192 Official | Direct Answer Off / 64 Official |
|---|---|---|---:|---:|---:|
| `depth_prediction_oc` | low | NA | 60.60 | 30.80 | 61.00 |
| `depth_prediction_oo` | low | NA | 23.80 | 2.20 | 22.60 |
| `distance_prediction_oc` | low | NA | 57.20 | 49.80 | 59.20 |
| `distance_prediction_oo` | low | NA | 37.00 | 19.20 | 34.00 |
| `depth_prediction_oc_mv` | low | NA | 51.20 | 25.20 | 48.00 |
| `depth_prediction_oo_mv` | low | NA | 17.00 | 5.20 | 18.60 |
| `distance_prediction_oo_mv` | low | NA | 26.40 | 12.80 | 21.60 |
| `distance_prediction_oc_mv` | low | NA | 66.40 | 31.00 | 60.60 |
| `view_change_infer` | medium | VCI | 7.08 | 13.72 | 7.36 |
| `position_matching` | medium | MCA | 36.00 | 54.00 | 52.00 |
| `camera_motion_infer` | medium | MCA | 4.00 | 6.00 | 28.00 |
| `obj_spatial_relation_oo` | high | MCA | 0.00 | 64.00 | 66.00 |
| `obj_spatial_relation_oc_mv` | high | MCA | 22.00 | 42.00 | 62.00 |
| `obj_spatial_relation_oo_mv` | high | MCA | 4.00 | 20.00 | 28.00 |
| `spatial_imagination_oc` | high | MCA | 0.00 | 20.00 | 22.00 |
| `spatial_imagination_oo` | high | MCA | 0.00 | 28.00 | 30.00 |
| `spatial_imagination_oc_mv` | high | MCA | 4.00 | 28.00 | 22.00 |
| `spatial_imagination_oo_mv` | high | MCA | 2.00 | 14.00 | 8.00 |
| `distance_infer_center_oo` | high | MCA | 62.00 | 80.00 | 68.00 |
| `distance_infer_center_oo_mv` | high | MCA | 44.00 | 68.00 | 66.00 |

## 7. 20 种任务得分（Strict，补充）

Strict 只接受最终回答区域里的严格格式：选择题必须是单个选项字母，数值题必须是单个数值，VCI 必须完整给出 5 组参数；不做事后模糊匹配。

| 任务 | 层级 | Thinking Off / 8192 Strict | Thinking On / 8192 Strict | Direct Answer Off / 64 Strict | Invalid 数（三协议合计） |
|---|---|---:|---:|---:|---:|
| `depth_prediction_oc` | low | 60.60 | 30.80 | 61.00 | 23 |
| `depth_prediction_oo` | low | 23.80 | 2.20 | 22.40 | 36 |
| `distance_prediction_oc` | low | 57.20 | 49.80 | 59.20 | 7 |
| `distance_prediction_oo` | low | 37.00 | 19.20 | 34.00 | 27 |
| `depth_prediction_oc_mv` | low | 51.20 | 25.20 | 48.00 | 27 |
| `depth_prediction_oo_mv` | low | 17.00 | 5.20 | 18.20 | 35 |
| `distance_prediction_oo_mv` | low | 26.40 | 12.80 | 21.60 | 36 |
| `distance_prediction_oc_mv` | low | 66.40 | 31.00 | 60.60 | 26 |
| `view_change_infer` | medium | 6.72 | 2.20 | 7.36 | 42 |
| `position_matching` | medium | 36.00 | 54.00 | 52.00 | 26 |
| `camera_motion_infer` | medium | 4.00 | 6.00 | 28.00 | 75 |
| `obj_spatial_relation_oo` | high | 0.00 | 64.00 | 66.00 | 60 |
| `obj_spatial_relation_oc_mv` | high | 6.00 | 42.00 | 62.00 | 58 |
| `obj_spatial_relation_oo_mv` | high | 0.00 | 20.00 | 26.00 | 81 |
| `spatial_imagination_oc` | high | 0.00 | 20.00 | 22.00 | 77 |
| `spatial_imagination_oo` | high | 0.00 | 28.00 | 30.00 | 69 |
| `spatial_imagination_oc_mv` | high | 0.00 | 28.00 | 22.00 | 85 |
| `spatial_imagination_oo_mv` | high | 0.00 | 14.00 | 8.00 | 117 |
| `distance_infer_center_oo` | high | 62.00 | 80.00 | 68.00 | 4 |
| `distance_infer_center_oo_mv` | high | 44.00 | 68.00 | 66.00 | 22 |

## 8. 单视角 / 多视角

| 协议 | Single-view（400） | Multi-view（600） |
|---|---:|---:|
| `think_off_8192` | 30.07 | 23.67 |
| `think_on_8192` | 36.75 | 26.66 |
| `direct_off_64` | 45.35 | 35.18 |

Single-view 与 Multi-view 的任务集合不同，这一对比反映的是**任务组合差异**，不能单独解释为“多图输入能力”的因果效应。

## 9. 性能指标

**（1）延迟与吞吐**

| 协议 | 加载(s) | 总耗时(h) | E2E 均值(s) | E2E 中位(s) | E2E P95(s) | E2E 最大(s) | Prefill 均值(s) | Decode 均值(s) | Samples/s | Decode Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 7.4 | 7.64 | 27.50 | 0.62 | 219.51 | 262.32 | 0.141 | 27.15 | 0.036 | 35.4 |
| `think_on_8192` | 7.7 | 32.00 | 115.19 | 87.73 | 227.66 | 244.04 | 0.134 | 114.82 | 0.009 | 37.1 |
| `direct_off_64` | 7.7 | 0.15 | 0.55 | 0.39 | 2.18 | 2.29 | 0.135 | 0.18 | 1.830 | 42.4 |

**（2）Token 与显存**

| 协议 | 平均 Prompt | 平均生成 | 平均 Thinking | 平均 Answer | 总生成 Tokens | 最大生成 | 截断数 | 加载后 allocated(GiB) | 单样本峰值(GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 2560 | 960 | 0 | 960 | 959831 | 8192 | 78 | 8.61 | 9.39 |
| `think_on_8192` | 2558 | 4257 | 4245 | 11 | 4257144 | 8192 | 378 | 8.61 | 9.39 |
| `direct_off_64` | 2577 | 8 | 0 | 8 | 7741 | 64 | 55 | 8.61 | 9.40 |

说明：E2E 为单条样本端到端耗时（图片解码 + 预处理 + Prefill + Decode）；Decode Tokens/s 只统计 decode 阶段；峰值显存为 PyTorch `max_memory_allocated`，与整卡 `nvidia-smi` 使用量不同。


## 10. 数据质量与异常

| 协议 | Invalid | 截断 | 推理错误 | OOM | Thinking 未闭合 | 模式校验失败 | 图片数校验失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `think_off_8192` | 425 | 78 | 0 | 0 | 0 | 0 | 0 |
| `think_on_8192` | 453 | 378 | 0 | 0 | 378 | 0 | 0 |
| `direct_off_64` | 55 | 55 | 0 | 0 | 0 | 0 | 0 |

### 失败案例示例

**Thinking Off / 8192**

- `distance_infer_center_oo` / id=231：reason=`mca_format_mismatch`，最终回答=`'To determine which object is farther from the chair (red point), we need to estimate the 3D positions of the objects rel'`，GT=`B`
- `obj_spatial_relation_oo` / id=286：reason=`truncated_no_final_answer`，最终回答=`'To determine the spatial relationship, we first identify the 3D center points of the two objects based on their 2D proje'`，GT=`C`
- `view_change_infer` / id=431：reason=`vci_field_count_mismatch`，最终回答=`'move_forward:1.5,rotate_down:15,rotate_left:10'`，GT=`move_right:0.8,move_up:1.8,move_forward:1.4,rotate_up:35,rotate_left:135`

**Thinking On / 8192**

- `depth_prediction_oc` / id=1：reason=`truncated_before_final_answer`，最终回答=`''`，GT=`3.5`
- `distance_prediction_oo` / id=154：reason=`na_format_mismatch`，最终回答=`'The red pillow center is at <bbox>[556, 371, 590, 413]</bbox><mask></mask> and the blue clothing center is at <bbox>[59,'`，GT=`1.3`
- `distance_infer_center_oo` / id=235：reason=`mca_format_mismatch`，最终回答=`'The chair with the green point <bbox>[264, 177, 469, 281]</bbox><mask></mask> is located in the background, adjacent to '`，GT=`A`

**Direct Answer Off / 64**

- `spatial_imagination_oo` / id=374：reason=`truncated_no_final_answer`，最终回答=`'The question asks for the spatial relationship between the trash can (red bbox) and the kitchen apron (yellow bbox) **af'`，GT=`C`


## 11. 结论与下一步

- 三个协议中 **Direct Answer Off / 64** 的官方总分最高：Official 39.25 / Strict 39.12。
- 该协议层级得分：Low 40.70、Medium 29.12、High 41.33；MCA 41.09、NA 40.70、VCI 7.36。
- Thinking Off：Official 比 Strict 高 +1.32 pp。官方 `exact_match` 的首字符 / 子串回退会奖励部分解释性文本，Strict 不奖励；两者不能混用。
- Thinking Off 有 425 条严格解析失败、78 条截断；平均生成 960 tokens。
- Thinking On 相对 Off 的官方变化 +4.46 pp；On 截断 378/1000，平均生成 4257 tokens。
- Direct Answer：E2E 均值 0.55s，截断 55/1000，VCI 7.36。
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

