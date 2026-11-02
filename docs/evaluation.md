# 评测协议与代码结构

## 1. 代码组织

```text
evaluation/
├── common/                 # 两个模型共用，只有一份
│   ├── cvbench_dataset.py  # parquet 读取、图片解码、uid、分布统计
│   ├── answer_parser.py    # 历史协议：逐行解析最终回答区
│   ├── metrics.py          # 官方加权 Accuracy、耗时/Token/显存
│   └── test_answer_parser.py
├── qwen35_4b/
│   ├── model_qwen35.py     # 模型适配器
│   ├── run_qwen35_cvbench.py
│   ├── make_report.py
│   └── test_metrics.py
├── xiaomi_lens/
│   ├── xiaomi_lens_adapter.py
│   ├── run_evaluation.py
│   ├── bench_speed.py
│   └── make_report.py
└── direct_answer/
    ├── prompts/direct_answer_v1.txt
    ├── common/{direct_parser.py, runner_core.py, module_loader.py}
    ├── run_qwen35_direct_answer.py
    ├── run_xiaomi_direct_answer.py
    └── tests/test_direct_parser.py
```

新增模型时，只需要：

1. 写一个适配器，提供与现有适配器相同的 `load()` / `build_inputs()` / `generate()`；
2. 复用 `evaluation/common/` 的数据与指标；
3. 在 `experiments/cv_bench/` 下建立结果目录；
4. 在 `docs/models.md` 与 `EXPERIMENT_INDEX.md` 登记。

不需要复制整套评测工程。

---

## 2. 历史协议（legacy_8192）

- 数据：CV-Bench 2638 条（2D 1438 / 3D 1200）。
- 输入：`record.prompt + FINAL_ANSWER 格式指令`。
- 生成：贪心，`max_new_tokens=8192`。
- 解析：按 `</think>` 切分，只在最终回答区逐行匹配 `FINAL_ANSWER: (X)`。
- 结果：`experiments/cv_bench/legacy_8192/<model>/thinking_{off,on}/`。

Thinking Off 与 On 只差 chat template 的 `enable_thinking` 布尔参数。

---

## 3. Direct Answer 协议（direct_answer_64_v1）

- 输入：`record.prompt + direct_answer_v1 指令`，要求只输出一行答案。
- 生成：贪心，`max_new_tokens=64`。
- Thinking：`enable_thinking=False`。
- 解析：整段严格匹配 `FINAL_ANSWER: (X)`；任何额外文字、多个答案、
  格式偏差都记为 `invalid`，并保留在 Accuracy 分母中。
- 结果：`experiments/cv_bench/direct_answer_64_v1/<model>/`。

Prompt 文件：`evaluation/direct_answer/prompts/direct_answer_v1.txt`
（SHA-1 记录在每个运行的 `metrics.json`）。

---

## 4. 官方评分

```text
Accuracy_2D = (Accuracy_ADE20K + Accuracy_COCO) / 2
Accuracy_3D = Accuracy_Omni3D
CV-Bench Accuracy = (Accuracy_2D + Accuracy_3D) / 2
```

同时报告全部样本普通 Accuracy。两者因来源样本量不等而略有差异，
对外比较使用官方加权总分。

Invalid、格式违规、截断、推理失败都计入分母，不从统计中删除。

---

## 5. 结果文件

每个运行目录包含：

| 文件 | 内容 |
|---|---|
| `predictions.jsonl` | 逐样本预测、原始生成、耗时、Token、显存 |
| `metrics.json` | 官方分数、分组统计、实验配置与运行状态 |
| `summary.csv` | 汇总表，便于快速查看 |
| `experiment.json` | 完整实验配置（含数据集指纹与状态） |

小样本试跑放在 `experiments/cv_bench/smoke/`，不属于正式成绩。

---

## 6. 独立核验

`analysis/` 下的脚本只读原始结果，重新解析并复算指标：

```bash
micromamba run -n vlm-test python -B analysis/legacy_8192/verify_results.py
micromamba run -n vlm-test python -B analysis/direct_answer_64_v1/verify_and_compare.py
```

核验内容包括：uid 覆盖、重复样本、解析一致性、官方分数偏差。

---

## 7. 注意事项

- 两个模型使用不同的 Attention 实现与 Transformers 版本，速度不能严格横向比较。
- 本机 GPU 长期与其他任务共享，`experiment.json` 中的 `gpu_shared` 会记录这一点。
- 历史 8192 与 Direct Answer 的 Prompt、生成长度、解析规则不同，不能当作严格对照组。
