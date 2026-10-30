# Changelog

## [0.2.0] - 2026-10-29

### Features
- 新增 `eval_qwen35_cvbench/`：用与 Xiaomi-Lens-0 完全相同的评测协议评测
  `Qwen/Qwen3.5-4B` 在 CV-Bench（2638 条）上的表现，支持 Thinking Off / On。
- 新增 `test_metrics.py`：19 个数据集 / 官方加权分数 / Invalid 计分 / 断点续跑测试。
- 新增跨模型对比报告：`make_report.py` 只读读取 Xiaomi 结果并生成横向对比表。

### Design Rationale
- **复制而非 import** Xiaomi 的 `answer_parser.py` / `cvbench_dataset.py` / `metrics.py`：
  避免向只读目录写入 `__pycache__`，同时用 sha1 证明两边是同一份评分代码，
  保证两个模型的准确率严格可比。
- **只新增模型适配器与 runner**：`model_qwen35.py` 实现与 Xiaomi 适配器相同的
  `load()` / `build_inputs()` / `generate()` 返回结构，其余模块不重写。
- **沿用 `max_new_tokens=8192` 与提示词模板**：与 Xiaomi 逐字符一致，
  Thinking On 的截断率才可直接比较。实测 Off/On 均无截断（On 最大约 2500 token）。
- **共享 GPU 显式标注**：`experiment.json` 记录 `gpu_shared`，
  耗时/吞吐/显存一律视为共享条件下的测量值，不与准确率混为一谈。

### Notes & Caveats
- Qwen3.5-4B 的 chat template 只有布尔 `enable_thinking`，**不支持原生 Adaptive
  Thinking**；`--mode adaptive` 会明确报不支持并退出（码 2），不会静默降级。
- 本环境未安装 `flash-attn`，Attention 使用 `sdpa`（Xiaomi 用 `flash_attention_2`），
  这影响速度但不影响准确率。
- 完整评测耗时受同机其他 GPU 任务影响很大（实测 4 s/样本 ~ 90 s/样本波动），
  运行可随时用 `--resume` 续跑，不会重复评测已完成样本。
