# Changelog

## [0.4.0] - 2026-11-01

### Features
- 项目结构调整为 `models/`、`datasets/`、`evaluation/`、`experiments/`、
  `reports/`、`analysis/`、`notebooks/`、`docs/`。
- 评测代码统一到 `evaluation/`：`common/` 保存两个模型共用的
  `cvbench_dataset.py`、`answer_parser.py`、`metrics.py`、`test_answer_parser.py`；
  `qwen35_4b/`、`xiaomi_lens/`、`direct_answer/` 保存各自的适配器与入口。
- 六组 CV-Bench 结果统一归档到
  `experiments/cv_bench/{legacy_8192,direct_answer_64_v1}/<model>/`。
- 小样本试跑归档到 `experiments/cv_bench/smoke/`。
- 正式报告集中到 `reports/legacy_8192/`、`reports/direct_answer_64_v1/`；
  独立核验脚本与派生数据移到 `analysis/`。
- Historical Notebook 集中到 `notebooks/` 并重命名，源码单元格中的旧绝对路径已更新。

### Design Rationale
- **删除 4 个逐字节重复的通用模块**，改为单一实现 + 显式导入，
  避免两个模型各自维护副本后分叉。
- **保留 `dataset -> datasets` 兼容符号链接**：历史 `experiment.json` 的
  数据集指纹使用 `dataset/CV-Bench` 字符串，改字符串会破坏 `--resume`。
- **不移动模型权重**：权重仍在 Hugging Face 本地缓存，`models/` 只放源码。
- **不修改历史实验输出内容**：只移动目录；`experiment.json` 中记录的
  运行时路径保持原样。

### Notes & Caveats
- 六组结果在重构后均通过 `--validate`：2638/2638、官方分与 `metrics.json` 一致。
- Direct Answer 历史结果的 `experiment.json` 记录了重构前的代码目录，
  若执行 `--resume` 会触发指纹不一致；这些运行已完成，核验请用 `--validate`。
- 删除了 `__pycache__` / `*.pyc` 等生成文件；未删除任何实验结果、日志或报告。

## [0.3.0] - 2026-11-01

### Features
- 新增 `eval_cvbench_direct_answer/`：Xiaomi-Lens-0 与 Qwen3.5-4B 的统一
  **Direct Answer（Thinking Off）** 评测，`max_new_tokens=64`，2638 条全量。
- 新增统一 Prompt `prompts/direct_answer_v1.txt` 与整段严格解析器
  `common/direct_parser.py`（17 个单元测试）。
- 新增历史实验索引 `EXPERIMENT_INDEX.md` 与机器可读清单
  `experiment_manifest.json`，区分 `legacy_max_new_tokens_8192` 与
  `direct_answer_think_off_max_new_tokens_64_v1`。
- 新增三份报告：两个单模型报告 + 一份综合对比报告。

### Design Rationale
- **只读复用**原脚本：数据加载、指标、模型适配器按文件路径导入，
  未修改任何历史脚本或预测文件。
- **不移动历史目录**：原有 README/脚本直接引用 `results/thinking_off|on`，
  因此只建立命名映射，避免破坏断点续跑与复现。
- **严格解析**：整段必须恰好是 `FINAL_ANSWER: (X)`，任何额外文字
  （包括 `(A) 1` 这种“选项+计数”）都记为 Invalid 并计入分母。
- **同一 Prompt、同一生成长度**：两个模型逐字符使用同一指令，
  Prompt SHA-1 记录在 `metrics.json`。

### Notes & Caveats
- Direct Answer 与历史 8192 实验的 Prompt、生成长度、解析规则不同，
  **不能作为严格对照组**；报告中的跨实验比较均标注为参考。
- Qwen 出现 24 条（0.91%）格式违规，全部为 `(A) 1` 类输出；
  Xiaomi 为 0 条。
- GPU 全程与其他任务共享，速度指标只作参考。

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
