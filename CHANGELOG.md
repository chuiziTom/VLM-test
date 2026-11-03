# Changelog

## [0.5.0] - 2026-11-03

### Features
- 新增 `evaluation/spar_bench/`：在 SPAR-Bench-Tiny（RGB，1,000 条、
  20 种任务）上评测 Xiaomi-Lens-0 与 Qwen3.5-4B。
- 三种协议：`think_off_8192`、`think_on_8192`、`direct_off_64`，
  两个模型共 6 组实验。
- 复现官方三类评分：MCA `exact_match`、NA `process_na` + `MRA:.5:.95:.05`、
  VCI `compute_vci_metric`；官方总分按 20 个任务分数的算术平均聚合。
- 额外提供 Strict 解析（严格选项 / 数值 / 五轴 VCI 格式），与 Official
  分数分开保存。
- 新增数据集检查、断点续跑评测、指标重算、结果核验与报告生成脚本。
- 新增 `reports/spar_bench_tiny_dataset_profile.md` 及单模型 / 对比报告。

### Design Rationale
- **不修改 CV-Bench 代码**：SPAR-Bench 的输入是多图、评分有三类，
  单图 Adapter 和 CV-Bench Accuracy 都不能直接复用，因此新建独立目录。
- **多图按原始顺序全部输入**：处理器返回后校验 `<|vision_start|>` 数量与
  `image_grid_thw` 行数，杜绝 Multi-view 被截断成首图。
- **Official 与 Strict 并列**：官方 `exact_match` 含子串与首字符回退，
  会把解释性文本判对；Strict 只接受严格格式，二者不能混为一谈。
- **官方函数逐行复现并做对拍测试**：`tests/test_sparbench_metrics.py`
  直接加载官方 `utils.py` 作为参考模块比较输出。
- **逐样本落盘 + 配置指纹**：长任务可中断续跑，且不同协议 / Prompt
  不会混进同一结果文件。

### Notes & Caveats
- 仅 Tiny RGB 版本，不代表完整 SPAR-Bench，也不含 RGB-D。
- Thinking On 的官方 / 严格评分都作用在 `</think>` 之后的最终回答；
  若达到 8192 token 仍未闭合 Thinking，最终回答为空，记 0 并标记 invalid。
- Direct Answer 使用独立 Prompt 版本，不与 Thinking Off 做单变量归因。

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
