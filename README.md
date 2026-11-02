# VLM-test

视觉语言模型（VLM）在空间理解 Benchmark 上的本地评测项目。

目前覆盖：

- **模型**：`Xiaomi-Lens-0`、`Qwen3.5-4B`
- **Benchmark**：`CV-Bench`（2638 条测试样本）
- **实验协议**：
  - `legacy_8192`：历史长回答评测，`max_new_tokens=8192`，Thinking Off / On；
  - `direct_answer_64_v1`：强制直接输出答案，`max_new_tokens=64`，Thinking Off。

六组 CV-Bench 全量实验均已完成（每组 2638/2638），结果与报告完整保留。

---

## 1. 目录结构

```text
VLM-test/
├── README.md                     # 项目入口（本文件）
├── CHANGELOG.md                  # 版本与结构调整记录
├── EXPERIMENT_INDEX.md           # 实验规范名称 ↔ 实际路径
├── experiment_manifest.json      # 机器可读实验清单
├── environment-vlm-test.yml      # Qwen 环境说明
├── requirements-vlm-test.txt
│
├── models/
│   └── Xiaomi-Lens-0/            # 官方模型源码（权重不在此目录）
│
├── datasets/
│   ├── CV-Bench/                 # 完整测试集（parquet + samples）
│   ├── SPAR-Bench/               # 仅说明文档，尚未下载
│   ├── VSI-Bench/                # 部分数据（test.jsonl）
│   └── download_*.py
│
├── dataset -> datasets           # 兼容符号链接（见第 6 节）
│
├── evaluation/
│   ├── common/                   # 两模型共用的数据加载、解析、指标
│   ├── qwen35_4b/                # Qwen 适配器 + legacy runner
│   ├── xiaomi_lens/              # Xiaomi 适配器 + legacy runner
│   └── direct_answer/            # Direct Answer 协议与入口
│
├── experiments/cv_bench/
│   ├── legacy_8192/
│   │   ├── xiaomi_lens/{thinking_off,thinking_on}/
│   │   └── qwen35_4b/{thinking_off,thinking_on}/
│   ├── direct_answer_64_v1/
│   │   ├── xiaomi_lens/
│   │   └── qwen35_4b/
│   ├── smoke/                    # 小样本试跑，不算正式成绩
│   └── logs/
│
├── reports/
│   ├── legacy_8192/              # 历史 Off/On 综合报告
│   └── direct_answer_64_v1/      # 两份单模型报告 + 综合对比
│
├── analysis/                     # 独立核验脚本与派生 JSON
│   ├── legacy_8192/
│   └── direct_answer_64_v1/
│
├── notebooks/                    # 历史 Notebook（路径已更新）
│
└── docs/
    ├── environment_setup.md      # 环境配置
    ├── datasets.md               # 数据集清单
    ├── models.md                 # 模型清单
    └── evaluation.md             # 评测协议与代码结构
```

---

## 2. 环境

两个环境**独立保留**，不要合并：

| 模型 | Micromamba 环境 | 关键版本 |
|---|---|---|
| Qwen3.5-4B | `vlm-test` | Python 3.11.18 / PyTorch 2.10.0+cu128 / Transformers 5.20.2 |
| Xiaomi-Lens-0 | `xiaomi-lens` | Python 3.11.18 / PyTorch 2.10.0+cu128 / Transformers 5.4.0 |

```bash
micromamba run -n vlm-test   python -c "import torch; print(torch.__version__)"
micromamba run -n xiaomi-lens python -c "import torch; print(torch.__version__)"
```

详细配置见 `docs/environment_setup.md`。

模型权重位于 Hugging Face 本地缓存，不在仓库内，也不需要复制：

- `Qwen/Qwen3.5-4B`
- `SeerRay-Lab/Xiaomi-Lens-0`

---

## 3. 运行评测

### 3.1 历史 8192 协议

```bash
# Qwen（vlm-test）
micromamba run -n vlm-test python -B \
    evaluation/qwen35_4b/run_qwen35_cvbench.py --mode all --resume

# Xiaomi（xiaomi-lens）
micromamba run -n xiaomi-lens python -B \
    evaluation/xiaomi_lens/run_evaluation.py --mode all --resume
```

默认结果目录：

- `experiments/cv_bench/legacy_8192/qwen35_4b/`
- `experiments/cv_bench/legacy_8192/xiaomi_lens/`

### 3.2 Direct Answer v1 协议

```bash
# Xiaomi
micromamba run -n xiaomi-lens python -B \
    evaluation/direct_answer/run_xiaomi_direct_answer.py --resume

# Qwen
micromamba run -n vlm-test python -B \
    evaluation/direct_answer/run_qwen35_direct_answer.py --resume
```

默认结果目录：`experiments/cv_bench/direct_answer_64_v1/<model>/`。

### 3.3 校验已有结果（不加载模型）

```bash
micromamba run -n vlm-test   python -B evaluation/qwen35_4b/run_qwen35_cvbench.py --mode all --validate
micromamba run -n xiaomi-lens python -B evaluation/xiaomi_lens/run_evaluation.py --mode all --validate
micromamba run -n vlm-test   python -B evaluation/direct_answer/run_qwen35_direct_answer.py --validate
micromamba run -n xiaomi-lens python -B evaluation/direct_answer/run_xiaomi_direct_answer.py --validate
```

--- 

## 4. 正式报告

| 内容 | 路径 |
|---|---|
| 历史 Off/On 综合对比 | `reports/legacy_8192/cvbench_qwen35_xiaomi_comparison.md` |
| Xiaomi Direct Answer | `reports/direct_answer_64_v1/xiaomi_lens_direct_answer_report.md` |
| Qwen Direct Answer | `reports/direct_answer_64_v1/qwen35_4b_direct_answer_report.md` |
| Direct Answer 横向对比 | `reports/direct_answer_64_v1/cvbench_direct_answer_comparison.md` |

实验结果与报告的对应关系见 `EXPERIMENT_INDEX.md`。

---

## 5. 命名与归档约定

- Python 文件、目录使用小写 `snake_case`。
- 实验目录使用协议名：`legacy_8192`、`direct_answer_64_v1`。
- 正式报告放 `reports/`；Agent 核验脚本与派生数据放 `analysis/`；
  原始预测、指标、日志放 `experiments/`，三者不混放。
- 新增模型时，只需：
  1. 在 `evaluation/<model>/` 添加模型适配器；
  2. 复用 `evaluation/common/` 的数据加载、解析与指标；
  3. 在 `experiments/cv_bench/<model>/` 归档结果。

---

## 6. 注意事项

1. **`dataset` 是兼容符号链接**，指向 `datasets/`。历史 `experiment.json`
   中的绝对路径和旧脚本的默认路径依赖 `dataset/CV-Bench`，保留该链接可以
   避免破坏断点续跑与历史配置。新代码请使用 `datasets/`。
2. 历史 `experiment.json` 记录的是**运行时**路径，未因目录重构而改写。
3. 历史 8192 与 Direct Answer 的 Prompt、生成长度、解析规则不同，
   不能作为严格对照组。
4. `reports/` 只放正式报告；`analysis/` 只放核验工具与派生数据；
   `experiments/` 只放原始实验输出。
