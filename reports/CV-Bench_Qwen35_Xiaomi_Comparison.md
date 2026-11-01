# CV-Bench 评测报告：Qwen3.5-4B 与 Xiaomi-Lens-0（Thinking Off / On）

生成日期：2026-11-01
数据来源：本仓库已有的全量预测文件，未重新运行任何模型推理。

---

## 0. 核心结论（先看这里）

1. **四种运行全部已完成**：Qwen3.5-4B Off / On、Xiaomi-Lens-0 Off / On 均为 **2638 / 2638** 样本，无缺失、无重复、无坏行，`status: completed`、`aborted: false`。
2. **无需续跑**：Qwen Thinking On 已于 2026-10-31 01:48 正常跑完（此前会话中断时后台进程已自然结束）。
3. **指标完全可复现**：从 `raw_generation` 重新解析并复算后，四种运行的官方总分与原 `metrics.json` **逐项一致（差值为 0）**。
4. **官方 CV-Bench 总分**：

   | 模型 | Thinking Off | Thinking On | 变化 |
   |---|---:|---:|---:|
   | Xiaomi-Lens-0 | **87.76%** | **87.29%** | −0.48 pp |
   | Qwen3.5-4B | **86.76%** | **85.34%** | −1.42 pp |

5. **两个模型开启 Thinking 后总分都下降**，但原因不同：
   - Qwen 的 172 条（6.52%）输出在 `max_new_tokens=8192` 内**没有写出 `</think>` 和最终答案**，按协议全部计入错误；
   - Xiaomi 的 Thinking 更“克制”（平均 302 个思考 token），分数只轻微下降。
6. **若只看能正常给出答案的样本**，Qwen On 的准确率反而最高（91.08%），说明其问题主要是**稳定性/收敛性**，而不是空间推理能力不足。
7. **Xiaomi-Lens-0 总体略优**，但在配对检验下，**Off 模式两者的差距不显著**（p = 0.084）；**On 模式下 Xiaomi 显著更好**（p = 0.001）。
8. **耗时/吞吐/显存不可直接横向比较**：两个模型使用不同 Attention 实现，且本机 GPU 全程与其他任务共享。

---

## 1. 实验目的

CV-Bench 用真实场景图片考察视觉语言模型的**空间理解**能力，分为：

- **2D**：物体计数（Count）、二维空间关系（Relation）；
- **3D**：深度顺序（Depth）、相对距离（Distance）。

本实验比较两个 4B 级视觉语言模型：

- **Qwen3.5-4B**：通用 VLM，作为“通用能力”基线；
- **Xiaomi-Lens-0**：面向视觉/空间任务专门化的模型。

同时对每个模型比较 **Thinking Off / On**：关闭思考时模型直接作答；开启思考时先在 `<think>` 内推理，再给出 `FINAL_ANSWER`。目的是回答两个问题：

1. 专门化模型在空间理解上是否真的优于通用模型？
2. 增加推理 token 能否换来准确率提升，成本是否值得？

---

## 2. 测试环境

| 项目 | 值 |
|---|---|
| 操作系统 | Linux 7.0.0-38-generic（x86_64, glibc 2.39） |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU 显存 | 94.96 GiB（97887 MiB） |
| NVIDIA Driver | 580.178.04（驱动侧 CUDA 13.0） |
| CUDA Runtime | 12.8（PyTorch `2.10.0+cu128`） |
| Python | 3.11.18（两个环境相同） |
| PyTorch | 2.10.0+cu128（两个环境相同） |
| Transformers | Qwen：**5.20.2**（`vlm-test`）；Xiaomi：**5.4.0**（`xiaomi-lens`） |
| Micromamba 环境 | Qwen：`vlm-test`；Xiaomi：`xiaomi-lens` |
| 模型精度 | 两个模型均为 `bfloat16` |
| Attention 实现 | Qwen：`sdpa`；Xiaomi：`flash_attention_2` |
| 生成设置 | `do_sample=False`（贪心）、`max_new_tokens=8192`、`seed=0` |
| 模型权重 | Qwen `851bf6e8…`；Xiaomi `68f57cdd…`（均为本地缓存快照） |

**GPU 共享情况**：Qwen Off 运行时 `gpu_shared=true`（同时有其他 Python 进程占用约 10 GiB）；Xiaomi 运行时整卡已占用约 89.8 GiB。因此本报告中所有**绝对耗时只能作参考**。

---

## 3. 数据集

本地路径：`dataset/CV-Bench/`，只使用测试集 `test_2d.parquet` 与 `test_3d.parquet`。四种运行的 `experiment.json` 记录的文件大小与修改时间完全一致，说明**使用的是同一份数据**。

| type | task | source | 数量 |
|---|---|---|---:|
| 2D | Count | ADE20K | 342 |
| 2D | Count | COCO | 446 |
| 2D | Relation | ADE20K | 291 |
| 2D | Relation | COCO | 359 |
| 3D | Depth | Omni3D | 600 |
| 3D | Distance | Omni3D | 600 |
| | | **合计** | **2638** |

- 2D 共 1438 条，3D 共 1200 条。
- 选项数不固定：3D 全为 2 选项；2D 为 2–6 选项，共 1850 / 493 / 156 / 139 条。
- 样本 uid 形如 `2D:1234`，四份预测文件使用同一套 uid，可逐题对齐。

---

## 4. 测试流程

```text
读取 CV-Bench 测试样本（2638 条）
        ↓
从 parquet 懒加载图片，读取问题与选项
        ↓
instruction = 官方 prompt + 统一的 FINAL_ANSWER 格式要求
        ↓
apply_chat_template（enable_thinking = False / True）
        ↓
校验渲染后的 prompt 后缀是否与请求模式一致
        ↓
模型贪心生成（max_new_tokens = 8192）
        ↓
按 </think> 切出“思考区 / 最终回答区”
        ↓
只在最终回答区逐行严格匹配 FINAL_ANSWER: (X)
        ↓
与 Ground Truth 比较，无效答案按错误计入分母
        ↓
记录耗时、Token、显存，按 type/task/source 汇总
        ↓
写入 predictions.jsonl / metrics.json / summary.csv
```

模型只加载一次；Thinking Off 与 On 使用同一份权重、同一张图片、同一条指令，仅 chat template 的布尔开关不同。

---

## 5. 评测方法

### 5.1 输入构造

图片以 PIL 形式进入 chat template，展开为 `<|vision_start|><|image_pad|><|vision_end|>`，再由 Processor 切成图像 patch 并归一化。文本部分为数据集自带的 `prompt`（问题 + 选项）加上统一的格式指令：

```text
After answering the question, put the final choice on a separate line using exactly this format:
FINAL_ANSWER: (X)
where X is one of the provided option letters.
```

两个模型的该指令 `sha1` 完全相同（`cd11a2a3…`），`instruction = record.prompt + FORMAT_INSTRUCTION` 的拼接代码也一致。

### 5.2 Thinking 控制

两个模型都使用 checkpoint 自带 chat template 的 `enable_thinking` 布尔参数：

| 模式 | 渲染后的 prompt 结尾 |
|---|---|
| Off | `<think>\n\n</think>\n\n` |
| On | `<think>\n` |

渲染后会检查 prompt 后缀（`mode_verified`），防止模板变化导致“开关看起来改了、实际没改”。两个模型均**不支持** Adaptive Thinking，因此不提供第三组结果。

### 5.3 答案解析（严格匹配）

1. 在第一个 `</think>` 处切分，**只在最终回答区解析**；
2. 逐行匹配整行 `FINAL_ANSWER: (X)`；
3. 字母必须属于本题真实存在的选项集合（A–F 动态确定）；
4. 出现两个不同选项 → `conflicting_final_answers`；
5. 解析失败 → `invalid`，**仍然计入分母，按答错处理**。

不搜索全文中的零散字母，不做选项文字模糊匹配，不从思考过程中猜答案，也不根据标准答案反推。

### 5.4 Invalid、截断与失败

| 情况 | 处理 |
|---|---|
| 格式错误（如加粗、小写、缺括号） | 记 `invalid`，算错 |
| 达到 8192 token 上限仍未给出答案 | 记 `truncated` + `invalid`，算错 |
| 推理异常 | 记 `error`，算错 |

**无效答案、失败、截断都不从分母中删除**，这是本评测与“只统计有效回答”的根本区别。

### 5.5 官方加权分数

CV-Bench 官方不是全样本简单平均，而是按数据来源先算再平均：

```text
Accuracy_2D = (Accuracy_ADE20K + Accuracy_COCO) / 2
Accuracy_3D = Accuracy_Omni3D
CV-Bench Accuracy = (Accuracy_2D + Accuracy_3D) / 2
```

报告中同时给出“全部样本普通 Accuracy”。两者不同是因为三个来源样本量不等（633 / 805 / 1200）。**对外比较应使用官方加权总分。**

---

## 6. 结果完整性核验

### 6.1 检查方法

本次没有重新运行模型，而是用独立脚本 `reports/analysis/verify_results.py` 做只读核验：

1. 用 CV-Bench 的 2638 个 uid 检查覆盖、重复与缺失；
2. 用原始 `raw_generation` 重新执行严格解析，与已保存的 `prediction` 对比；
3. 用逐样本记录重新计算官方加权分数，与 `metrics.json` 对比。

### 6.2 核验结果

| 运行 | 记录数 | 唯一 uid | 缺失 | 重复 | 元数据不一致 | 解析不一致 | 指标不一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen Off | 2638 | 2638 | 0 | 0 | 0 | 0 | 0 |
| Qwen On | 2638 | 2638 | 0 | 0 | 0 | 0 | 0 |
| Xiaomi Off | 2638 | 2638 | 0 | 0 | 0 | 0 | 0 |
| Xiaomi On | 2638 | 2638 | 0 | 0 | 0 | 0 | 0 |

| 运行 | 原 metrics.json 官方总分 | 独立复算 | 差值 |
|---|---:|---:|---:|
| Qwen Off | 86.7590% | 86.7590% | 0.000000 |
| Qwen On | 85.3367% | 85.3367% | 0.000000 |
| Xiaomi Off | 87.7626% | 87.7626% | 0.000000 |
| Xiaomi On | 87.2871% | 87.2871% | 0.000000 |

另核对：评测共用的 `answer_parser.py`、`cvbench_dataset.py`、`metrics.py`、`test_answer_parser.py` 在两个模型目录中 **SHA-1 完全相同**，因此评分协议确实同源。

**结论：四种运行均完整、可复现，无需续跑，未修改任何原始文件。**

---

## 7. Qwen3.5-4B：Thinking Off vs On

### 7.1 总览

| 指标 | Thinking Off | Thinking On | 变化 |
|---|---:|---:|---:|
| **官方 CV-Bench 总分** | **86.76%** | **85.34%** | **−1.42 pp** |
| 全部样本普通 Accuracy | 86.73% | 85.14% | −1.59 pp |
| 2D Accuracy（官方加权） | 82.27% | 79.09% | −3.18 pp |
| 3D Accuracy（Omni3D） | 91.25% | 91.58% | +0.33 pp |
| ADE20K Accuracy | 76.46%（484/633） | 73.46%（465/633） | −3.00 pp |
| COCO Accuracy | 88.07%（709/805） | 84.72%（682/805） | −3.35 pp |
| Omni3D Accuracy | 91.25%（1095/1200） | 91.58%（1099/1200） | +0.33 pp |
| Invalid Rate | 0.04%（1） | **6.52%（172）** | +6.48 pp |
| 截断率 | 0%（0） | 6.52%（172） | +6.52 pp |
| 平均生成 Token | 118.56 | 1365.24 | 11.51× |
| 平均 Thinking Token | 0 | 1301.77 | — |
| 平均端到端耗时 | 10.64 s | 37.13 s | 3.49× |
| P95 端到端耗时 | 35.71 s | 214.31 s | 6.00× |
| Tokens/s（含 Prefill） | 11.22 | 36.79 | +25.57 |
| 单样本峰值显存（均值/最大） | 8.66 / 9.22 GiB | 8.67 / 9.22 GiB | ≈ 0 |
| 总墙钟时间 | 28070 s（7.8 h） | 97942 s（27.2 h） | 3.49× |

### 7.2 细分任务

| 任务 | Off | On | 变化 |
|---|---:|---:|---:|
| 物体计数 Count | 572/788（72.59%） | 524/788（66.50%） | −6.09 pp |
| 二维空间关系 Relation | 621/650（95.54%） | 623/650（95.85%） | +0.31 pp |
| 深度顺序 Depth | 558/600（93.00%） | 559/600（93.17%） | +0.17 pp |
| 相对距离 Distance | 537/600（89.50%） | 540/600（90.00%） | +0.50 pp |

| 组合 | Off | On | 变化 |
|---|---:|---:|---:|
| 2D/Count/ADE20K | 210/342（61.40%） | 188/342（54.97%） | −6.43 pp |
| 2D/Count/COCO | 362/446（81.17%） | 336/446（75.34%） | −5.83 pp |
| 2D/Relation/ADE20K | 274/291（94.16%） | 277/291（95.19%） | +1.03 pp |
| 2D/Relation/COCO | 347/359（96.66%） | 346/359（96.38%） | −0.28 pp |
| 3D/Depth/Omni3D | 558/600（93.00%） | 559/600（93.17%） | +0.17 pp |
| 3D/Distance/Omni3D | 537/600（89.50%） | 540/600（90.00%） | +0.50 pp |

**关键事实：Qwen On 的失分几乎全部集中在 2D 计数任务**，3D 与非计数类 2D 任务反而略有提升。

### 7.3 逐样本差异

| 类别 | 数量 |
|---|---:|
| 两种模式都对 | 2137 |
| 两种模式都错 | 241 |
| Off 对、On 错 | 151 |
| Off 错、On 对 | 109 |
| 预测发生变化 | 367 |

净损失 42 题。McNemar 精确检验：**p = 0.011**，即 Qwen 关闭 Thinking 在普通逐题正确率上显著更好。

### 7.4 为什么下降：172 条截断

- 172 条无效样本全部是 `truncated_before_final_answer`：生成到 8192 token 仍未写出 `</think>` 和 `FINAL_ANSWER`。
- 分布：Count 118、Relation 14、Depth 19、Distance 21；按来源 ADE20K 79、COCO 53、Omni3D 40。
- 其中 **85 条在 Off 模式下原本答对**。
- 非截断样本的思考 Token 平均 821，而截断样本直接用满 8192。

**剔除无效答案后的诊断口径**（仅作分析，不是官方成绩）：

| 口径 | Off | On |
|---|---:|---:|
| 只统计有效回答的普通 Accuracy | 86.77% | **91.08%** |
| 只统计有效回答的官方加权分（诊断） | — | 2D 87.31% / 3D 94.74% / 合计 91.03% |

这说明 Qwen 的 Thinking **本身是有效的**，但它在这套题上容易陷入过长推理、无法收敛，最终因格式约束未满足而丢掉 6.5% 的样本。这是**模型行为 + 8192 上限**共同作用的结果，不是解析器误判。

---

## 8. Xiaomi-Lens-0 与 Qwen3.5-4B 横向对比

### 8.1 主对比表

| 指标 | Xiaomi Off | Xiaomi On | Qwen Off | Qwen On |
|---|---:|---:|---:|---:|
| **官方 CV-Bench 总分** | **87.76%** | **87.29%** | **86.76%** | **85.34%** |
| 全部样本普通 Accuracy | 87.76% | 87.23% | 86.73% | 85.14% |
| 2D Accuracy（官方加权） | 83.44% | 83.41% | 82.27% | 79.09% |
| 3D Accuracy（Omni3D） | 92.08% | 91.17% | 91.25% | 91.58% |
| 物体计数 Count | 74.49% | 73.86% | 72.59% | 66.50% |
| 二维空间关系 Relation | 95.85% | 96.15% | 95.54% | 95.85% |
| 深度顺序 Depth | 94.83% | 94.17% | 93.00% | 93.17% |
| 相对距离 Distance | 89.33% | 88.17% | 89.50% | 90.00% |
| ADE20K | 77.57% | 78.99% | 76.46% | 73.46% |
| COCO | 89.32% | 87.83% | 88.07% | 84.72% |
| Omni3D | 92.08% | 91.17% | 91.25% | 91.58% |
| Invalid Rate | 0.34% | 0.99% | 0.04% | 6.52% |
| 截断数 | 7 | 11 | 0 | 172 |
| 平均生成 Token | 137.24 | 417.44 | 118.56 | 1365.24 |
| 平均 Thinking Token | 0 | 302.26 | 0 | 1301.77 |
| 平均端到端耗时 | 4.98 s | 19.58 s | 10.64 s | 37.13 s |
| P95 端到端耗时 | 12.42 s | 65.45 s | 35.71 s | 214.31 s |
| Tokens/s（含 Prefill） | 27.80 | 21.39 | 11.22 | 36.79 |
| 单样本峰值显存（均值） | 8.81 GiB | 8.81 GiB | 8.66 GiB | 8.67 GiB |
| 全局峰值显存（最大） | 9.36 GiB | 9.36 GiB | 9.22 GiB | 9.22 GiB |
| 总墙钟时间 | 3.62 h | 14.35 h | 7.80 h | 27.21 h |

### 8.2 逐题胜负与显著性

| 对比 A vs B | 仅 A 对 | 仅 B 对 | 净差 | 精确 McNemar p |
|---|---:|---:|---:|---:|
| Qwen Off vs Qwen On | 151 | 109 | +42（Off 优） | **0.011** |
| Xiaomi Off vs Xiaomi On | 95 | 81 | +14（Off 优） | 0.327 |
| Qwen Off vs Xiaomi Off | 100 | 127 | −27（Xiaomi 优） | 0.084 |
| Qwen On vs Xiaomi On | 108 | 163 | −55（Xiaomi 优） | **0.001** |
| Qwen Off vs Xiaomi On | 110 | 123 | −13（Xiaomi 优） | 0.432 |
| Qwen On vs Xiaomi Off | 103 | 172 | −69（Xiaomi 优） | **0.00004** |

说明：McNemar 检验针对**逐题普通正确率**，不是官方加权分；它回答的是“同一道题上谁更可能答对”。

### 8.3 结论性分析

1. **总体**：Xiaomi-Lens-0 的官方总分（87.76% / 87.29%）在两个 Thinking 模式下都高于 Qwen3.5-4B（86.76% / 85.34%）。
   但 Off 模式下差距仅约 1.0 pp，逐题检验 **不显著（p = 0.084）**；On 模式下差距扩大到约 2.0 pp 且**显著（p = 0.001）**。
2. **2D**：Xiaomi 略强（83.44% vs 82.27%）。差异主要来自 **Count**（74.49% vs 72.59%），Relation 两者几乎持平（95.85% vs 95.54%）。
3. **3D**：Xiaomi Off 更强（92.08% vs 91.25%），优势在 **Depth**（94.83% vs 93.00%）；**Distance** 上 Qwen 反而略好（89.50% vs 89.33%）。
4. **Thinking 的影响不同**：
   - Xiaomi：几乎无收益也无明显损失（官方 −0.48 pp，逐题检验不显著），思考更像“温和复核”；
   - Qwen：官方 −1.42 pp 且显著，主要因为 **6.52% 的样本无法在 8192 token 内收敛**；但只看有效回答时准确率最高（91.08%）。
5. **成本**：
   - Xiaomi On：平均 417 token、19.6 s，约为 Off 的 3.0× Token、3.9× 时间；
   - Qwen On：平均 1365 token、37.1 s，约为 Off 的 11.5× Token、3.5× 时间，P95 达到 214 s。
   - 在当前设置下，**两个模型的 Thinking 都没有带来官方分数提升**，属于“付出成本但未获收益”。
6. **最值得注意的单一现象**：Qwen 的失败不是“不会做”，而是“想太久不给答案”。172 条截断中有 85 条在原 Off 模式下是答对的。

### 8.4 典型样例

**样例 1：Thinking 误导计数（uid `2D:5`，Count / ADE20K）**

```text
问题：How many beds are in the image?  选项：2 / 1 / 0 / 3
Ground Truth：B（1）
Qwen Off    ：B（1）✅  “a single bed … there is also a white crib”
Qwen On     ：A（2）❌  “two beds visible: a white crib on the left and a larger bed”
Xiaomi Off  ：A（2）❌
Xiaomi On   ：A（2）❌
```

分析：开启思考后，模型把婴儿床也归类为 bed，属于**过度细化导致的类别边界错误**；Xiaomi 也存在同样偏好。

**样例 2：思考修正计数（uid `2D:15`，Count / ADE20K）**

```text
问题：How many candle holders are in the image?  选项：1/2/3/0/4
Ground Truth：B（2）
Qwen Off    ：C（3）❌
Qwen On     ：B（2）✅
Xiaomi Off  ：B（2）✅
```

分析：Qwen 在思考中逐个定位了侧柜上的两个烛台，修正了关闭思考时的多计，说明 **Thinking 确实能改善细粒度计数**。

**样例 3：Thinking 无法收敛（uid `2D:25`，Count / ADE20K）**

```text
问题：How many folders are in the image?  选项：0/2/3/1
Ground Truth：D（1）
Qwen Off    ：D（1）✅（89 token）
Qwen On     ：无最终答案（8192 token，truncated_before_final_answer）❌
```

分析：模型在思考区反复扫描桌面物品，始终没有写出 `</think>`。这是 Qwen On 172 条截断的典型形态。

**样例 4：Thinking 过度计数（uid `2D:23`，Count / ADE20K）**

```text
问题：How many curtains are in the image?  选项：1/3/2/0/4
Ground Truth：C（2）
Qwen Off    ：C（2）✅（61 token）
Qwen On     ：B（3）❌（3031 token）
Xiaomi Off  ：C（2）✅
Xiaomi On   ：C（2）✅（322 token）
```

分析：思考把“一层纱帘 + 一层布帘 + 窗框遮挡”拆成 3 个对象；Xiaomi 的思考更短，保持了“两个帘片”的正确划分。这体现了两者 Thinking 策略的差异。

---

## 9. 局限与不可直接比较之处

1. **推理速度不可横向比较**：Qwen 使用 `sdpa`，Xiaomi 使用 `flash_attention_2`；且 GPU 在本机长期被其他任务共享（Qwen Off 运行时明确标记 `gpu_shared=true`）。表中的耗时只反映“当时那台机器、那个时刻”的表现，不是模型的固有性能。
2. **Transformers 版本不同**（5.20.2 vs 5.4.0）。数据加载、解析、指标代码相同，但底层推理实现和 chat template 处理可能带来细微差异。
3. **Thinking 模式并非完全等价**：虽然都用 `enable_thinking` 布尔开关，但两个模型生成思考的习惯差异很大（平均 302 vs 1302 个思考 token），Qwen 会大量截断。这既可能是模型差异，也受 `max_new_tokens=8192` 上限影响。
4. **Invalid 与截断进入分母**，会压低官方分数，尤其是 Qwen On。报告同时给出了“只统计有效回答”的诊断值，但它**不能替代官方成绩**。
5. **答案解析严格但机械**：格式稍有偏差（加粗、小写、缺括号）即判无效。该规则对两个模型完全一致，不存在偏袒；但它衡量的是“准确率 + 格式遵守度”的合成结果。
6. **图像预处理由各自 Processor 完成**，分辨率策略可能不同；虽然输入都是同一张图，但进入模型的视觉 token 不一定相同。
7. **模型差异的归因**：Qwen3.5-4B 是通用模型，Xiaomi-Lens-0 是空间任务专门化模型，架构与训练数据不同。观察到的差距**不能**单独归因于任一因素。
8. **CV-Bench 只覆盖四类任务**，不能代表完整的空间理解能力（例如遮挡推理、多步导航、三维重建）。
9. 本机 GPU 上还运行着无关的 `graphnav` 任务（约 12.9 GB），本次核验**未触碰**任何其他进程。

---

## 10. 下一步建议

1. **针对 Qwen 截断做专项分析**：统计 172 条截断的思考循环模式；尝试 `max_new_tokens=16384` 或思考超时/强制收尾，仅用于诊断，不修改原始结果。
2. **统一硬件条件的效率复测**：在 GPU 独占时，用同一批 200–500 条样本分别测两个模型的 TTFT、Decode 速度与峰值显存。
3. **Prompt 敏感性实验**：同一数据集换 2–3 种格式指令（例如把 `FINAL_ANSWER` 放到最前、使用 JSON 输出），检验格式约束对两个模型的影响是否一致。
4. **细分任务专项评测**：Count 是唯一明显分化的任务，建议单独扩大样本或增加变体（小物体、遮挡、同类多实例）。
5. **扩展三维空间评测**：增加 SPAR-Bench、VSI-Bench 等，检验 Depth/Distance 上的结论是否可迁移。
6. **Comparison 的统计口径**：后续报告建议固定同时给出“官方加权分 + 配对 McNemar 检验”，避免把 1 pp 左右的差异直接解读为能力差距。

---

## 11. 最终结论

1. **Xiaomi-Lens-0 总体表现更好**：Off 87.76%（Qwen 86.76%），On 87.29%（Qwen 85.34%）。但 Off 的差距在逐题检验下不显著（p = 0.084），On 的差距显著（p = 0.001）。
2. **差异最明显的任务**：2D 物体计数（Xiaomi 74.49% vs Qwen 72.59%）与 3D 深度顺序（94.83% vs 93.00%）；相对距离上 Qwen 略优（89.50% vs 89.33%）。
3. **Thinking 没有带来官方分数收益**：Xiaomi −0.48 pp（不显著），Qwen −1.42 pp（显著），且 Qwen 出现 6.52% 的截断。
4. **但 Qwen 的 Thinking 本身是有效的**：只统计有效回答时其准确率达 91.08%，为四种运行中最高；问题是稳定性与收敛性。
5. **性能成本**：Qwen On 的平均 Token 为 Off 的 11.5×、平均耗时 3.5×、P95 6.0×；Xiaomi On 为 3.0× Token、3.9× 耗时。在相同的 8192 token 预算下，Xiaomi 的思考更“经济”。
6. **最重要的发现**：在这套评测里，**“能否稳定输出可解析的最终答案”比“是否开启思考”更能决定最终成绩**。Qwen 的 6.52% 截断比其他所有准确率差异都更值得关注。

---

## 附录 A：本次新增文件

```text
reports/
├── CV-Bench_Qwen35_Xiaomi_Comparison.md   # 本报告
└── analysis/
    ├── verify_results.py                  # 独立只读核验脚本
    └── verification.json                  # 核验与复算的完整数值结果
```

**未修改**任何原有脚本、预测文件、指标文件、模型权重、数据集或环境配置。

## 附录 B：复现命令

```bash
cd /home/20261024_1/Workspace/qyh/VLM-test

# 独立核验（只读，不写入原始结果目录）
micromamba run -n vlm-test python -B reports/analysis/verify_results.py
```

脚本会重新读取四份 `predictions.jsonl`，从 `raw_generation` 重新解析并复算，输出：

- 覆盖/重复/缺失检查；
- 解析一致性检查；
- 官方加权分复算与差值；
- 四种运行的汇总表、逐题胜负矩阵与 McNemar 输入。
