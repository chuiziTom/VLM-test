# CV-Bench Direct Answer：Xiaomi-Lens-0 vs Qwen3.5-4B 综合对比

- 实验规范名：`direct_answer_think_off_max_new_tokens_64_v1`
- 协议：Thinking Off + Direct Answer（只输出一行 `FINAL_ANSWER: (X)`）
- 数据集：CV-Bench 测试集，2638 条
- 运行时间：2026-11-01
- 结果目录：`experiments/cv_bench/direct_answer_64_v1/`

---

## A. 实验设置对比

| 配置 | Xiaomi-Lens-0 | Qwen3.5-4B |
|---|---|---|
| Python 环境 | `xiaomi-lens` | `vlm-test` |
| 模型快照 | `68f57cdd…` | `851bf6e8…` |
| 模型精度 | `bfloat16` | `bfloat16` |
| Attention | `flash_attention_2` | `sdpa` |
| Transformers | 5.4.0 | 5.20.2 |
| PyTorch | 2.10.0+cu128 | 2.10.0+cu128 |
| Thinking | Off（`enable_thinking=False`） | Off（`enable_thinking=False`） |
| Prompt | `direct_answer_v1` | `direct_answer_v1` |
| Prompt SHA-1 | `89db6880…` | `89db6880…`（相同） |
| Max New Tokens | 64 | 64 |
| do_sample | False（贪心） | False（贪心） |
| Batch Size | 1 | 1 |
| 数据集 | CV-Bench Test（2638） | CV-Bench Test（2638） |
| 评分规则 | 官方统一规则 | 官方统一规则 |
| 解析器 | `direct-1.0`（整段严格匹配） | `direct-1.0`（整段严格匹配） |
| GPU | RTX PRO 6000 Blackwell 94.96 GiB | 同左 |
| GPU 共享 | 是（另有 `graphnav`） | 是（另有 `graphnav`） |

**完全一致的部分**：数据、任务 Prompt、生成长度、贪心解码、解析规则、官方评分公式、
batch size、GPU 型号。

**不一致的部分**：Attention 实现、Transformers 版本、两个模型各自的 tokenizer/processor
与 chat template（这些由模型本身决定，无法也不应强行统一）。

---

## B. 准确率对比

| 指标 | Xiaomi-Lens-0 | Qwen3.5-4B | 差值（Xiaomi − Qwen） |
|---|---:|---:|---:|
| **CV-Bench 官方总分** | **87.04%** | **85.88%** | **+1.16 pp** |
| 全体样本 Accuracy | 86.92%（2293/2638） | 85.63%（2259/2638） | +1.29 pp |
| 2D Accuracy（官方加权） | **82.17%** | **80.26%** | +1.91 pp |
| 3D Accuracy（Omni3D） | **91.92%** | **91.50%** | +0.42 pp |
| Object Counting | 72.46%（571/788） | 69.42%（547/788） | **+3.05 pp** |
| Spatial Relationship | 95.23%（619/650） | 94.46%（614/650） | +0.77 pp |
| Depth Order | 96.17%（577/600） | 94.83%（569/600） | **+1.33 pp** |
| Relative Distance | 87.67%（526/600） | **88.17%**（529/600） | −0.50 pp |
| ADE20K | 77.25%（489/633） | 76.30%（483/633） | +0.95 pp |
| COCO | 87.08%（701/805） | 84.22%（678/805） | **+2.86 pp** |
| Omni3D | 91.92%（1103/1200） | 91.50%（1098/1200） | +0.42 pp |
| Invalid Rate | **0.00%**（0） | 0.91%（24） | −0.91 pp |
| 格式违规率 | **0.00%**（0） | 0.91%（24） | −0.91 pp |
| 截断率 | 0.00% | 0.00% | 0 |
| 推理失败率 | 0.00% | 0.00% | 0 |
| 有效回答率 | **100.00%** | 99.09% | +0.91 pp |

### 逐题对比（同一道题谁答对）

| 类别 | 数量 |
|---|---:|
| 两者都对 | 2182 |
| 两者都错 | 268 |
| 仅 Xiaomi 对 | 111 |
| 仅 Qwen 对 | 77 |
| 净差（Xiaomi − Qwen） | **+34** |
| 精确 McNemar p | **0.0159** |

McNemar 检验表明：在逐题口径下，Xiaomi 的优势在 5% 水平上显著。

### 分任务逐题对比

| 任务 | 仅 Xiaomi 对 | 仅 Qwen 对 | 净差 | McNemar p |
|---|---:|---:|---:|---:|
| Count | 76 | 52 | +24 | **0.0416** |
| Relation | 11 | 6 | +5 | 0.332 |
| Depth | 11 | 3 | +8 | 0.057 |
| Distance | 13 | 16 | −3 | 0.711 |

差距主要来自 **Object Counting**：Xiaomi 净胜 24 题，统计显著。
Depth 的 +8 接近显著；Distance 与 Relation 无显著差异。

### 细分组合

| 组合 | Xiaomi | Qwen | 差值 |
|---|---:|---:|---:|
| 2D/Count/ADE20K | 63.45%（217/342） | 62.28%（213/342） | +1.17 pp |
| 2D/Count/COCO | 79.37%（354/446） | 74.89%（334/446） | **+4.48 pp** |
| 2D/Relation/ADE20K | 93.47%（272/291） | 92.78%（270/291） | +0.69 pp |
| 2D/Relation/COCO | 96.66%（347/359） | 95.82%（344/359） | +0.84 pp |
| 3D/Depth/Omni3D | 96.17%（577/600） | 94.83%（569/600） | +1.33 pp |
| 3D/Distance/Omni3D | 87.67%（526/600） | 88.17%（529/600） | −0.50 pp |

---

## C. 推理性能对比

| 指标 | Xiaomi-Lens-0 | Qwen3.5-4B |
|---|---:|---:|
| 模型加载耗时 | 4.99 s | **3.62 s** |
| 预热耗时 | 1.39 s | **1.23 s** |
| 平均图片解码 | 0.0036 s | **0.0028 s** |
| 平均预处理耗时 | 0.0231 s | **0.0120 s** |
| 平均生成耗时 | **0.2717 s** | 0.3187 s |
| 平均端到端耗时 | **0.2993 s** | 0.3343 s |
| 中位端到端耗时 | **0.2767 s** | 0.3184 s |
| P95 端到端耗时 | 0.4039 s | **0.4013 s** |
| 最长单条 | 0.6482 s | **0.6132 s** |
| 平均生成 Token | **8.00** | 9.98 |
| 中位生成 Token | 8 | 10 |
| P95 生成 Token | 8 | 10 |
| 总生成 Token | **21104** | 26316 |
| Tokens/s（含 Prefill） | 29.45 | **31.30** |
| Samples/s | **3.34** | 2.99 |
| 单样本峰值显存（均值） | 8.81 GiB | **8.66 GiB** |
| 整体峰值 allocated | 9.36 GiB | **9.22 GiB** |
| 总评测墙钟 | **789.7 s** | 882.1 s |

**解读**：

- 两者差距很小（平均端到端 0.299 vs 0.334 s，约 12%），且 P95 几乎相同。
- Qwen 每条多输出 2 个 token（`<|endoftext|>` 等），总 Token 略多。
- Xiaomi 的 Processor 预处理反而更慢（0.023 vs 0.012 s），与 Attention 实现无关。
- **速度差异不能直接归因于模型能力**：Attention 实现、Transformers 版本、
  tokenizer 都不同，且 GPU 全程共享。准确率可以比较，速度只应作为“同机顺序运行
  的参考值”。

---

## D. 结果分析

### 1. 哪个模型整体准确率更高？

**Xiaomi-Lens-0**：官方 87.04% vs 85.88%，高 1.16 pp。
逐题配对检验 p = 0.0159，差异显著。

### 2. 哪个模型 2D 能力更强？

**Xiaomi-Lens-0**：82.17% vs 80.26%，高 1.91 pp。
其中 Count 差距最大（+3.05 pp），Relation 差距较小（+0.77 pp）。

### 3. 哪个模型 3D 能力更强？

**Xiaomi-Lens-0 略强**：91.92% vs 91.50%，仅高 0.42 pp，接近持平。
但分任务看方向相反：

- Depth：Xiaomi 96.17% vs Qwen 94.83%，Xiaomi 高 1.33 pp；
- Distance：Qwen 88.17% vs Xiaomi 87.67%，Qwen 高 0.50 pp。

### 4. 哪些细分任务差距最大？

1. **2D/Count/COCO**：+4.48 pp（Xiaomi 79.37% vs Qwen 74.89%）；
2. **Object Counting 总体**：+3.05 pp；
3. **COCO 总体**：+2.86 pp。

这三个高度相关，说明差距主要集中在**基于 COCO 图像的物体计数**。

### 5. Direct Answer 是否减少了 Token 数和耗时？

**是，而且幅度很大**。与各自历史 Thinking Off（8192，允许分析）相比：

| 模型 | 平均生成 Token | 平均端到端 |
|---|---|---|
| Xiaomi 历史 → Direct | 137.24 → 8.00（−94.2%） | 4.98 s → 0.299 s（−94.0%） |
| Qwen 历史 → Direct | 118.56 → 9.98（−91.6%） | 10.64 s → 0.334 s（−96.9%） |

> 注意：两次实验的 Prompt、`max_new_tokens` 与解析规则均不同，
> 上述比例是“整体配置变化”的效果，不能全部归因于关闭显式 Thinking。

### 6. 哪个模型的无效回答比例更高？

**Qwen3.5-4B**：0.91%（24 条），全部为格式违规，`finish_reason` 均是 `eos`，
不是截断。典型输出是 `(A) 1`（选项字母 + 计数值），缺少 `FINAL_ANSWER` 前缀。

Xiaomi-Lens-0：0 条 Invalid。

按严格协议，24 条全部计错；其中有 17 条的字母本身正确。
若放宽为接受行首 `(X)`，Qwen 官方分约为 86.41%（+0.53 pp）——
但这只是诊断值，**正式成绩仍采用严格口径**。

### 7. 是否存在某些任务对显式 Thinking 更敏感的迹象？

把“历史 Thinking Off（允许分析）”与“本次 Direct Answer（禁止分析）”作为**参考对照**
（非严格实验对照），两个模型呈现相同方向：

| 任务 | Xiaomi 变化 | Qwen 变化 | 一致性 |
|---|---:|---:|---|
| Count | −2.03 pp | −3.17 pp | 都下降 |
| Relation | −0.62 pp | −1.08 pp | 都下降 |
| Distance | −1.67 pp | −1.33 pp | 都下降 |
| Depth | **+1.33 pp** | **+1.83 pp** | 都上升 |

这提示：

- **Count 与 Distance** 可能在需要中间分解、逐个定位时更依赖显式分析过程；
- **Depth** 的直接判断反而更稳，冗长的思考可能引入额外干扰；
- 由于 Prompt 与生成长度同时改变，这只是**一致的趋势信号**，不能当作因果结论。

---

## E. 局限

1. **不是严格的“Thinking On vs Off”对照**：本次只跑 Direct Answer；
   与历史 8192 实验在 Prompt、生成长度、解析规则上都不同。
2. **格式约束本身影响准确率**：Qwen 的 24 条格式违规中 17 条字母正确，
   严格协议下仍计错；Xiaomi 未出现此类问题。
3. **“无显式思考”≠“无推理”**：模型仍可能在内部完成计算。
4. **速度不可直接横向比较**：Attention 实现、Transformers 版本不同，
   GPU 与其他任务共享，绝对耗时只能作参考。
5. **单一数据集**：CV-Bench 只覆盖四类空间任务。
6. **没有跑 Adaptive Thinking**，也没有本次协议下的 Thinking On 对照。

---

## F. 下一步建议

1. **在同一 Direct Answer 协议下补一组 Thinking On**，才能真正分离
   “Thinking 开关”与“Prompt/格式”的影响。
2. **错误案例分析**：重点看 2D/Count/COCO 与 3D/Distance，判断是计数错误、
   类别边界错误还是格式问题。
3. **格式鲁棒性实验**：在 Prompt 中给出 `(A) 1` 这种“选项 + 数值”的合法示例，
   观察 Qwen 的格式违规率是否下降；注意这属于新协议，需重新命名版本。
4. **统一硬件条件的效率复测**：GPU 独占时测 TTFT、Decode 速度与峰值显存。
5. **扩展到 SPAR-Bench、VSI-Bench**，检验结论是否可迁移。

---

## G. 结论

1. **Xiaomi-Lens-0 在 Direct Answer 协议下整体更好**：87.04% vs 85.88%，
   逐题配对检验显著（p = 0.0159）。
2. **优势主要来自 2D 计数**（+3.05 pp）和 **COCO 来源**（+2.86 pp）；
   Depth 也有 1.33 pp 优势，Distance 则 Qwen 略高 0.50 pp。
3. **两模型的格式稳定性差异明显**：Xiaomi 0 条 Invalid，Qwen 24 条（0.91%），
   后者全部是“选项 + 计数”这种非标准输出。
4. **直接回答的收益主要是效率**：两个模型都只需约 8–10 个 token，
   单条端到端约 0.30–0.33 s；相比历史 8192 实验，耗时下降 94%–97%。
5. **代价是少量准确率**：Xiaomi −0.72 pp、Qwen −0.88 pp（相对历史 Thinking Off），
   其中 Count 与 Distance 下降、Depth 上升，两个模型方向一致。
6. **最重要的发现**：在强制直接回答时，两个模型的差距主要不在 3D 空间判断，
   而在 **2D 物体计数的稳定性**；同时，严格的输出格式要求会把 Qwen 的
   “答对但格式不对”转化为真实失分。

---

## 附录：核验与复现

独立核验脚本：`analysis/direct_answer_64_v1/verify_and_compare.py`

核验结果：

| 项目 | Xiaomi | Qwen |
|---|---|---|
| 记录数 / 唯一 uid | 2638 / 2638 | 2638 / 2638 |
| 缺失 / 重复 | 0 / 0 | 0 / 0 |
| 重新解析不一致 | 0 | 0 |
| 指标复算差值 | 0.000000 | 0.000000 |

复现命令：

```bash
# 单元测试
micromamba run -n vlm-test python -B evaluation/direct_answer/tests/test_direct_parser.py

# 校验
micromamba run -n xiaomi-lens python -B evaluation/direct_answer/run_xiaomi_direct_answer.py --validate
micromamba run -n vlm-test python -B evaluation/direct_answer/run_qwen35_direct_answer.py --validate

# 核验与对比
micromamba run -n vlm-test python -B analysis/direct_answer_64_v1/verify_and_compare.py
```
