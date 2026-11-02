# 模型清单

本项目当前评测两个模型。模型权重保存在 Hugging Face 本地缓存中，**不在仓库内**，
也不需要复制到项目目录。

| 项目 | Xiaomi-Lens-0 | Qwen3.5-4B |
|---|---|---|
| Hugging Face ID | `SeerRay-Lab/Xiaomi-Lens-0` | `Qwen/Qwen3.5-4B` |
| 参数量级 | 4B | 4B |
| Micromamba 环境 | `xiaomi-lens` | `vlm-test` |
| Transformers | 5.4.0 | 5.20.2 |
| 精度 | `bfloat16` | `bfloat16` |
| Attention | `flash_attention_2` | `sdpa` |
| 本地源码 | `models/Xiaomi-Lens-0/` | 无需源码，使用 Transformers 类 |
| 评测适配器 | `evaluation/xiaomi_lens/xiaomi_lens_adapter.py` | `evaluation/qwen35_4b/model_qwen35.py` |
| 本地快照 | `68f57cdd…` | `851bf6e8…` |

## Xiaomi-Lens-0

- 官方仓库源码位于 `models/Xiaomi-Lens-0/`，包含训练代码、示例与文档。
- 评测时通过 `AutoProcessor` 与 `XiaomiLensQwen3_5ForConditionalGeneration` 加载。
- 模型会输出 `<|mask|>` 等特殊 token；CV-Bench 只做文本选择题，不执行 mask 解码。

## Qwen3.5-4B

- 直接使用 Transformers 的 `Qwen3_5ForConditionalGeneration` 与 `AutoProcessor`。
- 本环境未安装 `flash-attn`，使用 `sdpa`；这影响速度，不影响准确率。

## Thinking 控制

两个模型都使用 checkpoint 自带 chat template 的布尔参数：

```text
enable_thinking=False  →  prompt 以 <think>\n\n</think>\n\n 结尾
enable_thinking=True   →  prompt 以 <think>\n 结尾
```

渲染后会校验 prompt 后缀，失败则直接报错，不会把未验证的模式标为成功。

两个模型都**不支持**原生 Adaptive Thinking。
