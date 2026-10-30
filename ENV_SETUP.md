# VLM-test 环境配置说明

本文档对应本机实际安装并验证通过的环境，目标是用 Micromamba 运行 Qwen3.5 系列
Transformers 推理代码，并具备 CV-Bench / SPAR-Bench / VSI-Bench 的数据处理能力。

- 配置日期：2026-10-28
- 环境名：`vlm-test`
- 环境路径：`/home/20261024_1/.local/share/mamba/envs/vlm-test`

---

## 1. 本机基线

| 项目 | 实际值 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.5 LTS |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| 显存 | 97887 MiB（约 95.6 GiB） |
| 计算能力 | `(12, 0)`，即 sm_120 |
| NVIDIA 驱动 | 580.178.04（驱动支持到 CUDA 13.0） |
| 系统 nvcc | CUDA 12.0（最高只支持 `compute_90`） |
| Micromamba | 2.9.0 |

关键结论：

1. Blackwell（sm_120）必须使用 CUDA 12.8 或更高版本编译/提供的 PyTorch Wheel。
   系统自带的 nvcc 12.0 **不能**编译 sm_120 扩展，不要用它编译 `causal-conv1d`
   或 `flash-attn`。
2. `nvidia-smi` 显示的 “CUDA Version: 13.0” 是驱动支持级别，不代表安装了
   CUDA 13.0 Toolkit。本环境没有安装或修改系统级 CUDA Toolkit。
3. 本项目 Notebook 明确固定 `transformers==5.20.2`，并使用
   `AutoModelForMultimodalLM` 加载 `Qwen/Qwen3.5-4B`。该版本已包含
   `qwen3_5` 模型实现和该 Auto 类，因此未升级到 5.21.0。

### 已安装版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.18 |
| PyTorch | 2.10.0+cu128 |
| TorchVision | 0.25.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| Triton | 3.6.0 |
| Transformers | 5.20.2 |
| Accelerate | 1.15.0 |
| flash-linear-attention | 0.6.0（fla-core 0.6.0） |
| causal-conv1d | 1.7.0（预编译 Wheel，含 sm_120） |
| flash-attn | 未安装（原因见第 3.3 节） |
| pip check | `No broken requirements found.` |

---

## 2. 从零创建环境

### 2.1 创建 Conda 层

```bash
micromamba create -n vlm-test -f environment-vlm-test.yml
```

`environment-vlm-test.yml` 只包含 Python、pip、ipykernel；Python 包由
`requirements-vlm-test.txt` 负责，原因见第 7 节。

### 2.2 安装 PyTorch（CUDA 12.8）

```bash
micromamba run -n vlm-test python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

选择 `torch 2.10.0 + cu128` 的原因：

- `causal-conv1d 1.7.0` 提供了 `cp311 / Linux x86_64 / torch2.10 / cu12`
  的预编译 Wheel，且 Wheel 内包含 `sm_120` 内核，可以直接匹配本机。
- 更高版本 torch（2.11/2.12/2.13/2.14）目前没有对应的 causal-conv1d
  预编译 Wheel，需要源码编译，风险和工作量都更高。
- 驱动 580 向后兼容 CUDA 12.8 Runtime，不需要升级驱动。

### 2.3 安装其余依赖

```bash
micromamba run -n vlm-test python -m pip install \
  -r requirements-vlm-test.txt
```

文件开头的 `--extra-index-url https://download.pytorch.org/whl/cu128` 用于让
`torch==2.10.0+cu128` 和 `torchvision==0.25.0+cu128` 从官方 CUDA 12.8 索引下载。

### 2.4 注册 Jupyter Kernel

```bash
micromamba run -n vlm-test python -m ipykernel install \
  --user --name vlm-test --display-name "Python (vlm-test)"
```

生成的 kernel 文件位于
`~/.local/share/jupyter/kernels/vlm-test/kernel.json`。

### 2.5 使用环境

交互式 shell 如果已经初始化过 micromamba：

```bash
micromamba activate vlm-test
python --version
```

非交互式场景（脚本、CI、当前 shell 未初始化 hook）请使用：

```bash
micromamba run -n vlm-test python your_script.py
```

---

## 3. GPU 加速组件

Qwen3.5 是混合架构：每 4 层由 3 层 Gated DeltaNet 线性注意力 +
1 层 Gated Attention 组成。对应两条优化路径：

- 线性注意力（Gated DeltaNet）→ `flash-linear-attention`（FLA）
- 因果卷积（`linear_conv_kernel_dim=4`）→ `causal-conv1d`
- 全注意力层（full attention）→ PyTorch SDPA，默认会使用 Flash 后端

### 3.1 flash-linear-attention

```bash
micromamba run -n vlm-test python -m pip install flash-linear-attention==0.6.0
```

- 包本身是纯 Python，实际 Kernel 由 `fla-core` 通过 Triton 在运行时编译，
  不需要 nvcc，也不需要源码编译 PyTorch 扩展。
- Transformers 5.20.2 的 Qwen3.5 实现会自动导入 `fla`，绑定：
  - `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`
  - `fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule`
- 已在本机 sm_120 上实际运行验证通过。

### 3.2 causal-conv1d

本机使用 GitHub Release 预编译 Wheel：

```text
https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/
causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

该 Wheel：

- 对应 CPython 3.11、Linux x86_64、torch 2.10、CUDA 12.x；
- 已确认包含 `sm_120` / `compute_120` 代码；
- 安装后 Transformers 会自动绑定
  `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn` 和
  `causal_conv1d_update`。

如果以后更换 torch 版本，这个 Wheel 不能复用。需要源码编译时，**不要使用系统
nvcc 12.0**，应在环境内安装 CUDA 12.8+ 工具链再编译，例如：

```bash
micromamba install -n vlm-test -c nvidia cuda-toolkit=12.8 -y
micromamba run -n vlm-test bash -lc '
  export CUDA_HOME=$CONDA_PREFIX
  export TORCH_CUDA_ARCH_LIST="12.0"
  export MAX_JOBS=8
  python -m pip install causal-conv1d==1.7.0 --no-build-isolation
'
```

以上源码编译路径本次未执行，仅作为换版本时的参考；编译前请先备份可用环境。

### 3.3 flash-attn 为什么没有安装

- PyPI 只提供源码包，没有 `cp311 / torch2.10 / cu12` 的预编译 Wheel；
- GitHub Release 的预编译 Wheel 最高只到 torch 2.8；
- 源码编译需要 CUDA 12.8+ 工具链，耗时长，而且会引入一套重型构建依赖；
- 更重要的是：本项目默认使用 Transformers 的 SDPA 路径，已实际验证
  PyTorch `SDPABackend.FLASH_ATTENTION` 在 sm_120 上可用。Qwen3.5 的
  全注意力层通过 SDPA 已经可以走到 Flash 内核，不安装独立 `flash-attn`
  不会导致默认推理回退到纯数学实现。

因此本环境没有为了 `flash-attn` 强行安装系统 CUDA Toolkit 或源码编译。
如果后续确实需要指定 `attn_implementation="flash_attention_2"`，请单独评估
`flash-attn` 与 torch 2.10 / CUDA 12.8 / sm_120 的兼容性，并在安装后重新跑
第 4.4 节的验证。

### 3.4 没有安装的可选组件

- `qwen-vl-utils`：当前 Notebook 直接使用 `AutoProcessor` + PIL，不依赖它；
  以后需要视频抽帧工具时可以再装。
- `lmms-eval`：`00_env_data_check.ipynb` 只检查是否存在，当前代码没有调用；
  它是独立评测框架，可能对 transformers/torch 有额外版本约束，建议单独建环境
  或确认兼容后再装。
- `vLLM`、`SGLang`、`xFormers`：当前项目没有使用，未安装。

---

## 4. 验证过程与结果

以下验证均在本机实际执行。

### 4.1 依赖完整性

```bash
micromamba run -n vlm-test python -m pip check
```

结果：

```text
No broken requirements found.
```

### 4.2 CUDA / BF16 / GPU 矩阵乘

```python
import torch
print(torch.__version__)                 # 2.10.0+cu128
print(torch.version.cuda)                # 12.8
print(torch.cuda.is_available())         # True
print(torch.cuda.get_device_name(0))     # NVIDIA RTX PRO 6000 Blackwell ...
print(torch.cuda.get_device_capability(0))  # (12, 0)
print(torch.cuda.is_bf16_supported())    # True

a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
```

实测结果：

- CUDA 可用，GPU 为 `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`；
- BF16 支持为 `True`；
- 2048×2048 BF16 矩阵乘成功，相对 FP32 参考的最大绝对误差约 `0.5005`
  （BF16 精度下的正常量级）；
- 测试时 GPU 空闲显存约 `75.83 / 94.96 GiB`。

### 4.3 Qwen3.5 模型接口

```python
import torch
import transformers
from transformers import AutoProcessor, AutoModelForMultimodalLM
print(transformers.__version__)  # 5.20.2
```

已实际验证：

- `AutoProcessor` 和 `AutoModelForMultimodalLM` 可导入；
- `AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")` 返回
  `Qwen3VLProcessor`，tokenizer 词表大小 `248044`；
- 用项目中的 `dataset/CV-Bench/samples/2D/2d_01.png` 调用
  `processor.apply_chat_template(..., return_tensors="pt")` 成功，
  得到 `input_ids (1, 99)`、`pixel_values (320, 1536)`、
  `image_grid_thw (1, 3)` 等输入；
- 只下载了 tokenizer / preprocessor / config 等小文件（缓存约 80 KB），
  **没有下载任何 `.safetensors` 权重**；
- 用完整配置在 `meta` device 上构造模型成功：
  `Qwen3_5ForConditionalGeneration`，参数量约 `4.539B`，未加载权重。

为了验证优化 Kernel 真的被调用，使用随机初始化的**微型 Qwen3.5**
（4 层：3 层线性注意力 + 1 层全注意力，约 111 万参数）在 GPU 上执行
前向和 `generate`：

```text
forward logits: (1, 1, 1024)
generate shape: (1, 40)
kernel calls: {'chunk': 6, 'recurrent': 21, 'conv_fn': 6, 'conv_update': 21}
peak GPU mem: 0.26 GiB
```

四个计数都大于 0，说明：

- `chunk_gated_delta_rule`（FLA，prefill）被调用；
- `fused_recurrent_gated_delta_rule`（FLA，decode）被调用；
- `causal_conv1d_fn`（prefill）被调用；
- `causal_conv1d_update`（decode）被调用。

这不是完整的 Qwen3.5-4B 推理；完整推理需要权重，见 4.5 节。

### 4.4 加速组件的实际绑定与微基准

绑定检查（在全新 Python 进程中导入 Transformers 后）：

```text
torch_chunk_gated_delta_rule
  -> fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule
     + transformers...torch_chunk_gated_delta_rule (fallback)
torch_recurrent_gated_delta_rule
  -> fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule
     + transformers...torch_recurrent_gated_delta_rule (fallback)
causal_conv1d_fn
  -> causal_conv1d.causal_conv1d_interface.causal_conv1d_fn
     + transformers...causal_conv1d_fn (fallback)
causal_conv1d_update
  -> causal_conv1d.causal_conv1d_interface.causal_conv1d_update
     + transformers...causal_conv1d_update (fallback)
```

即 Transformers 优先使用优化实现，缺失时才回退到 PyTorch。

数值一致性检查：同一个微型 Qwen3.5 模型、同一输入，分别走“FLA + causal-conv1d”
和“全部回退到 PyTorch”两条路径，比较最终 logits：

```text
logits max_abs_diff:  0.00635
logits mean_abs_diff: 0.001466
allclose(atol=5e-2, rtol=5e-2): True
```

差异属于 BF16 精度下的正常范围，说明优化 Kernel 没有改变模型数值行为。

Kernel 级微基准（同一输入、BF16、先预热再计时、`torch.cuda.synchronize()`）：

| 测试 | 优化实现 | PyTorch 参考 | 加速比 |
| --- | --- | --- | --- |
| `chunk_gated_delta_rule`，B=1,T=1024,H=8,K=V=64 | 0.82 ms | 3.92 ms | 约 4.8× |
| `causal_conv1d`，B=1,D=8192,S=1024,W=4 | 0.09 ms | 0.13 ms | 约 1.5× |

注意：测试时 GPU 还有其它进程占用（约 46% 利用率、19 GB 显存），
以上数字只用于确认优化 Kernel 生效，不代表最终模型端到端性能。

### 4.5 完整模型推理

当前机器没有 `Qwen/Qwen3.5-4B` 的权重缓存，本次**没有下载** 9 GB 级权重，
因此没有做完整单样本推理或端到端性能测试。

权重就绪后，可用如下方式做单样本冒烟测试（与项目 Notebook 一致）：

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "Qwen/Qwen3.5-4B"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto",
).eval()

image = Image.open("dataset/CV-Bench/samples/2D/2d_01.png").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "请简单描述图片中的场景。"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                       skip_special_tokens=True))
```

Notebook 记录该模型显存占用约 21 GB。本机 97 GB 显存足够，但注意当前有其它
训练进程占用。

### 4.6 数据集依赖冒烟测试

未下载任何数据集，只读取项目已有样本：

- `dataset/CV-Bench/samples/2D/annotations.csv`：`(30, 4)`，列为
  `image_file/task/question/answer`；
- `dataset/CV-Bench/samples/2D/2d_01.png`：RGB，`320×240`；
- `dataset/VSI-Bench/test.jsonl`：`(5130, 7)`，pandas 读取正常；
- `pyarrow` 26.0.0 的 Parquet 读写往返成功；
- `av` 18.1.0、`opencv-python-headless` 5.0.0.93 可导入。

---

## 5. IDE 配置

### VS Code

1. `Ctrl+Shift+P` → `Python: Select Interpreter`；
2. 选择：

   ```text
   /home/20261024_1/.local/share/mamba/envs/vlm-test/bin/python
   ```

3. Notebook 右上角 Kernel 选择 `Python (vlm-test)`。

### PyCharm

`Settings → Project → Python Interpreter → Add Interpreter → Existing`，选择：

```text
/home/20261024_1/.local/share/mamba/envs/vlm-test/bin/python
```

### 注意：Notebook 中仍有旧机器路径

当前 Notebook 里的路径是旧机器：

```text
/home/qiuyuhang_26/workspace/VLM-test
```

本机实际路径是：

```text
/home/20261024_1/Workspace/qyh/VLM-test
```

按照任务要求，本次没有改写 Notebook。运行前请把 `PROJECT`、`DATASET_ROOT`、
`IMAGE_PATH` 等变量改成新路径，或者在 Notebook 中统一使用
`pathlib.Path.cwd()` / 环境变量。

---

## 6. 常见问题

### 6.1 `micromamba activate` 提示找不到命令 / 不生效

当前 shell 未初始化 micromamba hook。无需修改 `.zshrc`，直接使用：

```bash
micromamba run -n vlm-test python your_script.py
```

或在当前 shell 中显式执行：

```bash
eval "$(micromamba shell hook --shell zsh)"
micromamba activate vlm-test
```

### 6.2 Hugging Face 下载慢或 Xet 报错

Notebook 已经设置：

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
```

这两行必须在 `import transformers` / `import huggingface_hub` **之前**执行。

### 6.3 SOCKS 代理报错

本机 shell 同时导出了 `ALL_PROXY=socks5h://127.0.0.1:7890`。新版
`huggingface_hub` 使用 `httpx2`，没有 SOCKS 支持时会报：

```text
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

本环境已安装 `socksio==1.0.0`。如果仍然失败，可以只保留 HTTP 代理：

```bash
unset ALL_PROXY all_proxy
```

### 6.4 `causal_conv1d` 导入失败或 `undefined symbol`

优先检查三件事：

```bash
micromamba run -n vlm-test python -c "import torch; print(torch.__version__)"
micromamba run -n vlm-test python -c "import sys; print(sys.version)"
micromamba run -n vlm-test python -c "import causal_conv1d; print(causal_conv1d.__version__)"
```

本 Wheel 只匹配 `cp311 / Linux x86_64 / torch 2.10 / cu12`。不要随意执行
`pip install -U torch`；升级 torch 后需要重新安装匹配的 causal-conv1d。

### 6.5 仍然看到 FLA / causal-conv1d 的 fallback 警告

Transformers 在导入模型代码时就完成绑定。请确认是在**新进程**中导入，
并且 `import fla` / `import causal_conv1d` 本身成功：

```bash
micromamba run -n vlm-test python -c "import fla, causal_conv1d; print('ok')"
```

如果之前在同一进程里先导入过 Transformers，再安装包，需要重启 Kernel。

### 6.6 源码编译 CUDA 扩展

不要用 `/usr/bin/nvcc`（12.0，最高 `compute_90`）编译 sm_120 扩展。
在环境内安装 `cuda-toolkit=12.8` 或更高版本，并设置：

```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"
```

### 6.7 不要随意升级核心版本

已验证组合是：

```text
torch 2.10.0+cu128
transformers 5.20.2
flash-linear-attention 0.6.0
causal-conv1d 1.7.0 (cu12torch2.10 预编译)
```

升级其中任何一个都可能破坏扩展 ABI 或 Qwen3.5 接口。升级后至少重新执行
第 4.1、4.2、4.4 节验证。

---

## 7. 可复现性说明与限制

1. `environment-vlm-test.yml` + `requirements-vlm-test.txt` 可以复现
   Python 包版本，但 **不能单独复现 CUDA 扩展的二进制兼容性**：
   `causal-conv1d` 是平台/架构/torch 版本相关的预编译 Wheel。
2. 换机器时先确认 GPU 架构：
   - sm_120（Blackwell）必须用 CUDA 12.8+ 的 PyTorch Wheel；
   - 如果新机器是 Hopper/Ada，仍可使用 cu128，但建议重新选择对应的
     causal-conv1d Wheel。
3. 本环境没有安装系统级 CUDA Toolkit，也没有修改 NVIDIA 驱动；
   PyTorch 的 CUDA Runtime 由 Wheel 自带。
4. 本次没有下载模型权重或完整数据集；只下载了 Qwen3.5 的 tokenizer /
   processor / config 小文件用于接口验证。
5. 没有做端到端性能测试，因此不给出模型级加速比例；第 4.4 节只是
   Kernel 级微基准。

---

## 8. 文件清单与快速重建

新增文件：

```text
VLM-test/
├── environment-vlm-test.yml    # Conda 层：Python / pip / ipykernel
├── requirements-vlm-test.txt   # pip 层：精确版本 + CUDA 12.8 索引 + causal-conv1d Wheel
└── ENV_SETUP.md                # 本文档
```

在另一台同架构机器上重建：

```bash
micromamba create -n vlm-test -f environment-vlm-test.yml

micromamba run -n vlm-test python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

micromamba run -n vlm-test python -m pip install \
  -r requirements-vlm-test.txt

micromamba run -n vlm-test python -m ipykernel install \
  --user --name vlm-test --display-name "Python (vlm-test)"

micromamba run -n vlm-test python -m pip check
```

如果新机器的 Python 小版本或架构与本机不同，`requirements-vlm-test.txt`
中的 `causal-conv1d` 直链 Wheel 需要替换为对应版本，或改用源码编译。
