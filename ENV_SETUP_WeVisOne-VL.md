# WeVisOne-VL 独立 Micromamba 环境配置说明

> 配置日期：2026-11-10  
> 项目路径：`/home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL`  
> 环境文件：`/home/20261024_1/Workspace/qyh/VLM-test/environment-wevisone-vl.yml`

本文档记录 `wevisone-vl` 环境的实际创建、依赖安装、CUDA 验证和后续权重下载方式。腾讯官方源码、`pyproject.toml`、`requirements.txt` 均未修改。

---

## 1. 环境概览

| 项目 | 实际值 |
|---|---|
| Micromamba | 2.9.0 |
| 环境名称 | `wevisone-vl` |
| 环境路径 | `/home/20261024_1/.local/share/mamba/envs/wevisone-vl` |
| Python | 3.12.16 |
| PyTorch | `2.11.0+cu130` |
| torchvision | `0.26.0+cu130` |
| Transformers | `5.12.1` |
| accelerate | `1.14.0` |
| CUDA runtime（PyTorch 内置） | 13.0 |
| NVIDIA 驱动 | 580.178.04（支持 CUDA 13.0） |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| 显存 | 94.96 GiB |
| Compute Capability | `(12, 0)`，即 sm_120 |
| BF16 | 支持，已用 4096×4096 BF16 矩阵乘验证 |
| 项目包 | `wevisone-vl==1.0.0`，普通安装（非 editable） |
| 安装位置 | `.../site-packages/wevisone_vl/` |

`WeVisOneVL` 已成功导入；由于本地没有模型权重，**尚未执行真实模型推理**（详见第 7、8 节）。

---

## 2. 为什么最终使用 Python 3.12，而不是 3.10

官方 `pyproject.toml` 写的是 `requires-python = ">=3.10"`，但官方同时固定了以下版本：

| 包 | 固定版本 | 实际 Python 要求 |
|---|---|---|
| matplotlib | `3.11.0` | `>=3.11` |
| scipy | `1.18.1` | `>=3.12` |

即官方声明的 `>=3.10` 与其依赖锁定自相矛盾。实际在 Python 3.10 下执行 `pip install .` 会直接失败：

```text
ERROR: Ignored the following versions that require a different python version:
3.11.0 Requires-Python >=3.11
ERROR: No matching distribution found for matplotlib==3.11.0
```

处理原则：

- **不修改**官方 `pyproject.toml` / `requirements.txt`；
- **不擅自降级**官方固定版本；
- 选择 Python 3.12，使全部官方固定版本可以精确安装；Python 3.12 也满足官方声明的 `>=3.10`。

这是本次配置中唯一的“偏离”，且偏离的是 Python 版本而不是依赖版本，原因如上。

---

## 3. 从零创建环境

### 3.1 直接命令方式（本次实际执行方式）

```bash
micromamba create -n wevisone-vl -c conda-forge python=3.12 pip -y
micromamba activate wevisone-vl

# 先安装 PyTorch，确认 CUDA 构建正确
python -m pip install torch==2.11.0 torchvision==0.26.0

# 再普通安装项目本体（非 editable）
cd /home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL
python -m pip install .
```

### 3.2 通过环境文件复现（推荐在其他机器上使用）

```bash
cd /home/20261024_1/Workspace/qyh/VLM-test
micromamba env create -f environment-wevisone-vl.yml -y
micromamba activate wevisone-vl

cd models/WeVisOne-VL
python -m pip install . --no-deps
```

说明：

- `--no-deps` 是因为 `environment-wevisone-vl.yml` 已经固定了全部依赖，避免 pip 再次解析时升级版本。
- 如果怀疑有依赖缺失，可以去掉 `--no-deps` 重新执行 `python -m pip install .`。
- `environment-wevisone-vl.yml` 中不包含 `wevisone-vl` 本身，因为它来自本地源码，不是 PyPI 包。
- 该 YAML 固定为 **Linux x86_64 + NVIDIA GPU** 方案，其中 `nvidia-*` / `cuda-toolkit` 等 pip 包仅适用于 Linux。macOS 或纯 CPU 机器需要删除这些包并改用对应平台的 torch 构建。

### 3.3 为什么没有使用 `requirements.txt`

官方 `requirements.txt` 包含：

```text
gradio
triton
vllm
```

本次只需要 Transformers Backend 的 Python API 推理，因此没有执行 `pip install -r requirements.txt`，避免安装：

- **vLLM**：大型推理框架，本次不需要；
- **Gradio**：只有启动 Web Demo 才需要；
- **vLLM 侧的 Triton**：与本次推理无关。

但需要注意：**`triton==3.6.0` 仍然是 `torch==2.11.0` 在 Linux 上的硬依赖**，pip 会自动安装它。这不是我们额外加入的 vLLM 依赖，也不会影响 Transformers/SDPA 推理。

### 3.4 为什么没有使用 `pip install -e .`

按照要求，本次使用普通安装 `pip install .`，把包实际复制到 `site-packages`，避免 PyCharm 无法正确解析 editable install 生成的 `__editable__` 路径。

代价是：如果以后修改了 `models/WeVisOne-VL/wevisone_vl/` 下的源码，需要重新安装才能生效：

```bash
cd /home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL
python -m pip install . --force-reinstall --no-deps
```

---

## 4. 已安装依赖

### 4.1 官方核心依赖（全部精确匹配）

| 包 | 版本 |
|---|---|
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| transformers | 5.12.1 |
| accelerate | 1.14.0 |
| huggingface-hub | 1.21.0 |
| numpy | 2.2.6 |
| Pillow | 12.2.0 |
| pycocotools | 2.0.11 |
| qwen_vl_utils | 0.0.14 |
| safetensors | 0.8.0 |
| scipy | 1.18.1 |
| supervision | 0.31.0 |
| matplotlib | 3.11.0 |

完整传递依赖（包括 CUDA 运行时、tokenizers、av、typer 等）已固定在同目录的 `environment-wevisone-vl.yml` 中。

### 4.2 明确未安装的组件

| 组件 | 状态 | 原因 |
|---|---|---|
| vLLM | 未安装 | 本次使用 Transformers Backend，不需要 |
| Gradio | 未安装 | 仅 Web Demo 需要；需要时再 `pip install gradio==6.30.0` |
| FlashAttention 2 | 未安装 | 无预编译 wheel，且本机 nvcc 与 torch CUDA 版本不匹配，详见第 9 节 |
| OpenCV | 未安装 | 官方依赖未包含；`supervision` 使用 NumPy fallback，可视化已验证可用 |

---

## 5. PyTorch 与 CUDA 兼容性

### 5.1 事实依据

- 驱动：`NVIDIA-SMI 580.178.04`，`CUDA Version: 13.0`。
- PyPI 上 `torch==2.11.0` 的 Linux x86_64 wheel 默认就是 CUDA 13.0 构建，其元数据要求：
  - `cuda-toolkit==13.0.2`
  - `nvidia-cudnn-cu13==9.19.0.56`
  - `nvidia-nccl-cu13==2.28.9`
  - `triton==3.6.0`
- 安装后实际报告：`torch 2.11.0+cu130`，`torch.version.cuda == "13.0"`。
- GPU 为 Blackwell 架构 sm_120，CUDA 13.0 支持该架构。
- **没有升级系统驱动，也没有安装 CPU-only 版本。**

### 5.2 验证命令

```bash
micromamba activate wevisone-vl
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

本次实际输出：

```text
PyTorch: 2.11.0+cu130
CUDA: 13.0
CUDA available: True
GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
VRAM: 94.9600830078125
```

---

## 6. 环境验证记录

### 6.1 依赖一致性

```bash
python -m pip check
```

实际输出：

```text
No broken requirements found.
```

### 6.2 基础依赖导入

```bash
python - <<'PY'
import torch, torchvision, transformers, accelerate, numpy, PIL, scipy, supervision, qwen_vl_utils
print("All core dependencies imported successfully!")
PY
```

实际输出（`supervision` 会打印一条 OpenCV 缺失的 warning，属于预期行为）：

```text
torch        : 2.11.0+cu130
torchvision  : 0.26.0+cu130
transformers : 5.12.1
accelerate   : 1.14.0
numpy        : 2.2.6
Pillow       : 12.2.0
scipy        : 1.18.1
supervision  : 0.31.0
All core dependencies imported successfully!
```

### 6.3 项目代码导入

在 `/tmp`（非项目目录）中执行，确认导入的是安装到 `site-packages` 的正确版本：

```bash
cd /tmp
python - <<'PY'
import wevisone_vl
from wevisone_vl import WeVisOneVL
print("package file:", wevisone_vl.__file__)
print("version     :", wevisone_vl.__version__)
print("WeVisOneVL imported successfully")
PY
```

实际输出：

```text
package file: /home/20261024_1/.local/share/mamba/envs/wevisone-vl/lib/python3.12/site-packages/wevisone_vl/__init__.py
version     : 1.0.0
WeVisOneVL imported successfully
```

### 6.4 CUDA / BF16 实测

```text
CUDA available: True
GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
Capability: (12, 0)
BF16 supported: True
BF16 matmul OK
SDPA available: True
```

### 6.5 可视化路径

使用一张合成图片和伪造的 `InferenceResult` 调用 `save_result`，成功生成：

```text
overlay.png
alpha.png
cutout.png
masks/
result.json
response.txt
```

说明即使没有 OpenCV，`supervision` 的 NumPy fallback 也能完成可视化。

### 6.6 权重缓存检查

```python
from huggingface_hub import try_to_load_from_cache
try_to_load_from_cache("tencent/WeVisOne-VL-4B", "config.json")  # -> None
try_to_load_from_cache("tencent/WeVisOne-VL-2B", "config.json")  # -> None
```

本地 `~/.cache/huggingface/hub` 中没有 `tencent/WeVisOne-VL-4B` 或 `tencent/WeVisOne-VL-2B`，因此没有执行真实模型推理。

---

## 7. 模型权重下载（未自动执行）

按任务要求，环境配置与权重下载分开处理。本次只查询了仓库元数据，**没有下载任何模型权重**。

### 7.1 权重规模

| 模型 | 主要文件 | 总大小 |
|---|---|---|
| `tencent/WeVisOne-VL-4B` | `model.safetensors` 9213.7 MiB + `mask_decoder.pt` 36.7 MiB | 约 **9.04 GiB** |
| `tencent/WeVisOne-VL-2B` | `model.safetensors` 4657.3 MiB + `mask_decoder.pt` 36.4 MiB | 约 **4.59 GiB** |

完整的快照还包含 `config.json`、`generation_config.json`、`processor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 等小文件。

### 7.2 下载命令

当前 shell 全局设置了 `all_proxy=socks5h://127.0.0.1:7890`，而环境内没有安装 `socksio`，Hugging Face CLI（httpx）会报：

```text
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

解决办法二选一：

```bash
# 方案 A（推荐，不增加依赖）：仅对当前命令取消 SOCKS 代理，保留 HTTP/HTTPS 代理
unset all_proxy ALL_PROXY

# 方案 B：安装 SOCKS 支持
python -m pip install 'httpx[socks]'
```

然后下载（`HF_ENDPOINT` 已全局设置为 `https://hf-mirror.com`）：

```bash
cd /home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL

unset all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com

# 4B（约 9.04 GiB）
hf download tencent/WeVisOne-VL-4B \
  --local-dir /home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL/checkpoints/WeVisOne-VL-4B

# 2B（可选，约 4.59 GiB）
# hf download tencent/WeVisOne-VL-2B \
#   --local-dir /home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL/checkpoints/WeVisOne-VL-2B
```

如果只使用默认 Hugging Face 缓存（`~/.cache/huggingface/hub`），可以省略 `--local-dir`：

```bash
hf download tencent/WeVisOne-VL-4B
```

注意事项：

- 必须下载**完整快照**，尤其是 `mask_decoder.pt`。使用本地路径加载时，若缺少该文件，`WeVisOneVL` 会抛出 `FileNotFoundError: missing alpha decoder sidecar`。
- 下载前确认磁盘空间：4B 约 9 GiB，2B 约 4.6 GiB。
- 如果 hf-mirror 不可用，可临时 `export HF_ENDPOINT=https://huggingface.co`（需要代理可访问）。
- 权重下载与当前环境配置相互独立，不需要重新安装环境。

---

## 8. 简单推理示例

权重下载完成后，使用本地路径加载（推荐，避免运行时再访问 Hugging Face）：

```python
from PIL import Image
from wevisone_vl import WeVisOneVL
from wevisone_vl.visualization import save_result

MODEL_DIR = (
    "/home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL/"
    "checkpoints/WeVisOne-VL-4B"
)
IMAGE_PATH = (
    "/home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL/"
    "demo/examples/images/livingroom.jpg"
)

model = WeVisOneVL(
    MODEL_DIR,
    backend="transformers",   # 不使用 vLLM
    device="auto",            # 自动选择 cuda:0
    dtype="bfloat16",         # Blackwell 支持 BF16
    attention_implementation="sdpa",
)

image = Image.open(IMAGE_PATH).convert("RGB")
result = model.predict(
    image,
    task="detection",
    query=["sofa", "pillow", "coffee table", "lamp"],
)

print(result.response)
print("boxes:", result.boxes)
print("latency_s:", result.latency_s)

save_result(
    "/home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL/outputs/detection",
    image,
    result,
)
```

也可以直接使用仓库 ID（会自动使用 Hugging Face 缓存或在线下载）：

```python
model = WeVisOneVL(
    "tencent/WeVisOne-VL-4B",
    backend="transformers",
    device="auto",
)
```

API 实际签名（已核对）：

```text
WeVisOneVL.__init__(
    model_name_or_path="tencent/WeVisOne-VL-4B",
    *,
    backend="transformers",
    device="auto",
    dtype="bfloat16",
    revision=None,
    min_pixels=16384,
    max_pixels=2073600,
    attention_implementation="sdpa",
    ...
)

WeVisOneVL.predict(
    image,
    *,
    task="freeform",
    query="",
    prompt=None,
    reference_boxes=None,
    decode_matting=True,
    **generation_kwargs,
)
```

---

## 9. 加速组件结论

| 组件 | 状态 | 说明 |
|---|---|---|
| SDPA | ✅ 可用并已采用 | `WeVisOneVL` 默认 `attention_implementation="sdpa"`；未安装 FlashAttention 时这是推荐后端 |
| BF16 | ✅ 已验证 | `torch.cuda.is_bf16_supported() == True`，BF16 矩阵乘测试通过 |
| FlashAttention 2 | ❌ 未安装 | PyPI 上 `flash-attn==2.8.3.post2` 只有源码包（sdist，8.5 MB），没有预编译 wheel；本机 `/usr/bin/nvcc` 是 CUDA 12.0，而 torch 是 cu130，编译环境不匹配，强行编译耗时长且可能破坏环境 |
| vLLM | ❌ 未安装 | 按任务要求不安装 |
| Triton | ⚠️ 已安装 3.6.0 | 这是 `torch==2.11.0` Linux 轮子的**硬依赖**，不是 vLLM 的可选 Triton；正常推理不会直接调用 |
| Gradio | ❌ 未安装 | 仅 Web Demo 需要；需要时执行 `python -m pip install gradio==6.30.0` |

### 运行官方 evaluation 脚本时的注意点

`evaluation/inference_refseg.py` 和 `evaluation/inference_matting.py` 的 `--attn_implementation` 默认值是 `flash_attention_2`。当前环境没有 FlashAttention，运行时应显式改为 `sdpa`：

```bash
python evaluation/inference_refseg.py \
  --attn_implementation sdpa \
  ...其他参数...
```

`evaluation/README.md` 也明确说明：未安装 flash-attn 时传 `sdpa`。

---

## 10. 在 PyCharm 中选择该环境

1. 打开 PyCharm，进入 `File → Settings`（Linux 下 `Ctrl+Alt+S`）。
2. 选择 `Project: VLM-test → Python Interpreter`。
3. 点击 `Add Interpreter → Add Local Interpreter`。
4. 选择 `System Interpreter`（或 `Existing environment`）。
5. 解释器路径填写：

   ```text
   /home/20261024_1/.local/share/mamba/envs/wevisone-vl/bin/python
   ```

6. 确认解释器版本为 Python 3.12.16，并在包列表中看到：
   - `torch 2.11.0`
   - `torchvision 0.26.0`
   - `transformers 5.12.1`
   - `wevisone-vl 1.0.0`
7. 如果包列表没有刷新，执行 `File → Invalidate Caches / Restart`。

补充说明：

- 由于本次使用普通安装而不是 `pip install -e .`，不会出现 PyCharm 无法解析 `__editable__` 路径的问题。
- 运行配置的 **Working directory** 建议设为：

  ```text
  /home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL
  ```

  这样官方示例中的相对路径（如 `demo/examples/images/livingroom.jpg`）可以直接使用；否则请改用绝对路径。
- 如果 PyCharm 以 `models/WeVisOne-VL` 为项目根目录，它会优先导入该目录下的本地 `wevisone_vl` 源码，这与 `site-packages` 中安装的版本内容一致；如果希望强制使用安装版本，把工作目录设为 `VLM-test` 根目录即可。

---

## 11. 常见问题（FAQ）

### Q1：为什么 Python 3.10 装不上？

官方固定了 `matplotlib==3.11.0`（要求 Python ≥3.11）和 `scipy==1.18.1`（要求 Python ≥3.12）。使用 Python 3.10 会出现 `No matching distribution found for matplotlib==3.11.0`。本环境改用 Python 3.12，未修改官方版本锁定。

### Q2：`Using SOCKS proxy, but the 'socksio' package is not installed`

这是 Hugging Face CLI 通过 `all_proxy=socks5h://...` 访问时缺少 SOCKS 支持导致的。执行：

```bash
unset all_proxy ALL_PROXY
```

或：

```bash
python -m pip install 'httpx[socks]'
```

### Q3：加载本地模型时报 `missing alpha decoder sidecar`

本地模型目录必须包含 `mask_decoder.pt`。请确认下载的是完整快照：

```bash
ls checkpoints/WeVisOne-VL-4B/mask_decoder.pt
```

### Q4：运行 evaluation 脚本时报 `flash_attn` 相关错误

这些脚本默认 `--attn_implementation flash_attention_2`。当前环境没有安装 FlashAttention，请改为：

```bash
--attn_implementation sdpa
```

不要为了一个可选加速组件破坏已经验证通过的 torch/CUDA 环境。

### Q5：导入 `supervision` 时提示 OpenCV 未安装

这是 warning，不是错误。`supervision==0.31.0` 会自动使用纯 NumPy fallback，本次已用合成数据验证 `save_result` 可正常生成 overlay/alpha/cutout。若需要完整性能，可选安装：

```bash
python -m pip install opencv-python-headless
```

但该包不在官方固定依赖中，是否安装不影响模型推理。

### Q6：`torch.cuda.is_available()` 返回 False

依次检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

当前环境使用 CUDA 13.0 构建，需要支持 CUDA 13 的驱动（本机 580.178.04 满足要求）。不要为了 PyTorch 升级系统 CUDA 驱动。

### Q7：修改了官方源码后没有生效

因为使用的是普通安装，需要重新安装：

```bash
cd /home/20261024_1/Workspace/qyh/VLM-test/models/WeVisOne-VL
python -m pip install . --force-reinstall --no-deps
```

### Q8：为什么不能直接 `pip install -r requirements.txt`？

会安装 `gradio`、`vllm` 和 vLLM 侧的可选组件，体积大且与本次 Transformers 推理无关。应使用 `pyproject.toml` 的基础依赖，即 `pip install .`。注意 `triton==3.6.0` 仍会作为 torch 的硬依赖被安装，这是正常的。

### Q9：模型下载到哪里？如何清理缓存？

默认缓存：

```text
~/.cache/huggingface/hub
```

使用 `--local-dir` 时文件在指定目录。清理缓存可使用：

```bash
hf cache delete
```

或直接删除对应 `models--tencent--WeVisOne-VL-4B` 目录。

### Q10：显存是否足够？

本机 94.96 GiB 显存，4B 模型权重约 9 GiB，BF16 推理余量充足。实际显存占用还取决于图片分辨率、视觉 token 数（`min_pixels` / `max_pixels`）和 batch 大小。若出现 OOM，优先降低 `max_pixels` 或 batch 大小。

---

## 12. 交付文件

| 文件 | 用途 |
|---|---|
| `environment-wevisone-vl.yml` | 固定 Python 3.12 + 全部 pip 依赖版本，便于在其他机器复现 |
| `ENV_SETUP_WeVisOne-VL.md` | 本说明文档 |
| `models/WeVisOne-VL/pyproject.toml` | 官方依赖定义（未修改） |
| `models/WeVisOne-VL/requirements.txt` | 官方完整依赖（未修改，本次未直接使用） |

现有 `environment-vlm-test.yml` 未被覆盖。
