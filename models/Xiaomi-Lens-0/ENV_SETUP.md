# Xiaomi-Lens-0 环境配置说明（Micromamba）

本文档记录在**当前这台服务器**上实际执行、实际验证通过的环境配置，不是通用教程。
命令中的版本号都是本机真实安装结果。

- 配置日期：2026-10-28
- 环境名：`xiaomi-lens`
- 环境路径：`/home/20261024_1/.local/share/mamba/envs/xiaomi-lens`

---

## 1. 机器与软件栈

| 项目 | 实际值 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.5 LTS (Noble), x86_64 |
| 内核 | 7.0.0-38-generic |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| 显存 | 97,887 MiB（约 96 GB） |
| Compute Capability | **12.0（sm_120，Blackwell）** |
| NVIDIA 驱动 | 580.178.04 |
| 驱动侧 CUDA 版本 | 13.0（仅代表驱动支持上限） |
| 系统自带 nvcc | 12.0（`/usr/bin/nvcc`，**不支持 sm_120，无法用于编译**） |
| Python | 3.11.18（Micromamba 环境内） |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| cuDNN | 9.10.02 |
| torchvision | 0.25.0+cu128 |
| transformers | 5.4.0 |

> **重要区分**：`nvidia-smi` 显示的 “CUDA Version: 13.0” 是**驱动支持的最高运行时版本**，
> 不是必须安装 CUDA 13。本项目在 CUDA 12.8 运行时下工作正常。

### 为什么选 cu128

RTX PRO 6000 Blackwell 是 **sm_120**。PyTorch 需要 `cu128` 及以上的 wheel 才包含 sm_120 内核。
实测本环境 `torch.cuda.get_arch_list()` 含 `sm_120`，GEMM 与 BF16 均正常。

---

## 2. 创建环境

```bash
micromamba create -n xiaomi-lens -c conda-forge python=3.11 pip -y
```

环境文件：`environment-xiaomi-lens.yml`、`requirements-xiaomi-lens.txt`。

> 项目自带的 `requirements.txt` / `pyproject.toml` 里 `torch`、`flash-attn` 等没有锁定 CUDA 变体，
> 直接 `pip install -r requirements.txt` 会从 PyPI 默认源拉到非 cu128 的 torch，**不要那样做**。
> 正确顺序见下文第 3 节。

---

## 3. 实际安装步骤（按顺序）

所有命令都在 `xiaomi-lens` 环境内执行，例如：

```bash
micromamba run -n xiaomi-lens python -m pip install ...
# 或先 micromamba activate xiaomi-lens 再直接操作
```

### 3.1 PyTorch（官方 cu128 源）

```bash
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

装完立即验证（未通过就不要继续）：

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())          # 需含 sm_120
print(torch.cuda.is_bf16_supported())      # True
```

### 3.2 Transformers 与基础依赖

```bash
pip install "transformers==5.4.0" "accelerate>=1.13.0"
pip install -e .          # 项目本体；pyproject 中 dependencies=[]，不会拉 vLLM/DeepSpeed
```

- 固定 `transformers==5.4.0` 的原因：模型代码 import 了
  `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration`，
  且 checkpoint 的 `generation_config.json` 中 `transformers_version` 就是 `5.4.0`。
- 修正 huggingface 代理问题（见第 6 节）：

```bash
pip install "httpx[socks]"     # 提供 socksio
```

### 3.3 Flash Linear Attention（纯 Triton，无需编译）

```bash
pip install "flash-linear-attention==0.5.2"
```

> **不要装 0.6.0**：它会强制把 transformers 升到 `>=5.6.2`，破坏 5.4.0 固定版本。
> 若已经装上，用 `pip install "transformers==5.4.0"` 回退即可。

Qwen3.5 的 `linear_attention` 层（GatedDeltaNet）会使用：

- `fla.ops.gated_delta_rule.chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`
- `fla.modules.FusedRMSNormGated`

### 3.4 编译工具链（CUDA 12.8）

系统 nvcc 只有 12.0，无法生成 sm_120 代码。**不安装系统级 CUDA Toolkit**，
只在 conda 环境内安装编译器：

```bash
micromamba install -n xiaomi-lens -c conda-forge \
    cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 -y
```

之后 `CUDA_HOME` 指向环境前缀：

```bash
export CUDA_HOME=/home/20261024_1/.local/share/mamba/envs/xiaomi-lens
$CUDA_HOME/bin/nvcc --version     # 应为 release 12.8, V12.8.93
```

### 3.5 causal-conv1d（源码编译）

```bash
export CUDA_HOME=/home/20261024_1/.local/share/mamba/envs/xiaomi-lens
export MAX_JOBS=16
cd /tmp && pip download --no-deps --no-binary :all: causal-conv1d==1.6.1
tar xzf causal_conv1d-1.6.1.tar.gz
cd causal_conv1d-1.6.1
pip install --no-build-isolation --no-deps .
```

其 `setup.py` 在 CUDA>=12.8 时会自动追加 `arch=compute_120,code=sm_120`。

### 3.6 flash-attn（源码编译，仅 sm_120）

官方 GitHub Releases 只提供到 **torch 2.8** 的预编译 wheel，与本机 torch 2.10 ABI 不匹配；
PyPI 上只有 sdist。因此需要本机编译。

```bash
export CUDA_HOME=/home/20261024_1/.local/share/mamba/envs/xiaomi-lens
export MAX_JOBS=16
export FORCE_BUILD=1                 # 跳过下载预编译 wheel 的尝试
export FLASH_ATTN_CUDA_ARCHS=120     # 只编译 sm_120，显著缩短编译时间

pip download --no-deps --no-binary :all: flash-attn==2.8.3.post2
tar xzf flash_attn-2.8.3.post2.tar.gz
cd flash_attn-2.8.3.post2
pip install --no-build-isolation --no-deps .
```

`setup.py` 在 CUDA>=12.8 且 `120 ∈ FLASH_ATTN_CUDA_ARCHS` 时生成
`arch=compute_120,code=sm_120`。

> ⚠️ 这样编译出的 wheel **只含 sm_120**，换到其它架构 GPU 上不可用，需要重新编译。

本机实测：编译成功，产物 `flash_attn-2.8.3.post2-cp311-cp311-linux_x86_64.whl`
（约 68 MB，因为只编了 sm_120；官方全架构 wheel 约 244 MB）。
当前宿主有其他用户任务、平均负载 20–35，整个编译耗时约 40 分钟。

编译产物没有归档，**环境一旦删除需要重新编译**。若想复用，可把
`pip wheel` 生成的 wheel 保存到项目外部的目录。

---

## 4. 验证

### 4.1 一条命令自检

```bash
micromamba run -n xiaomi-lens python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("arch", torch.cuda.get_arch_list())
print("bf16", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
from xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration
from transformers import AutoProcessor
print("model/processor classes OK")
import flash_attn, causal_conv1d, fla
print("flash_attn", flash_attn.__version__, "causal_conv1d", causal_conv1d.__version__)
PY
```

### 4.2 检查三个加速内核是否被 transformers 真正启用

```bash
micromamba run -n xiaomi-lens python -c "
import transformers.models.qwen3_5.modeling_qwen3_5 as m
print('flash_attn full-attn path:', m.is_flash_attn_2_available() if hasattr(m,'is_flash_attn_2_available') else 'see Qwen3_5 supports')
print('causal_conv1d:', 'LOADED' if m.causal_conv1d_fn is not None else 'FALLBACK')
print('fla chunk_gated_delta_rule:', 'LOADED' if m.chunk_gated_delta_rule is not None else 'FALLBACK')
print('fla fused_recurrent:', 'LOADED' if m.fused_recurrent_gated_delta_rule is not None else 'FALLBACK')
"
```

三个都应为 `LOADED`。若某个是 `FALLBACK`，说明该组件没装好（不会报错，只会变慢）。

### 4.3 依赖一致性

```bash
micromamba run -n xiaomi-lens python -m pip check
# -> No broken requirements found.
```

### 4.4 本机实测结果（2026-10-28）

| 检查项 | 结果 |
| --- | --- |
| `torch.cuda.is_available()` | True |
| `torch.cuda.get_device_capability(0)` | (12, 0) → sm_120 |
| `torch.cuda.get_arch_list()` | 含 `sm_120` |
| `torch.cuda.is_bf16_supported()` | True |
| BF16 4096×4096 GEMM | 正常，数值正确 |
| `pip check` | No broken requirements found |
| `infer.py --help` | 正常 |
| `AutoProcessor` 加载 | 正常（Qwen3VLProcessor） |
| `<|mask|>` token id | 248077 |
| `build_inputs`（图+文本） | 正常，输出 `input_ids/pixel_values/image_grid_thw/mm_token_type_ids` |
| `causal_conv1d_fn` 内核 | 正常（sm_120，结果有限） |
| fla `chunk_gated_delta_rule` | 正常；与 PyTorch 参考实现最大绝对误差 1.5e-3（bf16 量级） |
| fla `FusedRMSNormGated` | 正常 |
| `flash_attn_func` hdim 64/128/256 | 正常 |
| `flash_attn_varlen_func` | 正常；与 SDPA 最大绝对误差 3.9e-3 |
| PixelMaskDecoder（flash 路径） | 正常，输出 `(1,3,128,128)`，无回退警告 |
| 真实 GatedDeltaNet 层 | 走 `causal_conv1d` + `fla.ops.gated_delta_rule.chunk` |
| 端到端模型前向（真实 config，层数缩减为 4） | 正常；`attn_implementation=flash_attention_2`，输出 `pred_masks (1,1,480,640)`，峰值显存 2.7 GB |
| 真实权重推理 | **未验证**（本地无权重，按规定未自动下载） |

> 端到端前向用的是真实 config 维度 + 真实 Processor 输入 + **随机初始化权重**，
> 目的只在于验证计算路径（flash attention / GatedDeltaNet / Mask Decoder）可跑通，
> 输出内容没有语义意义。

---

## 5. 运行官方推理

```bash
micromamba activate xiaomi-lens
cd /home/20261024_1/Workspace/qyh/VLM-test/Xiaomi-Lens-0

python infer.py \
  --checkpoint SeerRay-Lab/Xiaomi-Lens-0 \
  --image /path/to/image.jpg \
  --text 'Locate all instances in the image that match the description "the cat on the left". Report bbox coordinates and masks.' \
  --output result.jpg --draw-bbox
```

常用参数：

- `--per-mask-output`：每个掩码单独存一张图
- `--mask-threshold`：默认 0.5
- `--max-new-tokens`：默认 16384，可调小加速
- `--temperature 0`（默认）为贪心解码

查看帮助：`python infer.py --help`

Python API 用法见 `README_CN.md` 的「Python API」小节。

---

## 6. 遇到的问题与解决

### 6.1 系统 nvcc 12.0 无法编译 sm_120

`/usr/bin/nvcc` 是 CUDA 12.0，早于 Blackwell 支持。**没有**安装系统级 CUDA Toolkit，
而是在 conda 环境中安装 `cuda-nvcc=12.8` 并把 `CUDA_HOME` 指过去。

### 6.2 `nvidia-cuda-nvcc-cu12` pip 包不含 nvcc

`pip install nvidia-cuda-nvcc-cu12==12.8.93` 只提供 `ptxas` 和 `crt` 头文件，**没有 `nvcc`**。
所以编译器必须走 conda-forge 的 `cuda-nvcc`。

### 6.3 flash-attn 没有匹配 torch 2.10 的预编译轮子

官方 release 最高到 torch 2.8；第三方镜像只有 cp312/torch2.10 或 torch2.11 的轮子。
最终选择源码编译（官方 `setup.py` 已支持 `compute_120`）。

### 6.4 flash-linear-attention 0.6.0 会把 transformers 升级到 5.21

`fla 0.6.0` 的元数据要求 `transformers>=5.6.2`，安装时会自动升级。
解决：使用 `fla 0.5.2`（只要求 `transformers>=4.45.0`）。

### 6.5 huggingface_hub 走 socks 代理报错

环境变量 `ALL_PROXY=socks5://127.0.0.1:7890` 时，`huggingface_hub`（基于 httpx）会报：

```
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

解决：`pip install "httpx[socks]"`（安装 `socksio`）。

### 6.6 模型权重缺失

Hugging Face 缓存中 `models--SeerRay-Lab--Xiaomi-Lens-0` 目录原先只有 `refs/main`，
**没有权重文件**。本次只拉取了约 19 MB 的配置 / tokenizer 文件用于验证 Processor，
没有下载约 8 GB 的 `model-*.safetensors`。首次真实推理时需要下载（见第 7 节）。

---

## 7. 首次推理前下载权重

```bash
micromamba activate xiaomi-lens
# 3 个分片，约 8 GB
hf download SeerRay-Lab/Xiaomi-Lens-0 --local-dir ~/models/Xiaomi-Lens-0
```

之后 `--checkpoint` 传本地目录即可。

> 当前 `HF_ENDPOINT=https://hf-mirror.com`，下载走镜像。

---

## 8. 在 PyCharm / VS Code 中选择解释器

解释器路径：

```
/home/20261024_1/.local/share/mamba/envs/xiaomi-lens/bin/python
```

- **PyCharm**：Settings → Project → Python Interpreter → Add Interpreter →
  Add Local Interpreter → System Interpreter → 选上面的路径。
- **VS Code**：`Python: Select Interpreter` → Enter interpreter path → 同上。
- 如需在 IDE 终端里编译 CUDA 扩展，记得先 `micromamba activate xiaomi-lens`。

---

## 9. 迁移到新服务器时注意

1. **先看 GPU 架构**：`torch.cuda.get_device_capability(0)`。Blackwell 需要 cu128+。
2. **torch 版本决定 flash-attn 能否用预编译轮子**：官方 wheel 覆盖到 torch 2.8；
   更新版本需要本机编译。
3. **CUDA 扩展与本机 ABI 强绑定**：`flash-attn`、`causal-conv1d` 的 wheel 不能跨
   torch 版本 / Python 版本 / CUDA 版本复制。
   能复用的是 `environment-xiaomi-lens.yml` + 本文档的编译步骤，不是编译产物。
4. **不要用系统 nvcc**：若系统 nvcc < 12.8，按 3.4 在环境内装 `cuda-nvcc`。
5. **`ALL_PROXY` 为 socks5 时**，务必先装 `httpx[socks]`。
6. **transformers 固定 5.4.0**，升级前先确认 `qwen3_5` 模块 API 未变。
7. 本环境**没有**安装 vLLM / SGLang / DeepSpeed / Ray，评测 / 训练 / serving 需要另建环境。
