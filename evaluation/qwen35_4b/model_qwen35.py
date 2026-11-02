"""Qwen3.5-4B 适配器：加载、Prompt 构建、生成、计时与显存统计。

这是本目录里唯一与"具体模型"相关的模块。数据加载、答案解析、指标计算全部
复用 Xiaomi 评测目录中的通用实现（逐字节复制，见 README 的文件指纹表），
因此两个模型跑的是同一套评测协议。

Thinking 控制方式与 Xiaomi-Lens-0 相同：都使用 checkpoint 自带 chat template 的
布尔参数 ``enable_thinking``。

* ``enable_thinking=False`` → prompt 以 ``<think>\\n\\n</think>\\n\\n`` 结尾
* ``enable_thinking=True``  → prompt 以 ``<think>\\n`` 结尾

渲染完成后会检查后缀，避免模板变化导致 Thinking 开关"看起来改了、实际没改"。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL_NAME = "Qwen3.5-4B"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def mode_to_enable_thinking(mode: str) -> bool:
    """把实验模式映射成 chat template 的布尔开关。

    只有 off / on 两种原生模式；adaptive 不存在，因此在这里直接报错，
    而不是悄悄退化成 on 或 off。
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(
        f"模式 {mode!r} 没有 enable_thinking 映射；"
        "Qwen3.5-4B 的 chat template 只支持布尔开关，不支持原生 Adaptive Thinking。"
    )


def resolve_model_path(value: str | Path) -> Path:
    """返回本地 checkpoint 目录；已有完整缓存时不会重复下载。"""
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(str(value), local_files_only=True))
    except Exception:
        return Path(snapshot_download(str(value)))


class Qwen35Adapter:
    """官方 Transformers 推理封装，接口与 Xiaomi 适配器保持一致。"""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        attn_implementation: str = "sdpa",
    ):
        self.model_path = resolve_model_path(model_path)
        self.device = torch.device(device)
        self.requested_attn = attn_implementation
        self.attn_implementation: str | None = None
        self.model = None
        self.processor = None
        self.think_close_id: int | None = None
        self.load_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ 加载
    def load(self) -> dict[str, Any]:
        """只加载一次模型与 processor，返回加载耗时和显存快照。"""
        from transformers import Qwen3_5ForConditionalGeneration

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        load_kwargs = dict(device_map=str(self.device))
        try:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,          # Transformers 5.x 用 dtype（旧版叫 torch_dtype）
                attn_implementation=self.requested_attn,
                **load_kwargs,
            )
            self.attn_implementation = self.requested_attn
        except (TypeError, ValueError, ImportError) as exc:
            # 例如没装 flash-attn 时会退回 sdpa；记录真实生效的实现。
            print(f"[adapter] {self.requested_attn} 不可用（{exc}），改用 sdpa")
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                **load_kwargs,
            )
            self.attn_implementation = "sdpa"

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.think_close_id = self.processor.tokenizer.convert_tokens_to_ids(THINK_CLOSE)

        # device_map 若不是全部落在 GPU 上，说明发生了 CPU offload，速度会失真。
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map:
            distinct = {str(v) for v in hf_device_map.values()}
            cpu_offload = any(v in ("cpu", "disk") for v in distinct)
        else:
            cpu_offload = None

        torch.cuda.synchronize(self.device)
        load_seconds = time.perf_counter() - start

        self.load_info = {
            "model_name": MODEL_NAME,
            "model_path": str(self.model_path),
            "attn_implementation": self.attn_implementation,
            "dtype": "bfloat16",
            "load_seconds": load_seconds,
            "vram_after_load": self._vram_snapshot(),
            "hf_device_map": {k: str(v) for k, v in (hf_device_map or {}).items()},
            "cpu_offload": cpu_offload,
            "eos_token_ids": sorted(self._eos_ids()),
            "think_close_token_id": self.think_close_id,
            "image_token_id": getattr(self.model.config, "image_token_id", None),
        }
        return self.load_info

    # ------------------------------------------------------------ 构建输入
    def build_inputs(
        self, image: Image.Image, instruction: str, mode: str
    ) -> dict[str, Any]:
        """渲染 chat template + Processor，得到可送入 GPU 的张量。

        图片以 PIL Image 传入 content 列表，模板会把它替换成
        ``<|vision_start|><|image_pad|><|vision_end|>``，Processor 再把
        ``<|image_pad|>`` 展开成真正的视觉 token。
        """
        enable_thinking = mode_to_enable_thinking(mode)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }]
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        # 验证模板确实按请求的模式渲染，而不是靠"我们传了参数"来假设。
        expected_suffix = (
            f"{THINK_OPEN}\n" if enable_thinking
            else f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
        )
        mode_ok = prompt_text.endswith(expected_suffix)

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        torch.cuda.synchronize(self.device)
        preprocess_seconds = time.perf_counter() - start

        return {
            "inputs": inputs,
            "prompt_text": prompt_text,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "enable_thinking": enable_thinking,
            "mode_verified": mode_ok,
            "preprocess_seconds": preprocess_seconds,
        }

    # -------------------------------------------------------------- 生成
    @torch.inference_mode()
    def generate(
        self,
        prepared: dict[str, Any],
        mode: str,
        max_new_tokens: int,
        reset_peak: bool = True,
    ) -> dict[str, Any]:
        """贪心生成，计时与显存峰值都在 CUDA 同步之后读取。"""
        inputs = prepared["inputs"]
        prompt_length = inputs["input_ids"].shape[1]

        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # 确定性贪心，保证可复现
            use_cache=True,
        )
        torch.cuda.synchronize(self.device)
        generate_seconds = time.perf_counter() - start

        # 只解码新生成的 token；保留特殊 token，Parser 会自行剥离。
        completion_ids = output_ids[:, prompt_length:]
        generated_tokens = int(completion_ids.shape[1])
        raw_text = self.processor.tokenizer.decode(
            completion_ids[0], skip_special_tokens=False
        )

        thinking_tokens, answer_tokens, closed = self._split_token_counts(
            completion_ids[0], mode
        )
        last_id = int(completion_ids[0, -1]) if generated_tokens else None
        if last_id is not None and last_id in self._eos_ids():
            finish_reason = "eos"
        elif generated_tokens >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "other"

        return {
            "raw_generation": raw_text,
            "generated_tokens": generated_tokens,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
            "closed_thinking": closed,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "generate_seconds": generate_seconds,
            "tokens_per_second": (
                generated_tokens / generate_seconds if generate_seconds > 0 else None
            ),
            "vram": self._vram_snapshot(),
        }

    # -------------------------------------------------------------- 辅助
    def _split_token_counts(self, completion_ids: torch.Tensor, mode: str):
        """用 ``</think>`` 的 token id 精确切分 Thinking 与最终回答。"""
        length = int(completion_ids.shape[0])
        if self.think_close_id is None or mode == "off":
            return 0, length, False
        positions = (completion_ids == self.think_close_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            return length, 0, False
        close_index = int(positions[0].item())
        return close_index, length - close_index - 1, True

    def _eos_ids(self) -> set[int]:
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return {int(x) for x in eos}

    def _vram_snapshot(self) -> dict[str, float]:
        """PyTorch 分配器视角 + 整卡视角的显存快照（单位 GiB）。"""
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        gib = 1024 ** 3
        return {
            "allocated_gib": allocated / gib,
            "reserved_gib": reserved / gib,
            "peak_allocated_gib": peak_allocated / gib,
            "peak_reserved_gib": peak_reserved / gib,
            "gpu_used_gib": (total - free) / gib,
            "gpu_total_gib": total / gib,
        }
