"""两个模型共用的 SPAR-Bench 推理基类。

与 CV-Bench 适配器的区别（本文件是新建的，不修改旧代码）：

* ``build_inputs`` 接收 **图片列表**，按原始顺序把全部图片放进同一条
  user message；处理器之后会逐张展开 image token；
* 每次渲染后校验：``<|vision_start|>`` 数量、``image_grid_thw`` 行数都必须
  等于原始图片数量，避免 Multi-view 被静默截断；
* 可选地单独测一次 prefill（``use_cache=False``），用于区分 Prefill 与
  Decode 耗时；失败不影响正式生成。

子类只负责实现 ``_load_model``。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def resolve_model_path(value: str | Path) -> Path:
    """优先使用本地目录 / HF 缓存，绝不主动下载新权重。"""
    path = Path(value)
    if path.is_dir():
        return path.resolve()

    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(str(value), local_files_only=True))
    except Exception as exc:
        raise FileNotFoundError(
            f"本地 Hugging Face 缓存中找不到 {value}，且评测不允许自动下载：{exc}"
        ) from exc


class BaseSparAdapter:
    """统一的加载 / 输入构建 / 生成接口。"""

    MODEL_NAME = "base"
    MODEL_ID = ""

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
        self.measure_prefill = True
        self.model = None
        self.processor = None
        self.think_close_id: int | None = None
        self.think_open_id: int | None = None
        self.eos_ids: set[int] = set()
        self.load_info: dict[str, Any] = {}

    # ------------------------------------------------------------- 子类实现
    def _load_model(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ 加载
    def load(self, measure_prefill: bool = True) -> dict[str, Any]:
        """加载模型与 processor，记录加载耗时和显存快照。"""
        self.measure_prefill = measure_prefill
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self._load_model()
        self.model.eval()
        self.processor = self._build_processor()
        self.think_close_id = self.processor.tokenizer.convert_tokens_to_ids(THINK_CLOSE)
        self.think_open_id = self.processor.tokenizer.convert_tokens_to_ids(THINK_OPEN)
        self.eos_ids = self._eos_ids()

        hf_device_map = getattr(self.model, "hf_device_map", None)
        cpu_offload = None
        if hf_device_map:
            values = {str(v) for v in hf_device_map.values()}
            cpu_offload = any(v in ("cpu", "disk") for v in values)

        torch.cuda.synchronize(self.device)
        load_seconds = time.perf_counter() - start
        self.load_info = {
            "model_name": self.MODEL_NAME,
            "model_id": self.MODEL_ID,
            "model_path": str(self.model_path),
            "attn_implementation": self.attn_implementation,
            "dtype": "bfloat16",
            "load_seconds": load_seconds,
            "vram_after_load": self.vram_snapshot(),
            "hf_device_map": {k: str(v) for k, v in (hf_device_map or {}).items()},
            "cpu_offload": cpu_offload,
            "eos_token_ids": sorted(self.eos_ids),
            "think_close_token_id": self.think_close_id,
        }
        return self.load_info

    def _build_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)

    # ------------------------------------------------------------- 构建输入
    def build_inputs(
        self,
        images: list[Image.Image],
        instruction: str,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        """把全部图片 + 问题文本渲染成模型输入。"""
        if not images:
            raise ValueError("SPAR-Bench 样本没有图片")

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        expected_suffix = (
            f"{THINK_OPEN}\n" if enable_thinking
            else f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
        )
        mode_ok = prompt_text.endswith(expected_suffix)
        vision_markers = prompt_text.count("<|vision_start|>")

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=[prompt_text],
            images=[images],  # 单样本多图：外层是 batch 维度
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

        grid = inputs.get("image_grid_thw")
        grid_rows = int(grid.shape[0]) if grid is not None else None
        image_count_ok = (vision_markers == len(images)) and (
            grid_rows is None or grid_rows == len(images)
        )
        if not image_count_ok:
            raise RuntimeError(
                f"图片数量校验失败：原始 {len(images)} 张，"
                f"vision marker={vision_markers}，grid rows={grid_rows}"
            )

        return {
            "inputs": inputs,
            "prompt_text": prompt_text,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "enable_thinking": enable_thinking,
            "mode_verified": mode_ok,
            "n_images_prompt": vision_markers,
            "image_count_ok": image_count_ok,
            "preprocess_seconds": preprocess_seconds,
        }

    # --------------------------------------------------------------- Prefill
    @torch.inference_mode()
    def measure_prefill_time(self, prepared: dict[str, Any]) -> float | None:
        """单独跑一次无 KV Cache 的前向，测 Prefill 耗时。

        只保留最后 1 个位置的 logits（``logits_to_keep=1``），避免
        ``seq_len × vocab`` 的临时显存。旧模型类不接受该参数时退回完整前向。
        任何异常都只返回 None，不影响正式生成。
        """
        inputs = prepared["inputs"]
        try:
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            try:
                self.model(**inputs, use_cache=False, logits_to_keep=1)
            except TypeError:
                self.model(**inputs, use_cache=False)
            torch.cuda.synchronize(self.device)
            return time.perf_counter() - start
        except Exception as exc:  # pragma: no cover - 仅作诊断
            print(f"[adapter] prefill 计时失败（不影响生成）：{exc}")
            return None

    # -------------------------------------------------------------- 生成
    @torch.inference_mode()
    def generate(
        self,
        prepared: dict[str, Any],
        enable_thinking: bool,
        max_new_tokens: int,
        reset_peak: bool = True,
    ) -> dict[str, Any]:
        """贪心生成；计时前后都做 CUDA 同步。"""
        inputs = prepared["inputs"]
        prompt_length = int(inputs["input_ids"].shape[1])

        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize(self.device)
        generate_seconds = time.perf_counter() - start

        completion_ids = output_ids[:, prompt_length:]
        generated_tokens = int(completion_ids.shape[1])
        raw_text = self.processor.tokenizer.decode(
            completion_ids[0], skip_special_tokens=False
        )

        thinking_tokens, answer_tokens, closed = self._split_token_counts(
            completion_ids[0], enable_thinking
        )
        last_id = int(completion_ids[0, -1]) if generated_tokens else None
        if last_id is not None and last_id in self.eos_ids:
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
            "closed_thinking_tokens": closed,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "generate_seconds": generate_seconds,
            "tokens_per_second": (
                generated_tokens / generate_seconds if generate_seconds > 0 else None
            ),
            "vram": self.vram_snapshot(),
        }

    # ---------------------------------------------------------------- 辅助
    def _split_token_counts(self, completion_ids: torch.Tensor, enable_thinking: bool):
        """用 ``</think>`` 的 token id 精确统计 Thinking / Answer token 数。"""
        length = int(completion_ids.shape[0])
        if not enable_thinking or self.think_close_id is None:
            return 0, length, True
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
            return {int(eos)}
        return {int(x) for x in eos}

    def vram_snapshot(self) -> dict[str, float]:
        """PyTorch allocator 指标与整卡指标分开记录。"""
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
            "gpu_free_gib": free / gib,
            "gpu_used_gib": (total - free) / gib,
            "gpu_total_gib": total / gib,
        }
