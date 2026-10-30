"""Xiaomi-Lens-0 adapter: loading, prompt building, generation and metrics.

This is the only model-specific module.  Everything else (dataset, parser,
metrics, runner) is model-agnostic, so evaluating another VLM means writing a
similar adapter with ``generate()`` returning the same dictionary.

Thinking control uses the checkpoint's own chat template:

* ``enable_thinking=False`` -> prompt ends with ``<think>\\n\\n</think>\\n\\n``
* ``enable_thinking=True``  -> prompt ends with ``<think>\\n``

The prompt suffix is checked after rendering, so a template change cannot
silently leave Thinking ON/OFF in the wrong state.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "Xiaomi-Lens-0"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def mode_to_enable_thinking(mode: str) -> bool:
    """Map an experiment mode to the chat-template flag."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(
        f"Mode {mode!r} has no enable_thinking mapping; "
        "adaptive thinking is not supported by this checkpoint."
    )


def resolve_model_path(value: str | Path) -> Path:
    """Return a local checkpoint directory, reusing the HF cache when possible."""
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(str(value), local_files_only=True))
    except Exception:
        return Path(snapshot_download(str(value)))


class XiaomiLensAdapter:
    """Thin wrapper around the official model/processor with instrumented generate."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        attn_implementation: str = "flash_attention_2",
    ):
        self.model_path = resolve_model_path(model_path)
        self.device = torch.device(device)
        self.requested_attn = attn_implementation
        self.attn_implementation: str | None = None
        self.model = None
        self.processor = None
        self.think_close_id: int | None = None
        self.mask_token_id: int | None = None
        self.load_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ load
    def load(self) -> dict[str, Any]:
        """Load model + processor once; return load-time and VRAM statistics."""
        from xiaomi_lens.constants import MASK_TOKEN
        from xiaomi_lens.model.modeling_xiaomi_lens import (
            XiaomiLensQwen3_5ForConditionalGeneration,
        )

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        # Transformers 5.x uses `dtype`; keep `torch_dtype` as a fallback for
        # older releases without changing the environment.
        load_kwargs = dict(device_map=str(self.device))
        try:
            self.model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                attn_implementation=self.requested_attn,
                **load_kwargs,
            )
            self.attn_implementation = self.requested_attn
        except (TypeError, ValueError, ImportError) as exc:
            print(f"[adapter] {self.requested_attn} failed ({exc}); using sdpa")
            self.model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                **load_kwargs,
            )
            self.attn_implementation = "sdpa"

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.think_close_id = self.processor.tokenizer.convert_tokens_to_ids(THINK_CLOSE)
        self.mask_token_id = self.processor.tokenizer.convert_tokens_to_ids(MASK_TOKEN)

        torch.cuda.synchronize(self.device)
        load_seconds = time.perf_counter() - start

        self.load_info = {
            "model_name": MODEL_NAME,
            "model_path": str(self.model_path),
            "attn_implementation": self.attn_implementation,
            "dtype": "bfloat16",
            "load_seconds": load_seconds,
            "vram_after_load": self._vram_snapshot(),
            "eos_token_ids": self._eos_ids(),
            "think_close_token_id": self.think_close_id,
            "mask_token_id": self.mask_token_id,
        }
        return self.load_info

    # ---------------------------------------------------------------- inputs
    def build_inputs(
        self, image: Image.Image, instruction: str, mode: str
    ) -> dict[str, Any]:
        """Render the chat template and run the processor.

        Returns the tensors plus the rendered prompt and a mode-verification
        flag (the rendered suffix must match the requested Thinking mode).
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

    # ---------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(
        self,
        prepared: dict[str, Any],
        mode: str,
        max_new_tokens: int,
        reset_peak: bool = True,
    ) -> dict[str, Any]:
        """Greedy generation with synchronized timing and per-sample VRAM peak."""
        inputs = prepared["inputs"]
        prompt_length = inputs["input_ids"].shape[1]

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
            completion_ids[0], mode
        )
        last_id = int(completion_ids[0, -1]) if generated_tokens else None
        if last_id is not None and last_id in self._eos_ids():
            finish_reason = "eos"
        elif generated_tokens >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "other"

        mask_tokens = 0
        if self.mask_token_id is not None:
            mask_tokens = int((completion_ids == self.mask_token_id).sum().item())

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
            "mask_tokens": mask_tokens,
            "vram": self._vram_snapshot(),
        }

    # ---------------------------------------------------------------- helpers
    def _split_token_counts(self, completion_ids: torch.Tensor, mode: str):
        """Count Thinking vs answer tokens using the exact ``</think>`` token id."""
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
        """Return PyTorch allocator and whole-GPU memory in GiB."""
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
