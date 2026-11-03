"""Qwen3.5-4B 的 SPAR-Bench 适配器。

使用 Transformers 原生 ``Qwen3_5ForConditionalGeneration`` + ``AutoProcessor``，
注意力后端使用该环境已验证的 SDPA；同样支持一次输入 1~3 张图片。
"""

from __future__ import annotations

import torch

from model_base import BaseSparAdapter


class SparQwen35Adapter(BaseSparAdapter):
    MODEL_NAME = "Qwen3.5-4B"
    MODEL_ID = "Qwen/Qwen3.5-4B"

    def _load_model(self):
        from transformers import Qwen3_5ForConditionalGeneration

        kwargs = dict(device_map=str(self.device))
        try:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                attn_implementation=self.requested_attn,
                **kwargs,
            )
            self.attn_implementation = self.requested_attn
        except (TypeError, ValueError, ImportError, RuntimeError) as exc:
            print(f"[adapter] {self.requested_attn} 不可用（{exc}），改用 sdpa")
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                **kwargs,
            )
            self.attn_implementation = "sdpa"
