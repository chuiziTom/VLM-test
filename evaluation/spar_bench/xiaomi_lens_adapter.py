"""Xiaomi-Lens-0 的 SPAR-Bench 适配器。

复用官方自定义模型类 ``XiaomiLensQwen3_5ForConditionalGeneration`` 与
checkpoint 自带 processor；只把 CV-Bench 的单图入口改成多图入口。
优先 Flash Attention 2，不可用时回退 SDPA 并记录真实生效的后端。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from model_base import BaseSparAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
XIAOMI_SRC = REPO_ROOT / "models" / "Xiaomi-Lens-0"


class SparXiaomiLensAdapter(BaseSparAdapter):
    MODEL_NAME = "Xiaomi-Lens-0"
    MODEL_ID = "SeerRay-Lab/Xiaomi-Lens-0"

    def _load_model(self):
        sys.path.insert(0, str(XIAOMI_SRC / "src"))
        sys.path.insert(0, str(XIAOMI_SRC))
        from xiaomi_lens.model.modeling_xiaomi_lens import (
            XiaomiLensQwen3_5ForConditionalGeneration,
        )

        kwargs = dict(device_map=str(self.device))
        try:
            self.model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                attn_implementation=self.requested_attn,
                **kwargs,
            )
            self.attn_implementation = self.requested_attn
        except (TypeError, ValueError, ImportError, RuntimeError) as exc:
            print(f"[adapter] {self.requested_attn} 不可用（{exc}），改用 sdpa")
            self.model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                **kwargs,
            )
            self.attn_implementation = "sdpa"
