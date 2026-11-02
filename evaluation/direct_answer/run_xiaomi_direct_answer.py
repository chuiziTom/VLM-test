"""Xiaomi-Lens-0：CV-Bench Direct Answer（Thinking Off）评测入口。

用法：
    micromamba run -n xiaomi-lens python \
        evaluation/direct_answer/run_xiaomi_direct_answer.py --resume

评测代码位于 ``evaluation/xiaomi_lens/``：本脚本导入
其数据加载、指标与模型适配器，不使用、也不修改其中的预测结果。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须在 CUDA 初始化前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent   # evaluation/direct_answer
EVAL_DIR = HERE.parent                   # evaluation
ROOT = EVAL_DIR.parent                   # 项目根目录
MODEL_DIR = EVAL_DIR / "xiaomi_lens"

sys.path.insert(0, str(HERE))

from common.module_loader import load_module  # noqa: E402
from common.runner_core import run  # noqa: E402

adapter_mod = load_module("xiaomi_lens_adapter_direct", MODEL_DIR / "xiaomi_lens_adapter.py")

if __name__ == "__main__":
    raise SystemExit(run(
        model_key="xiaomi_lens",
        model_name=adapter_mod.MODEL_NAME,
        model_dir=MODEL_DIR,
        adapter_cls=adapter_mod.XiaomiLensAdapter,
        default_model="SeerRay-Lab/Xiaomi-Lens-0",
        default_attn="flash_attention_2",
    ))
