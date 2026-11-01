"""Xiaomi-Lens-0：CV-Bench Direct Answer（Thinking Off）评测入口。

用法：
    micromamba run -n xiaomi-lens python \
        eval_cvbench_direct_answer/run_xiaomi_direct_answer.py --resume

原有 ``Xiaomi-Lens-0/eval_xiaomi_cvbench/`` 目录只读：本脚本按文件路径导入
其数据加载、指标与模型适配器，不使用、也不修改其中的预测结果。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须在 CUDA 初始化前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "Xiaomi-Lens-0" / "eval_xiaomi_cvbench"

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
