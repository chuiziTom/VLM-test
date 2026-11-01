"""Qwen3.5-4B：CV-Bench Direct Answer（Thinking Off）评测入口。

用法：
    micromamba run -n vlm-test python \
        eval_cvbench_direct_answer/run_qwen35_direct_answer.py --resume

原有 ``eval_qwen35_cvbench/`` 目录只读：本脚本按文件路径导入其数据加载、
指标与模型适配器，不使用、也不修改其中的预测结果。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须在 CUDA 初始化前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "eval_qwen35_cvbench"

sys.path.insert(0, str(HERE))

from common.module_loader import load_module  # noqa: E402
from common.runner_core import run  # noqa: E402

adapter_mod = load_module("qwen35_adapter_direct", MODEL_DIR / "model_qwen35.py")

if __name__ == "__main__":
    raise SystemExit(run(
        model_key="qwen35_4b",
        model_name=adapter_mod.MODEL_NAME,
        model_dir=MODEL_DIR,
        adapter_cls=adapter_mod.Qwen35Adapter,
        default_model="Qwen/Qwen3.5-4B",
        default_attn="sdpa",
    ))
