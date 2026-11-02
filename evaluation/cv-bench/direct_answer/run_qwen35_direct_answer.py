"""Qwen3.5-4B：CV-Bench Direct Answer（Thinking Off）评测入口。

用法：
    micromamba run -n vlm-test python \
        evaluation/direct_answer/run_qwen35_direct_answer.py --resume

评测代码位于 ``evaluation/qwen35_4b/``：本脚本导入其数据加载、
指标与模型适配器，不使用、也不修改其中的预测结果。
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
MODEL_DIR = EVAL_DIR / "qwen35_4b"

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
