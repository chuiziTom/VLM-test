"""按文件路径加载原有评测模块，避免修改或复制原脚本。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module(name: str, path: Path) -> ModuleType:
    """从指定文件加载模块，并以独立名字注册，避免两个模型的同名模块冲突。"""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"模块不存在: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
