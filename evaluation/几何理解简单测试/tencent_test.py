import sys
from pathlib import Path

# wevisone_vl's source lives in models/WeVisOne-VL and may not be installed
# into the current interpreter. Make it importable for this script only.
_MODEL_ROOT = Path(__file__).resolve().parents[2] / "models" / "WeVisOne-VL"
if not _MODEL_ROOT.is_dir():
    raise RuntimeError(f"cannot find WeVisOne-VL source at {_MODEL_ROOT}")
sys.path.insert(0, str(_MODEL_ROOT))

from PIL import Image
from wevisone_vl import WeVisOneVL

model = WeVisOneVL("tencent/WeVisOne-VL-4B", device="auto", dtype="bfloat16")
image = Image.open("/home/20261024_1/Workspace/qyh/VLM-test/datasets/CV-Bench/img/2D/000000.png").convert("RGB")
result = model.predict(image, task="matting", query="window", temperature=0.0)
print(result.response, len(result.masks))