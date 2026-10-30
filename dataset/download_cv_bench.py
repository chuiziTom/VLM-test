#!/usr/bin/env python3
"""下载 CV-Bench 数据集，并提取 2D / 3D 子集的样例图片与标注。

对应原 Notebook：dataset/加载&下载数据集.ipynb 的第 1 部分（CV-Bench）。

脚本做两件事：
1. 从 Hugging Face 下载 ``nyu-visionx/CV-Bench`` 到 ``dataset/CV-Bench/``；
2. 读取 ``test_2d.parquet`` / ``test_3d.parquet``，各自按 ``filename`` 去重后
   随机抽取 30 张图片，保存原始图片、5 列总览图和 ``annotations.csv``。

原 Notebook 中的处理逻辑保持不变：
- 2D / 3D 分别处理，不缩减任何子集；
- ``drop_duplicates(subset="filename")`` 后 ``sample(n=30, random_state=42)``；
- 图片输出到 ``samples/2D``、``samples/3D``；
- 总览图为 ``samples/2d_overview.jpg``、``samples/3d_overview.jpg``；
- ``annotations.csv`` 只保留 image_file / task / question / answer 四列。

运行示例：
    micromamba run -n vlm-test python dataset/download_cv_bench.py
    # 数据已下载好，只重新提取样例并覆盖旧样例：
    micromamba run -n vlm-test python dataset/download_cv_bench.py --skip-download --force

注意：import 本模块不会下载或处理任何数据，只有直接运行脚本才会执行。
"""

from __future__ import annotations

import argparse
import io
import math
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

# 默认路径以脚本自身位置为基准，避免依赖运行时的当前工作目录。
# 本脚本位于 dataset/ 下，因此 DATASET_ROOT 就是 dataset/ 目录。
DATASET_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "CV-Bench"

HF_REPO_ID = "nyu-visionx/CV-Bench"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"  # 原 Notebook 使用的镜像地址

DEFAULT_SAMPLES = 30  # 原 Notebook 每个子集抽取 30 张
DEFAULT_SEED = 42     # 原 Notebook 的随机种子

SPLITS = ("2d", "3d")

# 总览图的排版参数：5 列 × 每行 220 像素；30 张图正好是 5×6，1300×1320 像素。
COLS = 5
CELL_WIDTH = 260
CELL_HEIGHT = 220
THUMB_SIZE = (240, 185)
TEXT_OFFSET_Y = 190


# ---------------------------------------------------------------------------
# Hugging Face 下载
# ---------------------------------------------------------------------------

def configure_hf_endpoint(endpoint: str | None) -> None:
    """设置 Hugging Face 镜像地址。

    必须在导入 ``huggingface_hub`` 之前调用：该库在 import 时就把
    ``HF_ENDPOINT`` 读入 ``constants.ENDPOINT``，之后再改环境变量不生效。
    传入空字符串表示不改写环境变量（使用官方地址或 shell 中已有的设置）。
    """
    if not endpoint:
        print("    HF_ENDPOINT 未改写，使用 Hugging Face 官方地址或现有环境变量")
        return
    existing = os.environ.get("HF_ENDPOINT")
    if existing and existing != endpoint:
        print(f"    环境变量 HF_ENDPOINT={existing} 将被覆盖为 {endpoint}")
    os.environ["HF_ENDPOINT"] = endpoint


def log_hf_environment() -> None:
    """打印实际生效的 Hugging Face 配置，便于排查镜像/代理问题。

    对应原 Notebook 中检查 HF_ENDPOINT、Hub 版本和代理变量的单元格。
    """
    import huggingface_hub
    from huggingface_hub import constants

    print(f"    HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
    print(f"    实际使用端点 = {constants.ENDPOINT}")
    print(f"    huggingface_hub 版本 = {huggingface_hub.__version__}")
    proxy_keys = [
        "HTTP_PROXY", "HTTPS_PROXY",
        "http_proxy", "https_proxy",
        "ALL_PROXY", "all_proxy",
    ]
    configured = [k for k in proxy_keys if os.environ.get(k)]
    print(f"    代理变量 = {'、'.join(configured) if configured else '未配置'}")


def download_cv_bench(output_dir: Path, endpoint: str | None = DEFAULT_HF_ENDPOINT) -> Path:
    """下载 CV-Bench 数据到 ``output_dir``，返回本地目录。

    使用 ``snapshot_download`` 的原因：
    - 它是 Hugging Face 官方推荐的整体仓库下载方式，适合该数据集
      （7 个文件，包含 README、parquet 和图片构建脚本）；
    - 内部会复用本地缓存并校验文件，重复运行不会重复下载已完整的文件。

    原 Notebook 未指定 ``max_workers``，这里同样使用默认值。
    """
    configure_hf_endpoint(endpoint)

    # 延迟导入：确保上面的 HF_ENDPOINT 在 huggingface_hub 初始化前生效。
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/2] 下载 {HF_REPO_ID} -> {output_dir}")
    log_hf_environment()

    local_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    print(f"    下载/校验完成：{local_path}")
    return Path(local_path)


# ---------------------------------------------------------------------------
# 样例提取
# ---------------------------------------------------------------------------

def load_image_from_row(row: pd.Series) -> Image.Image:
    """从 parquet 的 ``image`` 字段还原 PIL 图片。

    该字段是 Hugging Face 图片结构，``bytes`` 子字段保存原始文件内容。
    如果 parquet 只保存了路径而没有内嵌字节，这里给出明确错误而不是静默失败。
    """
    image_field = row["image"]
    img_bytes = image_field["bytes"]
    if img_bytes is None:
        raise ValueError(
            "image.bytes 为空，无法还原图片。请确认下载的是包含内嵌图片的 "
            "test_2d.parquet / test_3d.parquet。"
        )
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _split_already_extracted(annotations_path: Path, overview_path: Path) -> bool:
    """判断该子集的样例是否已经完整生成（标注和总览图都在）。"""
    return annotations_path.exists() and overview_path.exists()


def extract_samples(
    output_dir: Path,
    num_samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> None:
    """为 2D / 3D 两个子集分别提取样例图片和标注。

    Args:
        output_dir: CV-Bench 数据目录（包含 test_2d.parquet 等文件）。
        num_samples: 每个子集抽取的图片数量，默认 30（与原 Notebook 一致）。
        seed: 随机抽样种子，默认 42（与原 Notebook 一致）。
        force: 为 True 时即使已有样例也重新生成；默认跳过已有结果。
    """
    if num_samples < 1:
        raise ValueError(f"--samples 必须为正整数，当前为 {num_samples}")

    print(f"[2/2] 提取样例：每个子集 {num_samples} 张，random_state={seed}")
    samples_root = output_dir / "samples"

    for split in SPLITS:
        parquet_path = output_dir / f"test_{split}.parquet"
        save_dir = samples_root / split.upper()
        annotations_path = save_dir / "annotations.csv"
        overview_path = samples_root / f"{split}_overview.jpg"

        if not force and _split_already_extracted(annotations_path, overview_path):
            print(f"    [跳过] {split.upper()} 样例已存在：{save_dir}")
            print("           如需重新生成，请加 --force")
            continue

        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"未找到 {parquet_path}。请先去掉 --skip-download 重新运行以完成下载，"
                "或检查 --output-dir 是否指向正确的 CV-Bench 目录。"
            )

        # 读取 Parquet，去除重复图片后随机抽取。
        df = pd.read_parquet(parquet_path)
        df = df.drop_duplicates(subset="filename")
        if len(df) < num_samples:
            raise ValueError(
                f"{parquet_path.name} 去重后只有 {len(df)} 条样本，"
                f"少于请求的 {num_samples} 条。请用 --samples 调小数量。"
            )
        samples = df.sample(n=num_samples, random_state=seed)

        save_dir.mkdir(parents=True, exist_ok=True)

        # 创建 5 列总览图；30 张图片时尺寸为 1300×1320，与 Notebook 相同。
        rows = math.ceil(num_samples / COLS)
        sheet = Image.new("RGB", (COLS * CELL_WIDTH, rows * CELL_HEIGHT), "white")
        draw = ImageDraw.Draw(sheet)

        records: list[dict[str, object]] = []

        for i, (_, row) in enumerate(samples.iterrows()):
            img = load_image_from_row(row)

            # 保存原始分辨率图片，命名沿用原 Notebook 的 2d_01.png / 3d_01.png。
            image_filename = f"{split}_{i + 1:02d}.png"
            img.save(save_dir / image_filename)

            # 生成缩略图并放到总览图对应格子中。
            thumb = ImageOps.contain(img, THUMB_SIZE)
            x = (i % COLS) * CELL_WIDTH
            y = (i // COLS) * CELL_HEIGHT
            sheet.paste(thumb, (x + (CELL_WIDTH - thumb.width) // 2, y))
            draw.text(
                (x + 10, y + TEXT_OFFSET_Y),
                f"{split.upper()} #{i + 1:02d}  {row['task']}",
                fill="black",
            )

            records.append({
                "image_file": image_filename,
                "task": row["task"],
                "question": row["question"],
                "answer": row["answer"],
            })

        # 总览图和对应题目：路径与 Notebook 完全一致。
        sheet.save(overview_path)
        pd.DataFrame(records).to_csv(annotations_path, index=False)

        print(f"    {split.upper()}：已保存 {len(samples)} 张图片 -> {save_dir}")
        print(f"           总览图 -> {overview_path}")

    print("    样例提取完成")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载 CV-Bench 并提取 2D/3D 样例图片与标注",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"CV-Bench 数据目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"每个子集抽取的图片数量（默认：{DEFAULT_SAMPLES}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机抽样种子（默认：{DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载，直接使用本地已有的 parquet 文件",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="已存在样例时也重新生成（默认跳过）",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help=(
            "Hugging Face 端点（默认镜像 https://hf-mirror.com；"
            "传空字符串则使用官方地址）"
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    print("=" * 60)
    print("CV-Bench 数据下载与样例提取")
    print(f"输出目录：{args.output_dir}")
    print("=" * 60)

    if args.skip_download:
        print("[1/2] 已指定 --skip-download，跳过下载")
    else:
        download_cv_bench(args.output_dir, endpoint=args.hf_endpoint)

    extract_samples(
        args.output_dir,
        num_samples=args.samples,
        seed=args.seed,
        force=args.force,
    )

    print("=" * 60)
    print("全部完成")
    print(f"样例目录：{args.output_dir / 'samples'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
