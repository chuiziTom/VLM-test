#!/usr/bin/env python3
"""下载 VSI-Bench 数据集到 dataset/VSI-Bench/。

对应原 Notebook：dataset/加载&下载数据集.ipynb 的第 3 部分（VSI-Bench）。

原 Notebook 对 VSI-Bench 只做了一件事：用 ``snapshot_download`` 把
``nyu-visionx/VSI-Bench`` 这个 Hugging Face 数据集仓库整体下载到本地。
没有解压 zip、没有抽取视频帧、没有生成样例图片或标注文件。
本脚本严格保留这一范围，只负责下载和下载结果检查。

仓库中通常包含（以官方 README 为准）：
- ``test.jsonl``：全部问题标注；
- ``test_debiased.parquet`` / ``test_pruned.parquet``：debiased / pruned 子集标注；
- ``arkitscenes.zip`` / ``scannet.zip`` / ``scannetpp.zip``：视频文件；
- ``pruned_ids.txt``、``create_pq.py``、``README.md`` 等辅助文件。

如果需要使用 full / debiased / pruned 配置或解压视频，请在这些文件下载完成后
自行处理；本脚本不改变原 Notebook 的处理范围。

运行示例：
    micromamba run -n vlm-test python dataset/download_vsi_bench.py
    # 只检查本地已有文件，不联网：
    micromamba run -n vlm-test python dataset/download_vsi_bench.py --skip-download

注意：import 本模块不会触发下载，只有直接运行脚本才会执行。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

# 默认路径以脚本自身位置为基准，不依赖运行时的当前工作目录。
DATASET_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "VSI-Bench"

HF_REPO_ID = "nyu-visionx/VSI-Bench"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"  # 原 Notebook 使用的镜像地址
DEFAULT_MAX_WORKERS = 4                         # 与原 Notebook 一致

# 下载后重点检查的文件（不强制存在，只用于给出提示）。
KEY_FILES = (
    "test.jsonl",
    "test_debiased.parquet",
    "test_pruned.parquet",
    "arkitscenes.zip",
    "scannet.zip",
    "scannetpp.zip",
    "pruned_ids.txt",
    "README.md",
)


# ---------------------------------------------------------------------------
# Hugging Face 下载
# ---------------------------------------------------------------------------

def configure_hf_endpoint(endpoint: str | None) -> None:
    """设置 Hugging Face 镜像地址。

    必须在导入 ``huggingface_hub`` 之前调用，因为该库在 import 时就会读取
    ``HF_ENDPOINT`` 并缓存到 ``constants.ENDPOINT``。
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
    """打印实际生效的 Hugging Face 配置，便于排查镜像/代理问题。"""
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


def download_vsi_bench(
    output_dir: Path,
    endpoint: str | None = DEFAULT_HF_ENDPOINT,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Path:
    """把 VSI-Bench 仓库整体下载到 ``output_dir``，返回本地目录。

    为什么用 ``snapshot_download``：
    - 原 Notebook 就是这么做的，且该仓库包含标注和视频 zip，需要一次性取全；
    - 它支持断点续传和缓存校验，重复运行时已完整的文件不会再次下载；
    - 指定 ``local_dir`` 后文件按仓库结构落到本地目录，便于后续直接按路径读取。
    """
    configure_hf_endpoint(endpoint)

    # 延迟导入：确保 HF_ENDPOINT 在 huggingface_hub 初始化前生效。
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"下载 {HF_REPO_ID} -> {output_dir}")
    print("提示：该仓库包含视频 zip，体积较大，请预留足够磁盘空间")
    log_hf_environment()

    local_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
        max_workers=max_workers,
    )
    print(f"下载/校验完成：{local_path}")
    return Path(local_path)


# ---------------------------------------------------------------------------
# 下载结果检查（只读，不修改数据）
# ---------------------------------------------------------------------------

def report_files(output_dir: Path) -> None:
    """列出关键文件是否存在，方便确认下载是否完整。

    这个函数只读取文件名和大小，不加载、不解压、不改写任何数据。
    """
    print("\n关键文件检查：")
    if not output_dir.is_dir():
        print(f"    [目录不存在] {output_dir}")
        return

    for name in KEY_FILES:
        path = output_dir / name
        if path.is_file():
            size_mb = path.stat().st_size / 1_000_000
            print(f"    [存在] {name}  {size_mb:.1f} MB")
        else:
            print(f"    [缺失] {name}")

    print("\n说明：本脚本不解压 zip、不抽取视频帧、不生成样例。")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载 VSI-Bench 数据集（标注 + 视频 zip）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"VSI-Bench 数据目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"并发下载线程数（默认：{DEFAULT_MAX_WORKERS}）",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载，只检查本地已有文件",
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
    print("VSI-Bench 数据下载")
    print(f"输出目录：{args.output_dir}")
    print("=" * 60)

    if args.skip_download:
        print("已指定 --skip-download，跳过下载")
    else:
        download_vsi_bench(
            args.output_dir,
            endpoint=args.hf_endpoint,
            max_workers=args.max_workers,
        )

    report_files(args.output_dir)

    print("=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
