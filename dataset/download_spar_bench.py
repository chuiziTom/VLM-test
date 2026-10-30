#!/usr/bin/env python3
"""下载 SPAR-Bench 数据集到 dataset/SPAR-Bench/。

对应原 Notebook：dataset/加载&下载数据集.ipynb 的第 2 部分（SPAR-Bench）。

原 Notebook 对 SPAR-Bench 只做了一件事：用 ``snapshot_download`` 把
``jasonzhango/SPAR-Bench`` 整体下载到本地。没有读取 Parquet、没有按任务抽样、
没有解码 ``image`` 字段、没有生成总览图或 JSON。本脚本严格保留这一范围。

需要注意：Notebook 的 Markdown 介绍提到了 Tiny 和 RGB-D 版本，但实际代码
下载的是 **完整 RGB 版** ``jasonzhango/SPAR-Bench``（config 为 ``default``，
数据文件位于仓库的 ``data/test-*``）。如需 Tiny 或 RGB-D 版本，需要改用
对应的仓库 ID，本脚本没有擅自替换。

运行示例：
    micromamba run -n vlm-test python dataset/download_spar_bench.py
    # 只检查本地已有文件，不联网：
    micromamba run -n vlm-test python dataset/download_spar_bench.py --skip-download

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
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "SPAR-Bench"

HF_REPO_ID = "jasonzhango/SPAR-Bench"           # 原 Notebook 使用的完整 RGB 版本
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"   # 原 Notebook 使用的镜像地址
DEFAULT_MAX_WORKERS = 4                          # 与原 Notebook 一致


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


def download_spar_bench(
    output_dir: Path,
    endpoint: str | None = DEFAULT_HF_ENDPOINT,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Path:
    """把 SPAR-Bench 仓库整体下载到 ``output_dir``，返回本地目录。

    为什么用 ``snapshot_download``：
    - 原 Notebook 就是这么做的，仓库中的 Parquet 分片需要整体取回；
    - 支持断点续传与缓存校验，重复运行不会重复下载已完整的文件；
    - ``local_dir`` 会保留仓库目录结构（如 ``data/test-*.parquet``），
      后续读取时可以直接按路径定位。
    """
    configure_hf_endpoint(endpoint)

    # 延迟导入：确保 HF_ENDPOINT 在 huggingface_hub 初始化前生效。
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"下载 {HF_REPO_ID} -> {output_dir}")
    print("提示：完整版本包含全部 Parquet 分片，体积较大，请预留磁盘空间")
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

def report_download(output_dir: Path) -> None:
    """检查 README 和 Parquet 分片是否已下载。

    只读取文件名和大小，不加载 Parquet、不解码图片、不生成任何输出。
    """
    print("\n下载结果检查：")
    if not output_dir.is_dir():
        print(f"    [目录不存在] {output_dir}")
        return

    readme = output_dir / "README.md"
    print(f"    README.md：{'存在' if readme.is_file() else '缺失'}")

    data_dir = output_dir / "data"
    parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.is_dir() else []
    if parquet_files:
        total_mb = sum(p.stat().st_size for p in parquet_files) / 1_000_000
        print(f"    data/ 下 Parquet 分片：{len(parquet_files)} 个，共 {total_mb:.1f} MB")
        for path in parquet_files[:3]:
            print(f"        - {path.name}")
        if len(parquet_files) > 3:
            print(f"        ... 其余 {len(parquet_files) - 3} 个文件未列出")
    else:
        print("    data/ 下未找到 Parquet 分片（下载可能未完成）")

    print("\n说明：本脚本不解析 Parquet、不按任务抽样、不保存图片或标注。")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载 SPAR-Bench 数据集（完整 RGB 版本）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"SPAR-Bench 数据目录（默认：{DEFAULT_OUTPUT_DIR}）",
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
    print("SPAR-Bench 数据下载")
    print(f"输出目录：{args.output_dir}")
    print("=" * 60)

    if args.skip_download:
        print("已指定 --skip-download，跳过下载")
    else:
        download_spar_bench(
            args.output_dir,
            endpoint=args.hf_endpoint,
            max_workers=args.max_workers,
        )

    report_download(args.output_dir)

    print("=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
