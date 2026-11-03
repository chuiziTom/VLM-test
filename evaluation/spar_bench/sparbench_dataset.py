"""SPAR-Bench-Tiny 数据读取。

本模块只负责把本地 Parquet 读成统一的 record，不依赖 ``datasets`` 包，
也不做任何抽样或图片裁剪：

* 每条样本的图片按 Parquet 中的原始顺序全部保留（1~3 张）；
* 图片以原始字节保存，推理时才解码，从而单独统计图片读取耗时；
* 样本顺序严格等于 Parquet 行序，保证三种实验协议、两个模型看到同一顺序。
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

PARQUET_NAME = "test-00000-of-00001.parquet"


@dataclass
class SparSample:
    """一条 SPAR-Bench-Tiny 样本（图片仍为编码后的字节）。"""

    idx: int
    id: int
    img_type: str
    format_type: str
    task: str
    source: str
    question: str
    answer: str
    image_bytes: list[bytes] = field(default_factory=list)

    @property
    def uid(self) -> str:
        return f"spar-{self.id}"

    @property
    def n_images(self) -> int:
        return len(self.image_bytes)


def load_dataset(dataset_dir: str | Path) -> tuple[list[SparSample], Path]:
    """读取 SPAR-Bench-Tiny 的单个 Parquet 文件，返回样本列表与文件路径。"""
    dataset_dir = Path(dataset_dir)
    parquet_path = dataset_dir / "data" / PARQUET_NAME
    if not parquet_path.exists():
        # 允许调用方直接传 data/ 目录。
        parquet_path = dataset_dir / PARQUET_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(f"找不到 SPAR-Bench-Tiny Parquet：{parquet_path}")

    table = pq.read_table(parquet_path)
    columns = table.to_pydict()

    required = ["id", "img_type", "format_type", "task", "source",
                "image", "question", "answer"]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Parquet 缺少字段：{missing}")

    samples: list[SparSample] = []
    for idx in range(table.num_rows):
        images = columns["image"][idx] or []
        # 每张图片是 struct<bytes, path>；只取 bytes，不解码。
        image_bytes = [item["bytes"] for item in images]
        samples.append(
            SparSample(
                idx=idx,
                id=int(columns["id"][idx]),
                img_type=str(columns["img_type"][idx]),
                format_type=str(columns["format_type"][idx]),
                task=str(columns["task"][idx]),
                source=str(columns["source"][idx]),
                question=str(columns["question"][idx]),
                answer=str(columns["answer"][idx]),
                image_bytes=image_bytes,
            )
        )
    return samples, parquet_path


def decode_images(sample: SparSample) -> tuple[list[Image.Image], float]:
    """按原始顺序解码全部图片，返回 (图片列表, 解码耗时秒)。

    Multi-view 样本必须全部解码并送入模型，不能只取第一张。
    """
    start = time.perf_counter()
    images = []
    for raw in sample.image_bytes:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            images.append(image.convert("RGB"))
    return images, time.perf_counter() - start


def select_samples(
    samples: list[SparSample],
    limit: int | None = None,
    stratified: bool = False,
) -> list[SparSample]:
    """按需截取样本；默认保持原始行序。

    ``stratified=True`` 时按 task 轮转取样，尽量让小规模冒烟测试覆盖 20 种任务。
    """
    if limit is None or limit >= len(samples):
        return list(samples)
    if not stratified:
        return list(samples[:limit])

    buckets: dict[str, list[SparSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.task, []).append(sample)

    picked: list[SparSample] = []
    keys = sorted(buckets)
    cursor = 0
    while len(picked) < limit:
        added = False
        for key in keys:
            bucket = buckets[key]
            if cursor < len(bucket) and len(picked) < limit:
                picked.append(bucket[cursor])
                added = True
        cursor += 1
        if not added:
            break
    # 恢复原始顺序，确保不同协议 / 模型之间的样本顺序完全一致。
    order = {sample.idx: pos for pos, sample in enumerate(samples)}
    return sorted(picked, key=lambda s: order[s.idx])
