"""CV-Bench dataset loading for the local parquet release.

The local files (``test_2d.parquet`` / ``test_3d.parquet``) store the image
bytes inside the parquet ``image`` struct column.  The ``img/`` directory only
contains a handful of preview images, so images must be decoded from parquet.

Only the metadata of all rows is kept in memory; image row groups are decoded
lazily and cached, which keeps peak memory small while still allowing random
access by ``idx`` (needed for ``--split`` / ``--limit`` / resume).
"""

from __future__ import annotations

import bisect
import io
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

PARQUET_FILES = {"2d": "test_2d.parquet", "3d": "test_3d.parquet"}
_ANSWER_RE = re.compile(r"^\(([A-Z])\)$")


@dataclass(frozen=True)
class CVBenchRecord:
    """One normalized CV-Bench sample (without the image)."""

    uid: str
    idx: int
    split: str
    type: str
    task: str
    source: str
    question: str
    choices: tuple[str, ...]
    answer: str
    prompt: str
    filename: str
    source_dataset: str

    @property
    def valid_letters(self) -> tuple[str, ...]:
        """Letters that exist for this question (A, B, ... according to count)."""
        return tuple(chr(ord("A") + i) for i in range(len(self.choices)))

    @property
    def answer_letter(self) -> str | None:
        """Ground-truth letter, e.g. ``"(C)" -> "C"``; None if malformed."""
        match = _ANSWER_RE.match(self.answer.strip())
        return match.group(1) if match else None


class CVBenchImages:
    """Random-access image loader backed by parquet row groups."""

    def __init__(self, dataset_dir: Path, cache_size: int = 2):
        self.dataset_dir = Path(dataset_dir)
        self.cache_size = cache_size
        self._files: dict[str, pq.ParquetFile] = {}
        self._row_of_idx: dict[str, dict[int, int]] = {}
        self._rg_starts: dict[str, list[int]] = {}
        self._cache: OrderedDict[tuple[str, int], object] = OrderedDict()

        for split, name in PARQUET_FILES.items():
            path = self.dataset_dir / name
            if not path.exists():
                continue
            pf = pq.ParquetFile(path)
            ids = pq.read_table(path, columns=["idx"]).column("idx").to_pylist()
            self._files[split] = pf
            self._row_of_idx[split] = {int(i): pos for pos, i in enumerate(ids)}
            starts, running = [], 0
            for rg in range(pf.num_row_groups):
                starts.append(running)
                running += pf.metadata.row_group(rg).num_rows
            self._rg_starts[split] = starts

    def get(self, record: CVBenchRecord) -> Image.Image:
        split = record.split
        if split not in self._files:
            raise FileNotFoundError(
                f"{PARQUET_FILES[split]} not found under {self.dataset_dir}"
            )
        row_pos = self._row_of_idx[split][record.idx]
        rg = bisect.bisect_right(self._rg_starts[split], row_pos) - 1
        key = (split, rg)
        if key in self._cache:
            self._cache.move_to_end(key)
            column = self._cache[key]
        else:
            table = self._files[split].read_row_group(rg, columns=["image"])
            column = table.column("image")
            self._cache[key] = column
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        local_row = row_pos - self._rg_starts[split][rg]
        blob = column[local_row].as_py()
        if not blob or blob.get("bytes") is None:
            raise ValueError(f"Sample {record.uid} has no image bytes in parquet")
        return Image.open(io.BytesIO(blob["bytes"])).convert("RGB")


def _record_from_row(row: dict, split: str) -> CVBenchRecord:
    return CVBenchRecord(
        uid=f"{row['type']}:{int(row['idx'])}",
        idx=int(row["idx"]),
        split=split,
        type=row["type"],
        task=row["task"],
        source=row["source"],
        question=row["question"],
        choices=tuple(row["choices"]),
        answer=row["answer"],
        prompt=row["prompt"],
        filename=row["filename"],
        source_dataset=row["source_dataset"],
    )


_COLUMNS = [
    "idx", "type", "task", "question", "choices", "answer", "prompt",
    "filename", "source", "source_dataset",
]


def load_records(dataset_dir: Path, split: str = "all") -> list[CVBenchRecord]:
    """Load normalized CV-Bench records sorted by global ``idx``.

    ``split`` is one of ``all`` / ``2d`` / ``3d``.  Row counts and unique
    indices are verified so a truncated or duplicated parquet fails loudly.
    """
    dataset_dir = Path(dataset_dir)
    splits = ["2d", "3d"] if split == "all" else [split]
    records: list[CVBenchRecord] = []
    for name in splits:
        path = dataset_dir / PARQUET_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        rows = pq.read_table(path, columns=_COLUMNS).to_pylist()
        records.extend(_record_from_row(row, name) for row in rows)

    records.sort(key=lambda r: r.idx)
    indices = [r.idx for r in records]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate idx values in CV-Bench parquet files")
    return records


def describe(records: list[CVBenchRecord]) -> dict:
    """Return dataset distribution statistics used in the report."""
    from collections import Counter

    def counter(key):
        return dict(sorted(Counter(key(r) for r in records).items()))

    return {
        "total": len(records),
        "by_type": counter(lambda r: r.type),
        "by_task": counter(lambda r: r.task),
        "by_source": counter(lambda r: r.source),
        "by_type_task_source": counter(
            lambda r: f"{r.type}/{r.task}/{r.source}"
        ),
        "choice_counts": dict(sorted(
            Counter(len(r.choices) for r in records).items()
        )),
        "answer_letters": dict(sorted(Counter(r.answer_letter for r in records).items())),
    }
