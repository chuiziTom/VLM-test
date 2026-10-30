"""Accuracy, timing, token and VRAM statistics for CV-Bench runs.

Invalid predictions are *never* dropped: they stay in every denominator, as
required by the evaluation protocol.  The official CV-Bench score is computed
with the dataset's source-weighted formula, not as a plain sample average.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

OFFICIAL_2D_SOURCES = ("ADE20K", "COCO")
OFFICIAL_3D_SOURCE = "Omni3D"


def _pct(values: Iterable[float], q: float) -> float | None:
    values = list(values)
    return float(np.percentile(values, q)) if values else None


def _acc(records: list[dict]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    invalid = sum(1 for r in records if r.get("invalid"))
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_percent": 100.0 * correct / total if total else None,
        "invalid": invalid,
        "invalid_rate": invalid / total if total else None,
    }


def _group(records: list[dict], key_fn) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[str(key_fn(record))].append(record)
    return {key: _acc(bucket) for key, bucket in sorted(buckets.items())}


def _timing(records: list[dict], field: str) -> dict[str, Any]:
    values = [r[field] for r in records if isinstance(r.get(field), (int, float))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None, "sum": 0.0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": _pct(values, 95),
        "max": float(max(values)),
        "sum": float(sum(values)),
    }


def _tokens(records: list[dict], field: str) -> dict[str, Any]:
    values = [int(r[field]) for r in records if r.get(field) is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None, "sum": 0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": _pct(values, 95),
        "max": int(max(values)),
        "sum": int(sum(values)),
    }


def compute_metrics(records: list[dict], experiment: dict | None = None) -> dict[str, Any]:
    """Compute all required metrics from per-sample prediction records."""
    records = list(records)
    n = len(records)

    # Official CV-Bench formula: source-weighted 2D, Omni3D for 3D, then 50/50.
    by_type_source = _group(records, lambda r: f"{r['type']}/{r['source']}")
    acc_ade = by_type_source.get("2D/ADE20K", {}).get("accuracy")
    acc_coco = by_type_source.get("2D/COCO", {}).get("accuracy")
    acc_omni = by_type_source.get(f"3D/{OFFICIAL_3D_SOURCE}", {}).get("accuracy")
    official_2d = (
        (acc_ade + acc_coco) / 2
        if acc_ade is not None and acc_coco is not None else None
    )
    official_3d = acc_omni
    official_total = (
        (official_2d + official_3d) / 2
        if official_2d is not None and official_3d is not None else None
    )

    invalid_reasons = Counter(
        r.get("invalid_reason") for r in records if r.get("invalid")
    )
    finish_reasons = Counter(r.get("finish_reason") for r in records)

    gen_time = _timing(records, "generate_seconds")
    e2e_time = _timing(records, "e2e_seconds")
    total_tokens = _tokens(records, "generated_tokens")
    peak_vram = [
        r["vram"]["peak_allocated_gib"] for r in records
        if isinstance(r.get("vram"), dict) and "peak_allocated_gib" in r["vram"]
    ]

    metrics = {
        "total_samples": n,
        "correct": sum(1 for r in records if r.get("correct")),
        "accuracy": _acc(records)["accuracy"],
        "accuracy_percent": _acc(records)["accuracy_percent"],
        "valid_answers": sum(1 for r in records if not r.get("invalid")),
        "valid_rate": (
            sum(1 for r in records if not r.get("invalid")) / n if n else None
        ),
        "invalid": sum(1 for r in records if r.get("invalid")),
        "invalid_rate": (
            sum(1 for r in records if r.get("invalid")) / n if n else None
        ),
        "truncated": sum(1 for r in records if r.get("truncated")),
        "truncation_rate": (
            sum(1 for r in records if r.get("truncated")) / n if n else None
        ),
        "errors": sum(1 for r in records if r.get("error")),
        "error_rate": sum(1 for r in records if r.get("error")) / n if n else None,
        "official": {
            "accuracy_2d_ade20k": acc_ade,
            "accuracy_2d_coco": acc_coco,
            "accuracy_2d": official_2d,
            "accuracy_3d_omni3d": acc_omni,
            "accuracy_3d": official_3d,
            "cvbench_accuracy": official_total,
            "cvbench_accuracy_percent": (
                100.0 * official_total if official_total is not None else None
            ),
            "computable": official_total is not None,
        },
        "plain_sample_accuracy": _acc(records)["accuracy"],
        "groups": {
            "by_type": _group(records, lambda r: r["type"]),
            "by_task": _group(records, lambda r: r["task"]),
            "by_source": _group(records, lambda r: r["source"]),
            "by_type_task": _group(records, lambda r: f"{r['type']}/{r['task']}"),
            "by_type_task_source": _group(
                records, lambda r: f"{r['type']}/{r['task']}/{r['source']}"
            ),
        },
        "invalid_reasons": dict(invalid_reasons),
        "finish_reasons": dict(finish_reasons),
        "timing_seconds": {
            "model_load": experiment.get("model_load_seconds") if experiment else None,
            "warmup": experiment.get("warmup_seconds") if experiment else None,
            "preprocess": _timing(records, "preprocess_seconds"),
            "image_load": _timing(records, "image_load_seconds"),
            "generate": gen_time,
            "end_to_end": e2e_time,
        },
        "throughput": {
            "samples_per_second_generate": (
                n / gen_time["sum"] if gen_time["sum"] else None
            ),
            "samples_per_second_e2e": (
                n / e2e_time["sum"] if e2e_time["sum"] else None
            ),
            "tokens_per_second_generate": (
                total_tokens["sum"] / gen_time["sum"] if gen_time["sum"] else None
            ),
            "note": (
                "tokens/s includes prefill + decode; it is not pure decode speed"
            ),
        },
        "tokens": {
            "generated": total_tokens,
            "thinking": _tokens(records, "thinking_tokens"),
            "answer": _tokens(records, "answer_tokens"),
            "prompt": _tokens(records, "prompt_tokens"),
        },
        "vram": {
            "model_load": (
                experiment.get("vram_after_load") if experiment else None
            ),
            "per_sample_peak_allocated_gib_mean": (
                float(np.mean(peak_vram)) if peak_vram else None
            ),
            "per_sample_peak_allocated_gib_max": (
                float(max(peak_vram)) if peak_vram else None
            ),
            "global_peak_allocated_gib": (
                experiment.get("global_peak_allocated_gib") if experiment else None
            ),
            "global_peak_reserved_gib": (
                experiment.get("global_peak_reserved_gib") if experiment else None
            ),
            "gpu_used_gib_end": (
                experiment.get("gpu_used_gib_end") if experiment else None
            ),
        },
        "subset": n != 2638,
        "experiment": experiment or {},
    }
    return metrics


def compare_runs(off_records: list[dict], on_records: list[dict]) -> dict[str, Any]:
    """Sample-level and task-level comparison between two thinking modes."""
    off_by_uid = {r["uid"]: r for r in off_records}
    on_by_uid = {r["uid"]: r for r in on_records}
    common = sorted(set(off_by_uid) & set(on_by_uid))

    off_right_on_wrong = sum(
        1 for uid in common
        if off_by_uid[uid].get("correct") and not on_by_uid[uid].get("correct")
    )
    off_wrong_on_right = sum(
        1 for uid in common
        if not off_by_uid[uid].get("correct") and on_by_uid[uid].get("correct")
    )
    prediction_changed = sum(
        1 for uid in common
        if off_by_uid[uid].get("prediction") != on_by_uid[uid].get("prediction")
    )

    def per_task(records, task):
        subset = [r for r in records if r.get("task") == task]
        correct = sum(1 for r in subset if r.get("correct"))
        return correct, len(subset)

    tasks = sorted({r.get("task") for r in off_records + on_records})
    task_deltas = {}
    for task in tasks:
        off_correct, off_n = per_task(off_records, task)
        on_correct, on_n = per_task(on_records, task)
        task_deltas[task] = {
            "off": {"correct": off_correct, "n": off_n,
                    "accuracy": off_correct / off_n if off_n else None},
            "on": {"correct": on_correct, "n": on_n,
                   "accuracy": on_correct / on_n if on_n else None},
            "accuracy_delta": (
                on_correct / on_n - off_correct / off_n
                if off_n and on_n else None
            ),
        }

    return {
        "common_samples": len(common),
        "only_in_off": len(set(off_by_uid) - set(on_by_uid)),
        "only_in_on": len(set(on_by_uid) - set(off_by_uid)),
        "off_right_on_wrong": off_right_on_wrong,
        "off_wrong_on_right": off_wrong_on_right,
        "prediction_changed": prediction_changed,
        "task_deltas": task_deltas,
    }
