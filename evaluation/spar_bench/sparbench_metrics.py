"""SPAR-Bench 官方评分规则与汇总。

官方实现（逐行对照，未做简化）：
https://github.com/LogosIntelligenceLab/SPAR/blob/main/lmms-eval/lmms_eval/tasks/sparbench_tiny/utils.py

三类评分：
* MCA（选择题）：``exact_match``，包含子串匹配与首字符回退；
* NA（数值题）：先用 ``process_na`` 抽数，再算 ``mean_relative_accuracy``；
* VCI（视角变化）：``compute_vci_metric``，对 5 个方向的差值各算一次 MRA。

官方总分 ``sparbench_aggregate_results`` 先算 20 个任务各自的分数，
再对任务分数做算术平均；本模块保持同样的聚合方式，绝不用全样本池化
accuracy 代替官方总分。

另外实现 Strict 指标作为补充：对最终回答做严格格式解析（不做模糊匹配、
不在自由文本里搜索答案），解析失败记 0 并标记 invalid。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------- 官方任务表
MCA_QUESTION_TYPES = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]

NA_QUESTION_TYPES = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
]

SPECIAL_QUESTION_TYPES = ["view_change_infer"]

LOW_TASKS = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
]

MIDDLE_TASKS = [
    "view_change_infer",
    "position_matching",
    "camera_motion_infer",
]

HIGH_TASKS = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]

ALL_TASKS = LOW_TASKS + MIDDLE_TASKS + HIGH_TASKS

TASK_LEVEL = {
    **{task: "low" for task in LOW_TASKS},
    **{task: "medium" for task in MIDDLE_TASKS},
    **{task: "high" for task in HIGH_TASKS},
}


def task_level(task: str) -> str:
    """返回任务所属层级；未知任务直接报错，避免静默归入某一类。"""
    try:
        return TASK_LEVEL[task]
    except KeyError as exc:
        raise ValueError(f"未知 SPAR-Bench 任务：{task}") from exc


def answer_type(task: str) -> str:
    """返回任务使用的评分方式：MCA / NA / VCI。"""
    if task in MCA_QUESTION_TYPES:
        return "MCA"
    if task in NA_QUESTION_TYPES:
        return "NA"
    if task == "view_change_infer":
        return "VCI"
    raise ValueError(f"未知 SPAR-Bench 任务：{task}")


# ------------------------------------------------------- 官方函数：逐行复现
def fuzzy_matching(pred: str) -> str:
    """官方辅助函数，原样保留（当前评分流程未使用）。"""
    return pred.split(" ")[0].rstrip(".").strip()


def process_na(pred: str, task: str):
    """官方数值抽取。

    * 普通（单视角）数值题取**第一个**数字；
    * ``*_mv`` 数值题取**最后一个**数字（多视图问题里可能先复述已知值）；
    * mv 且没有数字时返回 -1，交给 MRA 记最差分数；普通题无数字会抛异常，
      官方在上层捕获后记 0。
    """
    numbers = re.findall(r"(?<!\^)\d+\.\d+|(?<!\^)\d+", pred)
    extracted_numbers = [float(num) if "." in num else int(num) for num in numbers]
    if task in [
        "depth_prediction_oc_mv",
        "depth_prediction_oo_mv",
        "distance_prediction_oc_mv",
        "distance_prediction_oo_mv",
    ]:
        if len(extracted_numbers) == 0:
            extracted_numbers = [-1]
        extracted_numbers = [extracted_numbers[-1]]
    return extracted_numbers[0]


def parse_instruction(instruction: str) -> dict[str, float]:
    """官方 VCI 参数解析：``key:value,key:value,...``。

    任何一段缺少冒号都会抛异常，官方将整条样本记 0。
    """
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}


def compute_vci_metric(pred: str, answer: str) -> float:
    """官方 VCI 指标：5 个方向差值各算一次 MRA，再取均值。"""
    action_order = [
        "move_right_left",
        "move_up_down",
        "move_forward_backward",
        "rotate_right_left",
        "rotate_up_down",
    ]

    answer_dict = parse_instruction(pred)
    gt_dict = parse_instruction(answer)

    answer_list = []
    gt_list = []
    for action_pair in action_order:
        if action_pair == "move_right_left":
            answer_list.append(answer_dict.get("move_right", 0) - answer_dict.get("move_left", 0))
            gt_list.append(gt_dict.get("move_right", 0) - gt_dict.get("move_left", 0))
        elif action_pair == "move_up_down":
            answer_list.append(answer_dict.get("move_up", 0) - answer_dict.get("move_down", 0))
            gt_list.append(gt_dict.get("move_up", 0) - gt_dict.get("move_down", 0))
        elif action_pair == "move_forward_backward":
            answer_list.append(answer_dict.get("move_forward", 0) - answer_dict.get("move_backward", 0))
            gt_list.append(gt_dict.get("move_forward", 0) - gt_dict.get("move_backward", 0))
        elif action_pair == "rotate_right_left":
            answer_list.append(answer_dict.get("rotate_right", 0) - answer_dict.get("rotate_left", 0))
            gt_list.append(gt_dict.get("rotate_right", 0) - gt_dict.get("rotate_left", 0))
        elif action_pair == "rotate_up_down":
            answer_list.append(answer_dict.get("rotate_up", 0) - answer_dict.get("rotate_down", 0))
            gt_list.append(gt_dict.get("rotate_up", 0) - gt_dict.get("rotate_down", 0))

    mra_list = []
    for gt, answer_value in zip(gt_list, answer_list):
        mra_list.append(mean_relative_accuracy(gt, answer_value, start=.5, end=.95, interval=.05))
    return float(np.mean(mra_list))


def exact_match(pred: str, target: str) -> float:
    """官方 MCA 精确匹配（含子串与首字符回退）。

    注意官方原实现直接访问 ``pred[0]``；空串在 ``pred in target`` 一步
    就会命中。本函数保持原样，空串保护放在 ``official_score_for`` 里，
    以免把“没有回答”判成正确。
    """
    pred = pred.lower()
    target = target.lower()
    if pred == target:
        return 1.0
    elif pred in target:
        return 1.0
    elif pred[0] == target:
        return 1.0
    else:
        return 0.0


def abs_dist_norm(pred: float, target: float) -> float:
    if target == 0.0:
        return abs(pred - target)
    return abs((pred - target) / target)


def mean_relative_accuracy(pred: float, target: float, start: float,
                           end: float, interval: float) -> float:
    """官方 MRA：11 个置信阈值下相对误差达标比例的均值。"""
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return float(accuracy.mean())


def to_float(pred):
    """官方 ``to_float``：转换失败返回 None。"""
    try:
        return float(pred)
    except BaseException:
        return None


WORST_CASE_FOR_METRICS = {"accuracy": 0.0, "MRA:.5:.95:.05": 0.0}


def official_score_for(task: str, prediction: str, answer: str) -> tuple[float, str | None]:
    """按官方规则给一条样本打分。

    与 lmms-eval 的唯一差异：Thinking On 时 ``prediction`` 已经是
    ``</think>`` 之后的最终回答区域（由上游切分）；空最终回答直接记 0，
    并在返回值中给出原因，避免官方 ``exact_match`` 的空串子串缺陷把
    “没答完”算成正确。

    Returns
    -------
    (score, invalid_reason)
        ``invalid_reason`` 仅在无法按官方方式解析时给出（空回答 / 数值抽取失败 /
        VCI 参数解析失败）。
    """
    prediction = (prediction or "").strip()
    if not prediction:
        return WORST_CASE_FOR_METRICS.get(
            "MRA:.5:.95:.05" if task in NA_QUESTION_TYPES else "accuracy", 0.0
        ), "empty_final_answer"

    if task in MCA_QUESTION_TYPES:
        return exact_match(prediction, answer), None

    if task in NA_QUESTION_TYPES:
        try:
            extracted = to_float(process_na(prediction, task))
            if extracted is None:
                return 0.0, "na_extraction_failed"
            return mean_relative_accuracy(
                extracted, to_float(answer), start=.5, end=.95, interval=.05
            ), None
        except Exception:
            return 0.0, "na_extraction_failed"

    if task == "view_change_infer":
        try:
            return compute_vci_metric(prediction, answer), None
        except Exception:
            return 0.0, "vci_parse_failed"

    raise ValueError(f"未知 SPAR-Bench 任务：{task}")


# --------------------------------------------------------------- 统计工具
def _mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _quantiles(values: list[float]) -> dict[str, float] | None:
    """返回均值 / 中位数 / P95 / 最大值，空列表返回 None。"""
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def _task_block(records: list[dict], task: str) -> dict[str, Any]:
    """单个任务的官方 / 严格汇总。"""
    rows = [r for r in records if r["task"] == task]
    official = [float(r["official_score"]) for r in rows]
    strict = [float(r["strict_score"]) for r in rows]
    graded = answer_type(task) in ("NA", "VCI")

    return {
        "task": task,
        "level": task_level(task),
        "answer_type": answer_type(task),
        "n": len(rows),
        "official_score": _mean(official),
        # MCA 的“正确数”是官方 exact_match == 1 的样本数；
        # NA / VCI 是连续分数，不强行定义 correct 数（另给满分样本数）。
        "official_correct": (
            None if graded else int(sum(1 for v in official if v >= 1.0))
        ),
        "official_full_score": int(sum(1 for v in official if v >= 1.0)),
        "strict_score": _mean(strict),
        "strict_correct": (
            None if graded else int(sum(1 for v in strict if v >= 1.0))
        ),
        "strict_valid": int(sum(1 for r in rows if r.get("strict_valid"))),
        "invalid": int(sum(1 for r in rows if r.get("invalid"))),
        "truncated": int(sum(1 for r in rows if r.get("truncated"))),
        "errors": int(sum(1 for r in rows if r.get("error"))),
        "n_images_mean": _mean([r.get("n_images") or 0 for r in rows]),
    }


def _level_block(per_task: dict[str, dict], tasks: list[str]) -> dict[str, Any]:
    present = [t for t in tasks if t in per_task]
    return {
        "tasks": len(present),
        "n_samples": int(sum(per_task[t]["n"] for t in present)),
        "official_score": _mean(per_task[t]["official_score"] for t in present),
        "strict_score": _mean(per_task[t]["strict_score"] for t in present),
    }


def _answer_type_block(per_task: dict[str, dict], tasks: list[str]) -> dict[str, Any]:
    present = [t for t in tasks if t in per_task]
    return {
        "tasks": len(present),
        "n_samples": int(sum(per_task[t]["n"] for t in present)),
        "task_mean_official": _mean(per_task[t]["official_score"] for t in present),
        "task_mean_strict": _mean(per_task[t]["strict_score"] for t in present),
    }


def compute_metrics(records: list[dict], meta: dict | None = None) -> dict[str, Any]:
    """由预测记录重新计算全部指标。

    这是唯一的结果汇总入口；``recompute_metrics.py`` 也调用它，因此
    ``metrics.json`` 可以完全由 ``predictions.jsonl`` 复现。

    ``records`` 中每条至少需要：task / final_answer_text / ground_truth /
    official_score / strict_score / invalid / truncated / 计时与 token 字段。
    """
    meta = meta or {}
    per_task: dict[str, dict] = {}
    present_tasks: list[str] = []
    recomputed_mismatch = 0
    missing_scores = 0

    for task in ALL_TASKS:
        rows = [r for r in records if r.get("task") == task]
        if not rows:
            continue
        present_tasks.append(task)
        per_task[task] = _task_block(records, task)

    # 重新按最终回答打一次分，核对落盘分数（防止运行中逻辑变更）。
    for record in records:
        if record.get("error"):
            continue
        try:
            score, _ = official_score_for(
                record["task"], record.get("final_answer_text", ""),
                record["ground_truth"],
            )
        except Exception:
            missing_scores += 1
            continue
        stored = record.get("official_score")
        if stored is None:
            missing_scores += 1
        elif abs(float(stored) - score) > 1e-9:
            recomputed_mismatch += 1

    overall_official = _mean(per_task[t]["official_score"] for t in present_tasks)
    overall_strict = _mean(per_task[t]["strict_score"] for t in present_tasks)

    levels = {
        "low": _level_block(per_task, LOW_TASKS),
        "medium": _level_block(per_task, MIDDLE_TASKS),
        "high": _level_block(per_task, HIGH_TASKS),
    }
    answer_types = {
        "MCA": _answer_type_block(per_task, MCA_QUESTION_TYPES),
        "NA": _answer_type_block(per_task, NA_QUESTION_TYPES),
        "VCI": _answer_type_block(per_task, SPECIAL_QUESTION_TYPES),
    }
    # 池化到样本层面的均值，与上面的“任务分数均值”并列展示；
    # 全量 50/任务时两者相等，小规模冒烟测试时可能不同。
    for name, tasks in (
        ("MCA", MCA_QUESTION_TYPES), ("NA", NA_QUESTION_TYPES),
        ("VCI", SPECIAL_QUESTION_TYPES),
    ):
        rows = [r for r in records if r["task"] in tasks]
        answer_types[name]["pooled_official"] = _mean(
            float(r["official_score"]) for r in rows
        )
        answer_types[name]["pooled_strict"] = _mean(
            float(r["strict_score"]) for r in rows
        )

    by_img_type: dict[str, dict] = {}
    for img_type in sorted({r.get("img_type") for r in records if r.get("img_type")}):
        rows = [r for r in records if r.get("img_type") == img_type]
        by_img_type[img_type] = {
            "n": len(rows),
            "tasks": len({r["task"] for r in rows}),
            "official_mean": _mean(float(r["official_score"]) for r in rows),
            "strict_mean": _mean(float(r["strict_score"]) for r in rows),
            "invalid": int(sum(1 for r in rows if r.get("invalid"))),
        }

    ids = [r.get("id") for r in records]
    duplicates = len(ids) - len(set(ids))

    quality = {
        "n_records": len(records),
        "n_unique_ids": len(set(ids)),
        "duplicate_ids": duplicates,
        "errors": int(sum(1 for r in records if r.get("error"))),
        "oom": int(sum(1 for r in records if r.get("oom"))),
        "truncated": int(sum(1 for r in records if r.get("truncated"))),
        "invalid": int(sum(1 for r in records if r.get("invalid"))),
        "invalid_reasons": dict(Counter(
            r.get("invalid_reason") for r in records if r.get("invalid_reason")
        )),
        "thinking_unclosed": int(sum(
            1 for r in records
            if r.get("thinking_mode") == "on" and not r.get("closed_thinking")
        )),
        "mode_unverified": int(sum(1 for r in records if not r.get("mode_verified"))),
        "image_count_mismatch": int(sum(
            1 for r in records if r.get("image_count_ok") is False
        )),
        "score_recompute_mismatch": recomputed_mismatch,
        "score_recompute_missing": missing_scores,
        "complete": (
            len(records) >= int(meta.get("selected_samples", 0) or 0)
            and not any(r.get("error") for r in records)
        ),
    }

    performance = _performance(records, meta)

    result = {
        "model": meta.get("model_name"),
        "protocol": meta.get("protocol"),
        "thinking_mode": meta.get("thinking_mode"),
        "enable_thinking": meta.get("enable_thinking"),
        "max_new_tokens": meta.get("max_new_tokens"),
        "prompt_version": meta.get("prompt_version"),
        "n_samples": len(records),
        "n_tasks": len(present_tasks),
        "expected_tasks": len(ALL_TASKS),
        "is_partial": len(present_tasks) < len(ALL_TASKS)
                      or (meta.get("selected_samples") is not None
                          and len(records) < int(meta["selected_samples"])),
        "overall": {
            "official": overall_official,
            "strict": overall_strict,
            "official_percent": None if overall_official is None else overall_official * 100,
            "strict_percent": None if overall_strict is None else overall_strict * 100,
            # 与官方聚合口径一致：20 个任务分数的算术平均。
            "aggregation": "mean_of_task_scores",
            # 仅作参考：把所有样本池化后的样本级均值，不是官方总分。
            "pooled_official_reference": _mean(
                float(r["official_score"]) for r in records
            ),
            "pooled_strict_reference": _mean(
                float(r["strict_score"]) for r in records
            ),
        },
        "levels": levels,
        "answer_types": answer_types,
        "img_types": by_img_type,
        "tasks": per_task,
        "quality": quality,
        "performance": performance,
    }
    return result


def _performance(records: list[dict], meta: dict) -> dict[str, Any]:
    """从预测记录汇总耗时、Token 与显存指标。"""
    def values(key):
        return [r[key] for r in records if isinstance(r.get(key), (int, float))]

    timing_keys = [
        "image_decode_seconds", "preprocess_seconds", "prefill_seconds",
        "generate_seconds", "e2e_seconds",
    ]
    timings = {key: _quantiles(values(key)) for key in timing_keys}

    gen_tokens = values("generated_tokens")
    think_tokens = values("thinking_tokens")
    answer_tokens = values("answer_tokens")
    prompt_tokens = values("prompt_tokens")

    total_generate = sum(values("generate_seconds"))
    total_e2e = sum(values("e2e_seconds"))
    total_tokens = sum(gen_tokens)

    # Prefill 与 Decode 分开：generate_seconds 包含 prefill；单独测过 prefill 时
    # 可以估算 decode 阶段耗时。
    decode_pairs = [
        (r["generate_seconds"], r["prefill_seconds"])
        for r in records
        if isinstance(r.get("generate_seconds"), (int, float))
        and isinstance(r.get("prefill_seconds"), (int, float))
        and r["generate_seconds"] >= r["prefill_seconds"]
    ]
    decode_seconds = [g - p for g, p in decode_pairs]
    total_decode = sum(decode_seconds)
    decode_tokens = sum(
        r["generated_tokens"] for r in records
        if isinstance(r.get("generate_seconds"), (int, float))
        and isinstance(r.get("prefill_seconds"), (int, float))
        and r["generate_seconds"] >= r["prefill_seconds"]
    )

    per_sample_peak = []
    gpu_used = []
    for r in records:
        vram = r.get("vram") or {}
        peak = vram.get("peak_allocated_gib")
        if isinstance(peak, (int, float)):
            per_sample_peak.append(float(peak))
        used = vram.get("gpu_used_gib")
        if isinstance(used, (int, float)):
            gpu_used.append(float(used))

    return {
        "model_load_seconds": meta.get("load_seconds"),
        "timing_seconds": timings,
        "total_generate_seconds": total_generate,
        "total_e2e_seconds": total_e2e,
        "total_wall_seconds": meta.get("wall_seconds"),
        "samples_per_second": (len(records) / total_e2e) if total_e2e > 0 else None,
        "samples_per_second_wall": (
            len(records) / float(meta["wall_seconds"])
            if meta.get("wall_seconds") else None
        ),
        "tokens_per_second_generate": (total_tokens / total_generate)
                                      if total_generate > 0 else None,
        "decode_seconds": _quantiles(decode_seconds),
        "total_decode_seconds": total_decode,
        "tokens_per_second_decode": (decode_tokens / total_decode)
                                    if total_decode > 0 else None,
        "tokens": {
            "prompt": _quantiles(prompt_tokens),
            "generated": _quantiles(gen_tokens),
            "thinking": _quantiles(think_tokens),
            "answer": _quantiles(answer_tokens),
            "total_generated": int(total_tokens),
        },
        "vram": {
            "after_load": meta.get("vram_after_load"),
            "per_sample_peak_allocated_gib_max": max(per_sample_peak) if per_sample_peak else None,
            "per_sample_peak_allocated_gib_mean": _mean(per_sample_peak),
            "gpu_used_gib_mean": _mean(gpu_used),
            "gpu_used_gib_max": max(gpu_used) if gpu_used else None,
            "gpu_start": meta.get("gpu_start"),
            "gpu_end": meta.get("gpu_end"),
            "global_peak_allocated_gib": meta.get("global_peak_allocated_gib"),
            "global_peak_reserved_gib": meta.get("global_peak_reserved_gib"),
        },
        "prefill_measured": any("prefill_seconds" in r for r in records),
    }


def format_percent(value: float | None) -> str:
    """把 0~1 的分数格式化成百分比字符串，缺失时返回 N/A。"""
    return "N/A" if value is None else f"{value * 100:.2f}"


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary_csv(path, metrics: dict) -> None:
    """把任务 / 层级 / 评分方式汇总写成 CSV。"""
    lines = [
        "scope,task,answer_type,level,n,official_score,strict_score,"
        "official_correct,strict_correct,invalid,truncated,errors"
    ]
    for task, block in metrics["tasks"].items():
        lines.append(",".join([
            "task", task, block["answer_type"], block["level"], str(block["n"]),
            _fmt(block["official_score"]), _fmt(block["strict_score"]),
            _fmt(block["official_correct"]), _fmt(block["strict_correct"]),
            str(block["invalid"]), str(block["truncated"]), str(block["errors"]),
        ]))
    overall = metrics["overall"]
    for scope, value in (
        ("overall_task_mean", overall["official"]),
        ("overall_task_mean_strict", overall["strict"]),
        ("overall_pooled_reference", overall.get("pooled_official_reference")),
    ):
        lines.append(f"{scope},,,,{metrics['n_samples']},{_fmt(value)},,,,,,")
    for level, block in metrics["levels"].items():
        lines.append(f"level,{level},,,{block['n_samples']},"
                     f"{_fmt(block['official_score'])},{_fmt(block['strict_score'])},,,,,,")
    for kind, block in metrics["answer_types"].items():
        lines.append(f"answer_type,{kind},,,{block['n_samples']},"
                     f"{_fmt(block['task_mean_official'])},"
                     f"{_fmt(block['task_mean_strict'])},,,,,,")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
