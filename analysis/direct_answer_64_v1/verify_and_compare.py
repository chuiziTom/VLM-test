"""Direct Answer 结果核验与两模型对比（只读）。

做四件事：
1. 用 CV-Bench 的 uid 集合检查两个模型是否完整覆盖、是否有重复；
2. 从 raw_generation 重新解析，确认保存的 prediction 可复现；
3. 用逐样本记录重新计算官方加权分，与 metrics.json 对比；
4. 生成两模型汇总、逐题胜负、McNemar 精确检验，并读取历史实验作为参考。

用法：
    micromamba run -n vlm-test python -B \
        analysis/direct_answer_64_v1/verify_and_compare.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent          # analysis/direct_answer_64_v1
ROOT = HERE.parents[1]                          # 项目根目录
EXP_DIR = ROOT / "evaluation" / "direct_answer"
RESULT_DIR = ROOT / "experiments" / "cv_bench" / "direct_answer_64_v1"
DATASET_DIR = ROOT / "datasets" / "CV-Bench"
COMMON_DIR = ROOT / "evaluation" / "common"
LEGACY_DIR = ROOT / "experiments" / "cv_bench" / "legacy_8192"

sys.path.insert(0, str(EXP_DIR))

from common.direct_parser import parse_direct_answer  # noqa: E402
from common.module_loader import load_module  # noqa: E402

dataset_mod = load_module("cvbench_dataset_compare",
                          COMMON_DIR / "cvbench_dataset.py")
metrics_mod = load_module("metrics_compare",
                          COMMON_DIR / "metrics.py")

MODELS = {
    "xiaomi_lens": RESULT_DIR / "xiaomi_lens",
    "qwen35_4b": RESULT_DIR / "qwen35_4b",
}

LEGACY = {
    "xiaomi_lens": {
        "off": LEGACY_DIR / "xiaomi_lens" / "thinking_off",
        "on": LEGACY_DIR / "xiaomi_lens" / "thinking_on",
    },
    "qwen35_4b": {
        "off": LEGACY_DIR / "qwen35_4b" / "thinking_off",
        "on": LEGACY_DIR / "qwen35_4b" / "thinking_on",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def exact_binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pct(value):
    return None if value is None else round(100.0 * value, 4)


def verify_model(name: str, run_dir: Path, expected_uids: set[str]) -> dict:
    records = load_jsonl(run_dir / "predictions.jsonl")
    uids = [r["uid"] for r in records]
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    duplicates = sorted(uid for uid, c in Counter(uids).items() if c > 1)
    missing = sorted(expected_uids - set(uids))
    unexpected = sorted(set(uids) - expected_uids)

    parse_mismatch = []
    correct_mismatch = []
    for r in records:
        letters = tuple(chr(ord("A") + i) for i in range(len(r["choices"])))
        parse = parse_direct_answer(
            r.get("raw_generation", ""), letters,
            truncated=bool(r.get("truncated")),
        )
        if parse.prediction != r.get("prediction"):
            parse_mismatch.append({
                "uid": r["uid"], "stored": r.get("prediction"),
                "reparsed": parse.prediction,
                "stored_reason": r.get("invalid_reason"),
                "reparsed_reason": parse.invalid_reason,
            })
        truth = r["ground_truth"]
        correct = parse.prediction is not None and parse.prediction == truth
        if correct != bool(r.get("correct")):
            correct_mismatch.append({"uid": r["uid"], "stored": r.get("correct"),
                                     "reparsed": correct})

    recomputed = metrics_mod.compute_metrics(records, stored.get("experiment"))
    stored_official = stored["official"]
    recomputed_official = recomputed["official"]
    diffs = {
        key: {
            "stored": stored_official.get(key),
            "recomputed": recomputed_official.get(key),
            "delta": (
                recomputed_official.get(key) - stored_official.get(key)
                if isinstance(recomputed_official.get(key), (int, float))
                and isinstance(stored_official.get(key), (int, float)) else None
            ),
        }
        for key in ("accuracy_2d_ade20k", "accuracy_2d_coco", "accuracy_2d",
                    "accuracy_3d_omni3d", "cvbench_accuracy")
    }

    fmt = stored.get("direct_answer", {})
    return {
        "model": name,
        "run_dir": str(run_dir),
        "n_records": len(records),
        "unique_uids": len(set(uids)),
        "duplicates": duplicates[:10],
        "missing_count": len(missing),
        "missing_first10": missing[:10],
        "unexpected_count": len(unexpected),
        "parse_mismatch_count": len(parse_mismatch),
        "parse_mismatch": parse_mismatch[:10],
        "correct_mismatch_count": len(correct_mismatch),
        "correct_mismatch": correct_mismatch[:10],
        "stored": {
            "status": stored.get("experiment", {}).get("status"),
            "completed": stored.get("experiment", {}).get("completed_samples"),
            "expected": stored.get("experiment", {}).get("expected_samples"),
            "accuracy_percent": stored.get("accuracy_percent"),
            "official": stored_official,
            "invalid": stored.get("invalid"),
            "truncated": stored.get("truncated"),
            "errors": stored.get("errors"),
            "invalid_reasons": stored.get("invalid_reasons"),
            "format_violations": fmt.get("format_violations"),
            "format_violation_rate": fmt.get("format_violation_rate"),
        },
        "recomputed": {
            "accuracy_percent": recomputed.get("accuracy_percent"),
            "official": recomputed_official,
            "invalid": recomputed.get("invalid"),
            "truncated": recomputed.get("truncated"),
            "errors": recomputed.get("errors"),
            "groups": recomputed.get("groups"),
            "tokens": recomputed.get("tokens"),
            "timing": recomputed.get("timing_seconds"),
            "throughput": recomputed.get("throughput"),
            "vram": recomputed.get("vram"),
        },
        "official_diff": diffs,
        "_records": records,
    }


def main() -> None:
    dataset = dataset_mod.load_records(DATASET_DIR)
    expected_uids = {r.uid for r in dataset}
    results = {name: verify_model(name, run_dir, expected_uids)
               for name, run_dir in MODELS.items()}

    summary = {}
    for name, r in results.items():
        rec = r["recomputed"]
        summary[name] = {
            "official_cvbench_percent": pct(rec["official"]["cvbench_accuracy"]),
            "official_2d_percent": pct(rec["official"]["accuracy_2d"]),
            "official_3d_percent": pct(rec["official"]["accuracy_3d"]),
            "ade20k_percent": pct(rec["official"]["accuracy_2d_ade20k"]),
            "coco_percent": pct(rec["official"]["accuracy_2d_coco"]),
            "omni3d_percent": pct(rec["official"]["accuracy_3d_omni3d"]),
            "plain_accuracy_percent": rec["accuracy_percent"],
            "invalid": rec["invalid"],
            "invalid_rate_percent": pct(rec["invalid"] / r["n_records"]),
            "truncated": rec["truncated"],
            "format_violations": r["stored"]["format_violations"],
            "task_percent": {k: pct(v["accuracy"]) for k, v in rec["groups"]["by_task"].items()},
            "source_percent": {k: pct(v["accuracy"]) for k, v in rec["groups"]["by_source"].items()},
            "type_task_source_percent": {
                k: pct(v["accuracy"]) for k, v in rec["groups"]["by_type_task_source"].items()
            },
            "task_counts": {k: {"correct": v["correct"], "n": v["n"]}
                            for k, v in rec["groups"]["by_task"].items()},
            "source_counts": {k: {"correct": v["correct"], "n": v["n"]}
                              for k, v in rec["groups"]["by_source"].items()},
            "mean_e2e_seconds": round(rec["timing"]["end_to_end"]["mean"], 4),
            "median_e2e_seconds": round(rec["timing"]["end_to_end"]["median"], 4),
            "p95_e2e_seconds": round(rec["timing"]["end_to_end"]["p95"], 4),
            "mean_generate_seconds": round(rec["timing"]["generate"]["mean"], 4),
            "mean_preprocess_seconds": round(rec["timing"]["preprocess"]["mean"], 4),
            "mean_image_load_seconds": round(rec["timing"]["image_load"]["mean"], 4),
            "mean_generated_tokens": round(rec["tokens"]["generated"]["mean"], 3),
            "median_generated_tokens": rec["tokens"]["generated"]["median"],
            "p95_generated_tokens": rec["tokens"]["generated"]["p95"],
            "tokens_per_second": round(rec["throughput"]["tokens_per_second_generate"], 3),
            "samples_per_second": round(rec["throughput"]["samples_per_second_e2e"], 4),
            "total_generated_tokens": rec["tokens"]["generated"]["sum"],
            "total_wall_seconds": round(rec["timing"]["end_to_end"]["sum"], 2),
            "peak_allocated_gib": round(rec["vram"]["global_peak_allocated_gib"] or 0, 4),
            "per_sample_peak_mean_gib": round(
                rec["vram"]["per_sample_peak_allocated_gib_mean"] or 0, 4),
        }

    # ---- 逐题对比 ----
    a, b = "xiaomi_lens", "qwen35_4b"
    by_uid = {name: {r["uid"]: r for r in results[name]["_records"]}
              for name in results}
    common = sorted(set(by_uid[a]) & set(by_uid[b]))
    a_right_b_wrong = sum(1 for u in common
                          if by_uid[a][u]["correct"] and not by_uid[b][u]["correct"])
    a_wrong_b_right = sum(1 for u in common
                          if not by_uid[a][u]["correct"] and by_uid[b][u]["correct"])
    both_right = sum(1 for u in common
                     if by_uid[a][u]["correct"] and by_uid[b][u]["correct"])
    both_wrong = sum(1 for u in common
                     if not by_uid[a][u]["correct"] and not by_uid[b][u]["correct"])
    n_disc = a_right_b_wrong + a_wrong_b_right
    p_value = exact_binom_two_sided(min(a_right_b_wrong, a_wrong_b_right), n_disc)

    by_task_diff = {}
    for task in sorted({r["task"] for r in results[a]["_records"]}):
        au = [u for u in common if by_uid[a][u]["task"] == task]
        aw = sum(1 for u in au if by_uid[a][u]["correct"] and not by_uid[b][u]["correct"])
        bw = sum(1 for u in au if not by_uid[a][u]["correct"] and by_uid[b][u]["correct"])
        by_task_diff[task] = {
            "n": len(au),
            f"{a}_only_right": aw,
            f"{b}_only_right": bw,
            "net_xiaomi_minus_qwen": aw - bw,
            "exact_mcnemar_p": exact_binom_two_sided(min(aw, bw), aw + bw)
            if aw + bw else 1.0,
        }

    head_to_head = {
        "common": len(common),
        f"{a}_right_{b}_wrong": a_right_b_wrong,
        f"{a}_wrong_{b}_right": a_wrong_b_right,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "net_xiaomi_minus_qwen": a_right_b_wrong - a_wrong_b_right,
        "exact_mcnemar_p": p_value,
        "by_task": by_task_diff,
    }

    # ---- 历史实验参考值（只读，不参与正式对比结论） ----
    legacy = {}
    for name, modes in LEGACY.items():
        legacy[name] = {}
        for mode, path in modes.items():
            m = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            legacy[name][mode] = {
                "official_cvbench_percent": m["official"]["cvbench_accuracy_percent"],
                "official_2d_percent": 100 * m["official"]["accuracy_2d"],
                "official_3d_percent": 100 * m["official"]["accuracy_3d"],
                "accuracy_percent": m["accuracy_percent"],
                "invalid": m["invalid"],
                "truncated": m["truncated"],
                "mean_generated_tokens": m["tokens"]["generated"]["mean"],
                "mean_e2e_seconds": m["timing_seconds"]["end_to_end"]["mean"],
                "p95_e2e_seconds": m["timing_seconds"]["end_to_end"]["p95"],
                "max_new_tokens": m["experiment"]["max_new_tokens"],
                "prompt_sha1": m["experiment"]["prompt_template_sha1"],
            }

    payload = {
        "dataset": {"total": len(dataset),
                    "distribution": dataset_mod.describe(dataset)},
        "runs": {name: {k: v for k, v in results[name].items() if k != "_records"}
                 for name in results},
        "summary": summary,
        "head_to_head": head_to_head,
        "legacy_reference": legacy,
        "notes": {
            "same_protocol": "两模型使用同一份 Direct Answer Prompt、同一 max_new_tokens=64、同一解析器与官方评分公式。",
            "legacy_caveat": "历史 8192 实验的 Prompt、生成上限与解析规则不同，只能作为参考，不是严格对照组。",
        },
    }

    out_path = HERE / "verification.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_path}\n")

    print("== 完整性 ==")
    for name, r in results.items():
        print(f"{name:12s} records={r['n_records']} unique={r['unique_uids']} "
              f"missing={r['missing_count']} dup={len(r['duplicates'])} "
              f"parse_mismatch={r['parse_mismatch_count']} "
              f"correct_mismatch={r['correct_mismatch_count']}")
        d = r["official_diff"]["cvbench_accuracy"]
        print(f"             stored={100 * d['stored']:.4f}% "
              f"recomputed={100 * d['recomputed']:.4f}% delta={100 * d['delta']:.6f}")

    print("\n== 汇总 ==")
    for name, s in summary.items():
        print(f"{name:12s} official={s['official_cvbench_percent']:.2f}% "
              f"2D={s['official_2d_percent']:.2f}% 3D={s['official_3d_percent']:.2f}% "
              f"invalid={s['invalid']} fmt_viol={s['format_violations']} "
              f"tok={s['mean_generated_tokens']:.1f} e2e={s['mean_e2e_seconds']:.3f}s")

    print("\n== 逐题对比 ==")
    print(json.dumps(head_to_head, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
