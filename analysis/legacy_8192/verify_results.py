"""独立核验脚本（只读）。

目的：
1. 用 CV-Bench parquet 的 uid 集合检查四种运行（Qwen Off/On、Xiaomi Off/On）
   是否完整覆盖 2638 条测试样本、是否有重复。
2. 从每条记录的 raw_generation 重新解析最终答案，与已保存的 prediction 对比，
   检查答案解析是否可复现。
3. 用原始逐样本记录重新计算 Accuracy（普通平均与官方加权），与 metrics.json 对比。
4. 生成四种运行的横向汇总与逐样本胜负矩阵。

本脚本不写入任何原始结果目录，输出写到 analysis/legacy_8192/。
用法：
    micromamba run -n vlm-test python -B analysis/legacy_8192/verify_results.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COMMON_DIR = ROOT / "evaluation" / "common"
LEGACY_DIR = ROOT / "experiments" / "cv_bench" / "legacy_8192"
OUT_DIR = HERE

sys.dont_write_bytecode = True
sys.path.insert(0, str(COMMON_DIR))

import cvbench_dataset  # noqa: E402
import answer_parser  # noqa: E402
import metrics as metrics_mod  # noqa: E402

DATASET_DIR = ROOT / "datasets" / "CV-Bench"

RUNS = {
    "qwen_off": (LEGACY_DIR / "qwen35_4b" / "thinking_off", "Qwen3.5-4B", "off"),
    "qwen_on": (LEGACY_DIR / "qwen35_4b" / "thinking_on", "Qwen3.5-4B", "on"),
    "xiaomi_off": (LEGACY_DIR / "xiaomi_lens" / "thinking_off", "Xiaomi-Lens-0", "off"),
    "xiaomi_on": (LEGACY_DIR / "xiaomi_lens" / "thinking_on", "Xiaomi-Lens-0", "on"),
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} JSON 解析失败: {exc}")
    return records


def answer_letter(answer: str) -> str | None:
    text = str(answer).strip()
    if len(text) == 3 and text[0] == "(" and text[2] == ")":
        return text[1]
    if len(text) == 1 and text.isalpha():
        return text.upper()
    return None


def verify_run(name: str, run_dir: Path, expected: dict[str, cvbench_dataset.CVBenchRecord]) -> dict:
    predictions = load_jsonl(run_dir / "predictions.jsonl")
    stored_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    uids = [r["uid"] for r in predictions]
    uid_set = set(uids)
    duplicates = [uid for uid, count in Counter(uids).items() if count > 1]
    missing = sorted(set(expected) - uid_set)
    unexpected = sorted(uid_set - set(expected))

    metadata_mismatch = []
    parse_mismatch = []
    stored_correct_mismatch = []
    reparsed = []

    for record in predictions:
        uid = record["uid"]
        ref = expected.get(uid)
        if ref is not None:
            for field, want in (
                ("type", ref.type), ("task", ref.task), ("source", ref.source),
                ("question", ref.question), ("choices", list(ref.choices)),
            ):
                if record.get(field) != want:
                    metadata_mismatch.append({"uid": uid, "field": field})

        parse = answer_parser.parse_final_answer(
            record.get("raw_generation") or "",
            tuple(chr(ord("A") + i) for i in range(len(record["choices"]))),
            requires_think_close=bool(record.get("enable_thinking")),
            truncated=bool(record.get("truncated")),
        )
        if parse.prediction != record.get("prediction"):
            parse_mismatch.append({
                "uid": uid,
                "stored": record.get("prediction"),
                "reparsed": parse.prediction,
                "stored_reason": record.get("invalid_reason"),
                "reparsed_reason": parse.invalid_reason,
            })

        truth = answer_letter(record["ground_truth"])
        correct = parse.prediction is not None and parse.prediction == truth
        if correct != bool(record.get("correct")):
            stored_correct_mismatch.append({
                "uid": uid, "stored": record.get("correct"), "reparsed": correct,
            })

        copy = dict(record)
        copy["prediction"] = parse.prediction
        copy["invalid"] = parse.prediction is None
        copy["invalid_reason"] = parse.invalid_reason
        copy["correct"] = correct
        reparsed.append(copy)

    recomputed = metrics_mod.compute_metrics(reparsed, stored_metrics.get("experiment"))

    def official(m):
        return m["official"]

    stored_off = official(stored_metrics)
    recomputed_off = official(recomputed)
    official_diff = {
        key: {
            "stored": stored_off.get(key),
            "recomputed": recomputed_off.get(key),
            "delta": (
                recomputed_off.get(key) - stored_off.get(key)
                if isinstance(recomputed_off.get(key), (int, float))
                and isinstance(stored_off.get(key), (int, float)) else None
            ),
        }
        for key in ("accuracy_2d_ade20k", "accuracy_2d_coco", "accuracy_2d",
                    "accuracy_3d_omni3d", "cvbench_accuracy")
    }

    return {
        "name": name,
        "run_dir": str(run_dir),
        "n_records": len(predictions),
        "unique_uids": len(uid_set),
        "duplicates": duplicates,
        "missing_count": len(missing),
        "missing_first10": missing[:10],
        "unexpected_count": len(unexpected),
        "expected_samples": len(expected),
        "metadata_mismatch": metadata_mismatch[:10],
        "metadata_mismatch_count": len(metadata_mismatch),
        "parse_mismatch_count": len(parse_mismatch),
        "parse_mismatch": parse_mismatch[:10],
        "stored_correct_mismatch_count": len(stored_correct_mismatch),
        "stored_correct_mismatch": stored_correct_mismatch[:10],
        "stored_metrics": {
            "accuracy_percent": stored_metrics["accuracy_percent"],
            "official": stored_off,
            "invalid": stored_metrics["invalid"],
            "truncated": stored_metrics["truncated"],
            "valid_answers": stored_metrics["valid_answers"],
            "errors": stored_metrics["errors"],
            "status": stored_metrics["experiment"].get("status"),
            "completed": stored_metrics["experiment"].get("completed_samples"),
            "expected": stored_metrics["experiment"].get("expected_samples"),
            "aborted": stored_metrics.get("aborted"),
        },
        "recomputed": {
            "accuracy_percent": recomputed["accuracy_percent"],
            "official": recomputed_off,
            "invalid": recomputed["invalid"],
            "invalid_reasons": recomputed["invalid_reasons"],
            "truncated": recomputed["truncated"],
            "errors": recomputed["errors"],
            "groups": recomputed["groups"],
            "tokens": recomputed["tokens"],
            "timing": recomputed["timing_seconds"],
            "throughput": recomputed["throughput"],
            "vram": recomputed["vram"],
        },
        "official_diff": official_diff,
        "_records": reparsed,
    }


def pct(value):
    return None if value is None else round(100.0 * value, 4)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = cvbench_dataset.load_records(DATASET_DIR)
    expected = {r.uid: r for r in dataset}
    print(f"数据集：{len(expected)} 条（唯一 uid {len(set(expected))}）")

    results = {name: verify_run(name, run_dir, expected)
               for name, (run_dir, _, _) in RUNS.items()}

    # ---- 逐样本胜负矩阵（四种运行两两对比）----
    head_to_head = {}
    names = list(RUNS)
    by_uid = {name: {r["uid"]: r for r in results[name]["_records"]} for name in names}
    common_uids = sorted(set.intersection(*(set(by_uid[n]) for n in names)))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [uid for uid in common_uids]
            a_ok_b_bad = sum(1 for u in both if by_uid[a][u]["correct"] and not by_uid[b][u]["correct"])
            a_bad_b_ok = sum(1 for u in both if not by_uid[a][u]["correct"] and by_uid[b][u]["correct"])
            both_ok = sum(1 for u in both if by_uid[a][u]["correct"] and by_uid[b][u]["correct"])
            both_bad = sum(1 for u in both if not by_uid[a][u]["correct"] and not by_uid[b][u]["correct"])
            head_to_head[f"{a}_vs_{b}"] = {
                "common": len(both),
                f"{a}_right_{b}_wrong": a_ok_b_bad,
                f"{a}_wrong_{b}_right": a_bad_b_ok,
                "both_right": both_ok,
                "both_wrong": both_bad,
            }

    # ---- 汇总表 ----
    summary_table = {}
    for name in names:
        m = results[name]["recomputed"]
        status = results[name]["stored_metrics"]
        summary_table[name] = {
            "official_cvbench_percent": pct(m["official"]["cvbench_accuracy"]),
            "official_2d_percent": pct(m["official"]["accuracy_2d"]),
            "official_3d_percent": pct(m["official"]["accuracy_3d"]),
            "ade20k_percent": pct(m["official"]["accuracy_2d_ade20k"]),
            "coco_percent": pct(m["official"]["accuracy_2d_coco"]),
            "omni3d_percent": pct(m["official"]["accuracy_3d_omni3d"]),
            "plain_accuracy_percent": m["accuracy_percent"],
            "invalid": m["invalid"],
            "invalid_rate_percent": pct(m["invalid"] / results[name]["n_records"]),
            "truncated": m["truncated"],
            "mean_e2e_seconds": round(m["timing"]["end_to_end"]["mean"], 4),
            "p95_e2e_seconds": round(m["timing"]["end_to_end"]["p95"], 4),
            "mean_generate_seconds": round(m["timing"]["generate"]["mean"], 4),
            "mean_generated_tokens": round(m["tokens"]["generated"]["mean"], 3),
            "mean_thinking_tokens": round(m["tokens"]["thinking"]["mean"], 3),
            "tokens_per_second": round(m["throughput"]["tokens_per_second_generate"], 3),
            "total_wall_seconds": round(m["timing"]["end_to_end"]["sum"], 2),
            "peak_allocated_gib": round(m["vram"]["global_peak_allocated_gib"], 4),
            "status": status["status"],
            "completed": status["completed"],
            "expected": status["expected"],
            "task_percent": {
                task: pct(group["accuracy"]) for task, group in m["groups"]["by_task"].items()
            },
            "type_task_source_percent": {
                key: pct(group["accuracy"]) for key, group in m["groups"]["by_type_task_source"].items()
            },
        }

    # ---- 按任务看 Thinking 对两个模型的影响 ----
    thinking_effect = {}
    for model in ("qwen", "xiaomi"):
        off = results[f"{model}_off"]["recomputed"]["groups"]["by_task"]
        on = results[f"{model}_on"]["recomputed"]["groups"]["by_task"]
        thinking_effect[model] = {
            task: {
                "off_percent": pct(off[task]["accuracy"]),
                "on_percent": pct(on[task]["accuracy"]),
                "delta_pp": round(100 * (on[task]["accuracy"] - off[task]["accuracy"]), 4),
                "off_correct": off[task]["correct"], "on_correct": on[task]["correct"],
                "n": off[task]["n"],
            }
            for task in sorted(off)
        }

    # 剔除 invalid 后的有效回答准确率（仅作参考，不替代官方分数）
    valid_only = {}
    for name in names:
        recs = results[name]["_records"]
        valid = [r for r in recs if not r["invalid"]]
        ok = sum(1 for r in valid if r["correct"])
        valid_only[name] = {
            "valid": len(valid),
            "correct": ok,
            "accuracy_percent": round(100 * ok / len(valid), 4) if valid else None,
        }

    payload = {
        "dataset": {
            "total": len(expected),
            "distribution": cvbench_dataset.describe(dataset),
        },
        "runs": {
            name: {k: v for k, v in results[name].items() if k != "_records"}
            for name in names
        },
        "summary_table": summary_table,
        "thinking_effect_by_task": thinking_effect,
        "valid_only_accuracy": valid_only,
        "head_to_head": head_to_head,
    }

    out_path = OUT_DIR / "verification.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out_path}")

    print("\n== 完整性 ==")
    for name in names:
        r = results[name]
        print(f"{name:11s} records={r['n_records']} unique={r['unique_uids']} "
              f"missing={r['missing_count']} dup={len(r['duplicates'])} "
              f"metadata_mismatch={r['metadata_mismatch_count']} "
              f"parse_mismatch={r['parse_mismatch_count']} "
              f"correct_mismatch={r['stored_correct_mismatch_count']}")

    print("\n== 复算 vs 原 metrics.json (官方总分 %) ==")
    for name in names:
        d = results[name]["official_diff"]["cvbench_accuracy"]
        print(f"{name:11s} stored={100*d['stored']:.4f} recomputed={100*d['recomputed']:.4f} "
              f"delta={100*d['delta']:.6f}")

    print("\n== 汇总 ==")
    for name in names:
        s = summary_table[name]
        print(f"{name:11s} official={s['official_cvbench_percent']:.2f}% "
              f"2D={s['official_2d_percent']:.2f}% 3D={s['official_3d_percent']:.2f}% "
              f"invalid={s['invalid']} mean_e2e={s['mean_e2e_seconds']:.2f}s "
              f"mean_tok={s['mean_generated_tokens']:.1f} status={s['status']}")

    print("\n== 逐样本胜负 ==")
    for key, value in head_to_head.items():
        print(key, value)


if __name__ == "__main__":
    main()
