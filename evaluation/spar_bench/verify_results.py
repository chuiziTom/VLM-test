"""全量结果核验：样本数、任务覆盖、图片完整性、指标可复算性。

不加载模型、不使用 GPU。输出：

* reports/spar_bench_tiny_verification.md
* 控制台摘要

检查项：
1. 六组实验是否各 1,000 条、ID 唯一；
2. 20 种任务是否各 50 条；
3. Single-view 400 / Multi-view 600；
4. 每条样本的图片数量是否与原始数据集一致（2~3 图样本未被截断）；
5. Thinking 模式后缀是否全部校验通过；
6. `metrics.json` 是否能由 `predictions.jsonl` 重算且分数一致；
7. 是否存在未完成、OOM、截断、推理错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from sparbench_metrics import ALL_TASKS, compute_metrics  # noqa: E402

MODELS = ["xiaomi_lens_0", "qwen3_5_4b"]
PROTOCOLS = ["think_off_8192", "think_on_8192", "direct_off_64"]


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def check_run(run_dir: Path, expected: int = 1000) -> dict:
    result = {"dir": str(run_dir), "checks": {}, "ok": True}
    pred_path = run_dir / "predictions.jsonl"
    exp_path = run_dir / "experiment.json"
    met_path = run_dir / "metrics.json"

    def fail(name: str, detail: str) -> None:
        result["checks"][name] = {"ok": False, "detail": detail}
        result["ok"] = False

    def ok(name: str, detail: str) -> None:
        result["checks"][name] = {"ok": True, "detail": detail}

    if not pred_path.exists():
        fail("产物存在", f"缺少 {pred_path}")
        return result
    if not exp_path.exists() or not met_path.exists():
        fail("产物存在", "缺少 experiment.json 或 metrics.json")
        return result

    records = _read_jsonl(pred_path)
    experiment = json.loads(exp_path.read_text(encoding="utf-8"))
    stored_metrics = json.loads(met_path.read_text(encoding="utf-8"))

    result["n_records"] = len(records)
    if len(records) == expected:
        ok("样本数", f"{len(records)} / {expected}")
    else:
        fail("样本数", f"{len(records)} / {expected}")

    ids = [r.get("id") for r in records]
    task_counts = Counter(r.get("task") for r in records)
    img_counts = Counter(r.get("img_type") for r in records)

    if len(set(ids)) == len(ids):
        ok("ID 唯一", f"{len(set(ids))} 个唯一 ID")
    else:
        fail("ID 唯一", f"重复 {len(ids) - len(set(ids))} 条")

    missing_tasks = [t for t in ALL_TASKS if task_counts.get(t, 0) != 50]
    if not missing_tasks:
        ok("任务覆盖", "20 种任务各 50 条")
    else:
        fail("任务覆盖", f"数量异常的任务：{missing_tasks}")

    if img_counts.get("single_view") == 400 and img_counts.get("multi_view") == 600:
        ok("输入类型", "single_view=400, multi_view=600")
    else:
        fail("输入类型", f"{dict(img_counts)}")

    images_ok = all(
        (r.get("n_images_prompt") == r.get("n_images")) and r.get("image_count_ok")
        for r in records
        if not r.get("error")
    )
    if images_ok:
        ok("图片完整性", "所有样本的 Prompt 图片数 = 原始图片数")
    else:
        fail("图片完整性", "存在图片数量不一致的样本")

    mode_ok = all(r.get("mode_verified") for r in records if not r.get("error"))
    if mode_ok:
        ok("Thinking 模式", "所有样本的 chat template 后缀校验通过")
    else:
        fail("Thinking 模式", "存在模式校验失败")

    status = experiment.get("status")
    q = stored_metrics.get("quality", {})
    if status == "complete" and not q.get("errors"):
        ok("运行状态", "complete")
    else:
        fail("运行状态", f"status={status}, errors={q.get('errors')}")

    meta = {
        **experiment.get("config", {}),
        **experiment,
        **(experiment.get("load_info") or {}),
    }
    recomputed = compute_metrics(records, meta=meta)
    mismatch = recomputed["quality"]["score_recompute_mismatch"]
    if mismatch == 0:
        ok("指标可复算", "官方分数与落盘一致")
    else:
        fail("指标可复算", f"{mismatch} 条分数不一致")

    stored_overall = stored_metrics.get("overall", {}).get("official")
    recomputed_overall = recomputed["overall"]["official"]
    stored_n = stored_metrics.get("n_samples")
    if (stored_n == recomputed["n_samples"]
            and stored_overall is not None
            and abs(stored_overall - recomputed_overall) < 1e-12):
        ok("metrics 一致", f"n_samples={stored_n}, overall={recomputed_overall:.6f}")
    else:
        fail("metrics 一致", f"落盘 n={stored_n}/overall={stored_overall}，"
                              f"重算 n={recomputed['n_samples']}/overall={recomputed_overall}")

    for key in ("overall", "levels", "answer_types", "tasks", "performance"):
        if key not in stored_metrics:
            fail("metrics 结构", f"缺少 {key}")
            break

    result["official"] = recomputed["overall"]["official"]
    result["strict"] = recomputed["overall"]["strict"]
    result["invalid"] = recomputed["quality"]["invalid"]
    result["truncated"] = recomputed["quality"]["truncated"]
    result["errors"] = recomputed["quality"]["errors"]
    result["thinking_unclosed"] = recomputed["quality"]["thinking_unclosed"]
    result["status"] = status
    result["n_tasks"] = recomputed["n_tasks"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SPAR-Bench results")
    parser.add_argument("--root", default=str(ROOT / "experiments" / "spar_bench" / "tiny_rgb"))
    parser.add_argument("--report", default=str(ROOT / "reports" / "spar_bench_tiny_verification.md"))
    args = parser.parse_args()

    root = Path(args.root)
    results = {}
    for model in MODELS:
        for protocol in PROTOCOLS:
            run_dir = root / model / protocol
            results[(model, protocol)] = check_run(run_dir)

    lines = ["# SPAR-Bench-Tiny 结果核验", ""]
    lines.append("> 全部检查直接读取 `predictions.jsonl` / `experiment.json` / `metrics.json`，"
                 "并用评分代码重新计算。")
    lines.append("")
    lines.append("| 模型 | 协议 | 样本 | 任务 | Official | Strict | Invalid | 截断 | 错误 | 结论 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for (model, protocol), result in results.items():
        if "official" not in result:
            lines.append(f"| {model} | `{protocol}` | - | - | - | - | - | - | - | ❌ 未完成 |")
            continue
        verdict = "✅ 通过" if result["ok"] else "❌ 失败"
        lines.append(
            f"| {model} | `{protocol}` | {result['n_records']} | {result['n_tasks']} | "
            f"{result['official'] * 100:.2f} | {result['strict'] * 100:.2f} | "
            f"{result['invalid']} | {result['truncated']} | {result['errors']} | {verdict} |"
        )
    lines.append("")

    lines.append("## 明细")
    lines.append("")
    for (model, protocol), result in results.items():
        lines.append(f"### {model} / `{protocol}`")
        lines.append("")
        for name, check in result["checks"].items():
            mark = "✅" if check["ok"] else "❌"
            lines.append(f"- {mark} **{name}**：{check['detail']}")
        lines.append("")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出 {report_path}")

    all_ok = all(result["ok"] for result in results.values())
    print("全部通过 ✅" if all_ok else "存在失败项 ❌，详见报告")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
