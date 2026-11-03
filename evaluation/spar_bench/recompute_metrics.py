"""从 predictions.jsonl 重新计算 metrics.json / summary.csv。

用途：验证结果可复现，且不需要加载模型或占用 GPU。

    # 单个实验目录
    micromamba run -n vlm-test python evaluation/spar_bench/recompute_metrics.py \
        experiments/spar_bench/tiny_rgb/qwen3_5_4b/think_off_8192

    # 全部 6 组
    micromamba run -n vlm-test python evaluation/spar_bench/recompute_metrics.py \
        --all experiments/spar_bench/tiny_rgb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sparbench_metrics import compute_metrics, write_summary_csv  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def recompute(run_dir: Path) -> bool:
    predictions = run_dir / "predictions.jsonl"
    experiment_path = run_dir / "experiment.json"
    if not predictions.exists() or not experiment_path.exists():
        print(f"跳过（缺少产物）：{run_dir}")
        return False

    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    records = _read_jsonl(predictions)
    meta = {
        **experiment.get("config", {}),
        **experiment,
        **(experiment.get("load_info") or {}),
    }
    metrics = compute_metrics(records, meta=meta)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_csv(run_dir / "summary.csv", metrics)

    quality = metrics["quality"]
    print(
        f"{run_dir}: {metrics['n_samples']} 条, "
        f"tasks={metrics['n_tasks']}, "
        f"official={metrics['overall']['official']}, "
        f"strict={metrics['overall']['strict']}, "
        f"invalid={quality['invalid']}, truncated={quality['truncated']}, "
        f"errors={quality['errors']}, "
        f"score_mismatch={quality['score_recompute_mismatch']}"
    )
    if quality["score_recompute_mismatch"]:
        print("  [警告] 落盘分数与重新计算结果不一致，请检查评分代码版本！")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute SPAR-Bench metrics")
    parser.add_argument("run_dir", nargs="?", help="单个实验目录")
    parser.add_argument("--all", help="实验根目录，递归查找所有 experiment.json")
    args = parser.parse_args()

    if args.all:
        root = Path(args.all)
        dirs = sorted({path.parent for path in root.rglob("experiment.json")})
        count = sum(1 for run_dir in dirs if recompute(run_dir))
        print(f"共重算 {count} 个实验目录")
        return 0

    if not args.run_dir:
        parser.error("需要 run_dir 或 --all")
    return 0 if recompute(Path(args.run_dir)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
