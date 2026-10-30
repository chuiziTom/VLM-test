"""Build a Chinese comparison report from saved evaluation runs.

Usage::

    micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/make_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import compare_runs  # noqa: E402

TASK_LABELS = {
    "Count": "物体计数 (Count)",
    "Relation": "二维空间关系 (Relation)",
    "Depth": "深度顺序 (Depth)",
    "Distance": "相对距离 (Distance)",
}


def _load(mode_dir: Path):
    metrics_path = mode_dir / "metrics.json"
    if not metrics_path.exists():
        return None, []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = mode_dir / "predictions.jsonl"
    records = []
    if predictions.exists():
        for line in predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return metrics, records


def _pct(value, digits=2):
    if value is None:
        return "N/A"
    return f"{100 * value:.{digits}f}%"


def _num(value, digits=2):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _metric(metrics, *keys, default=None):
    current = metrics
    for key in keys:
        if current is None:
            return default
        current = current.get(key) if isinstance(current, dict) else None
    return default if current is None else current


def _task_acc(metrics, task):
    block = metrics["groups"]["by_task"].get(task, {})
    return block.get("accuracy")


def _source_acc(metrics, source):
    block = metrics["groups"]["by_source"].get(source, {})
    return block.get("accuracy")


def comparison_table(off, on):
    lines = [
        "| 指标 | Thinking Off | Thinking On | On - Off |",
        "| --- | ---: | ---: | ---: |",
    ]

    def row(label, off_value, on_value, formatter=_pct, delta_pp=False):
        off_text = formatter(off_value)
        on_text = formatter(on_value)
        if off_value is None or on_value is None:
            delta = "N/A"
        elif delta_pp:
            delta = f"{100 * (on_value - off_value):+.2f} pp"
        else:
            delta = formatter(on_value - off_value)
        lines.append(f"| {label} | {off_text} | {on_text} | {delta} |")

    row("CV-Bench 官方总分", _metric(off, "official", "cvbench_accuracy"),
        _metric(on, "official", "cvbench_accuracy"), delta_pp=True)
    row("2D Accuracy（官方加权）", _metric(off, "official", "accuracy_2d"),
        _metric(on, "official", "accuracy_2d"), delta_pp=True)
    row("3D Accuracy（Omni3D）", _metric(off, "official", "accuracy_3d"),
        _metric(on, "official", "accuracy_3d"), delta_pp=True)
    for task in ("Count", "Relation", "Depth", "Distance"):
        row(f"{TASK_LABELS[task]}", _task_acc(off, task), _task_acc(on, task),
            delta_pp=True)
    for source in ("ADE20K", "COCO", "Omni3D"):
        row(f"{source} Accuracy", _source_acc(off, source), _source_acc(on, source),
            delta_pp=True)

    row("全部样本普通 Accuracy", off["accuracy"], on["accuracy"], delta_pp=True)
    row("平均端到端耗时 (s)", _metric(off, "timing_seconds", "end_to_end", "mean"),
        _metric(on, "timing_seconds", "end_to_end", "mean"), formatter=_num)
    row("P95 端到端耗时 (s)", _metric(off, "timing_seconds", "end_to_end", "p95"),
        _metric(on, "timing_seconds", "end_to_end", "p95"), formatter=_num)
    off_e2e = _metric(off, "timing_seconds", "end_to_end", "mean")
    on_e2e = _metric(on, "timing_seconds", "end_to_end", "mean")
    if off_e2e and on_e2e:
        ratio = on_e2e / off_e2e
        lines.append(
            f"| 平均耗时倍率 (On/Off) | 1.00x | {ratio:.2f}x | {ratio - 1:+.2f}x |"
        )
    else:
        lines.append("| 平均耗时倍率 (On/Off) | N/A | N/A | N/A |")
    row("平均生成耗时 (s)", _metric(off, "timing_seconds", "generate", "mean"),
        _metric(on, "timing_seconds", "generate", "mean"), formatter=_num)
    row("平均生成 Token", _metric(off, "tokens", "generated", "mean"),
        _metric(on, "tokens", "generated", "mean"), formatter=_num)
    row("平均 Thinking Token", _metric(off, "tokens", "thinking", "mean"),
        _metric(on, "tokens", "thinking", "mean"), formatter=_num)
    row("Tokens/s（含 Prefill）",
        _metric(off, "throughput", "tokens_per_second_generate"),
        _metric(on, "throughput", "tokens_per_second_generate"), formatter=_num)
    row("单样本峰值显存 (GiB)",
        _metric(off, "vram", "per_sample_peak_allocated_gib_mean"),
        _metric(on, "vram", "per_sample_peak_allocated_gib_mean"), formatter=_num)
    row("Invalid Rate", off["invalid_rate"], on["invalid_rate"])
    row("截断率", off["truncation_rate"], on["truncation_rate"])
    row("有效回答率", off["valid_rate"], on["valid_rate"])
    return lines


def mode_section(name, metrics):
    official = metrics["official"]
    lines = [
        f"### {name}",
        "",
        f"- 完成样本：{metrics['total_samples']}"
        f"（状态：{metrics['experiment'].get('status', 'unknown')}）",
        f"- 官方 CV-Bench：{_pct(official['cvbench_accuracy'])}",
        f"- 官方 2D：{_pct(official['accuracy_2d'])}"
        f"（ADE20K {_pct(official['accuracy_2d_ade20k'])} / "
        f"COCO {_pct(official['accuracy_2d_coco'])}）",
        f"- 官方 3D：{_pct(official['accuracy_3d'])}",
        f"- 全部样本普通 Accuracy：{_pct(metrics['accuracy'])}",
        f"- Invalid：{metrics['invalid']}（{_pct(metrics['invalid_rate'])}）"
        f"；截断：{metrics['truncated']}（{_pct(metrics['truncation_rate'])}）",
        f"- 平均生成 Token：{_num(_metric(metrics, 'tokens', 'generated', 'mean'))}"
        f"；Thinking Token：{_num(_metric(metrics, 'tokens', 'thinking', 'mean'))}",
        f"- 平均生成耗时：{_num(_metric(metrics, 'timing_seconds', 'generate', 'mean'))} s"
        f"；P95：{_num(_metric(metrics, 'timing_seconds', 'generate', 'p95'))} s",
        f"- Tokens/s（Prefill+Decode）："
        f"{_num(_metric(metrics, 'throughput', 'tokens_per_second_generate'))}",
        f"- 单样本峰值显存（平均/最大）："
        f"{_num(_metric(metrics, 'vram', 'per_sample_peak_allocated_gib_mean'))} / "
        f"{_num(_metric(metrics, 'vram', 'per_sample_peak_allocated_gib_max'))} GiB",
    ]
    if metrics.get("invalid_reasons"):
        lines.append(f"- Invalid 原因：`{json.dumps(metrics['invalid_reasons'], ensure_ascii=False)}`")
    if metrics.get("subset"):
        lines.append("- **注意：这是子集结果，不是完整测试集成绩。**")
    lines.append("")
    return lines


def data_conclusions(off, on, comparison) -> list[str]:
    """Factual, number-driven conclusions (no speculation)."""
    if not (off and on):
        return ["Off/On 尚未同时完成，暂不生成对比结论。"]
    o, n = off["official"], on["official"]
    lines = []

    total_off = o.get("cvbench_accuracy")
    total_on = n.get("cvbench_accuracy")
    if total_off is not None and total_on is not None:
        delta = 100 * (total_on - total_off)
        direction = "提升" if delta > 0 else ("下降" if delta < 0 else "不变")
        lines.append(
            f"- 官方 CV-Bench：Off {100 * total_off:.2f}%，On {100 * total_on:.2f}%，"
            f"Thinking {direction} {abs(delta):.2f} pp。"
        )

    for label, key in (("2D", "accuracy_2d"), ("3D", "accuracy_3d")):
        a, b = o.get(key), n.get(key)
        if a is not None and b is not None:
            lines.append(
                f"- {label}：Off {100 * a:.2f}%，On {100 * b:.2f}%，"
                f"变化 {100 * (b - a):+.2f} pp。"
            )

    deltas = comparison.get("task_deltas", {}) if comparison else {}
    improved = [t for t, b in deltas.items()
                if b["accuracy_delta"] is not None and b["accuracy_delta"] > 0]
    worsened = [t for t, b in deltas.items()
                if b["accuracy_delta"] is not None and b["accuracy_delta"] < 0]
    same = [t for t, b in deltas.items()
            if b["accuracy_delta"] is not None and b["accuracy_delta"] == 0]
    if deltas:
        lines.append(
            "- 受益任务：" + (", ".join(TASK_LABELS.get(t, t) for t in improved)
                          if improved else "无") +
            "；下降任务：" + (", ".join(TASK_LABELS.get(t, t) for t in worsened)
                          if worsened else "无") +
            "；持平任务：" + (", ".join(TASK_LABELS.get(t, t) for t in same)
                          if same else "无") + "。"
        )

    off_mean = _metric(off, "timing_seconds", "end_to_end", "mean")
    on_mean = _metric(on, "timing_seconds", "end_to_end", "mean")
    if off_mean and on_mean:
        lines.append(
            f"- 平均端到端耗时：Off {off_mean:.2f}s，On {on_mean:.2f}s，"
            f"On/Off = {on_mean / off_mean:.2f}x。"
        )
    off_tok = _metric(off, "tokens", "generated", "mean")
    on_tok = _metric(on, "tokens", "generated", "mean")
    if off_tok is not None and on_tok is not None:
        lines.append(
            f"- 平均生成 Token：Off {off_tok:.1f}，On {on_tok:.1f}，"
            f"增加 {on_tok - off_tok:.1f}。"
        )
    lines.append(
        f"- Invalid Rate：Off {_pct(off['invalid_rate'])}，"
        f"On {_pct(on['invalid_rate'])}；截断率：Off {_pct(off['truncation_rate'])}，"
        f"On {_pct(on['truncation_rate'])}。"
    )
    if comparison:
        lines.append(
            f"- 同题对比（{comparison['common_samples']} 条）："
            f"Off 错/On 对 {comparison['off_wrong_on_right']} 条，"
            f"Off 对/On 错 {comparison['off_right_on_wrong']} 条，"
            f"预测改变 {comparison['prediction_changed']} 条。"
        )
    return lines


def deep_analysis(off, on, comparison) -> list[str]:
    """Deeper, still purely factual, analysis of the Off/On difference."""
    if not (off and on):
        return []
    lines = []

    def valid_accuracy(metrics):
        valid = metrics.get("valid_answers") or 0
        return metrics["correct"] / valid if valid else None

    va_off, va_on = valid_accuracy(off), valid_accuracy(on)
    if va_off is not None and va_on is not None:
        lines.append(
            f"- 剔除 invalid 后的有效回答准确率：Off {_pct(va_off)}，"
            f"On {_pct(va_on)}（变化 {100 * (va_on - va_off):+.2f} pp）。"
        )
    lines.append(
        f"- Invalid 数量：Off {off['invalid']}（{_pct(off['invalid_rate'])}），"
        f"On {on['invalid']}（{_pct(on['invalid_rate'])}）；"
        f"截断：Off {off['truncated']}，On {on['truncated']}。"
    )

    on_think = _metric(on, "tokens", "thinking", "mean")
    on_gen = _metric(on, "tokens", "generated", "mean")
    off_gen = _metric(off, "tokens", "generated", "mean")
    if on_think and on_gen:
        lines.append(
            f"- On 的 Thinking Token 占全部输出 Token 的 "
            f"{100 * on_think / on_gen:.1f}%（平均 {on_think:.1f} / {on_gen:.1f}）。"
        )
    if off_gen and on_gen:
        lines.append(f"- 输出 Token 倍率 On/Off = {on_gen / off_gen:.2f}x。")

    off_e2e = _metric(off, "timing_seconds", "end_to_end", "mean")
    on_e2e = _metric(on, "timing_seconds", "end_to_end", "mean")
    off_tps = _metric(off, "throughput", "tokens_per_second_generate")
    on_tps = _metric(on, "throughput", "tokens_per_second_generate")
    if off_e2e and on_e2e:
        lines.append(
            f"- 端到端耗时倍率 On/Off = {on_e2e / off_e2e:.2f}x；"
            f"Tokens/s：Off {_num(off_tps)}，On {_num(on_tps)}。"
        )

    lines += [
        "",
        "| 2D 组合分组 | Off | On | 变化 |",
        "| --- | ---: | ---: | ---: |",
    ]
    combos = (
        "2D/Count/ADE20K", "2D/Count/COCO",
        "2D/Relation/ADE20K", "2D/Relation/COCO",
        "3D/Depth/Omni3D", "3D/Distance/Omni3D",
    )
    for key in combos:
        a = off["groups"]["by_type_task_source"].get(key)
        b = on["groups"]["by_type_task_source"].get(key)
        if not a or not b:
            continue
        delta = 100 * (b["accuracy"] - a["accuracy"])
        lines.append(
            f"| {key} | {a['correct']}/{a['n']} ({_pct(a['accuracy'])}) | "
            f"{b['correct']}/{b['n']} ({_pct(b['accuracy'])}) | {delta:+.2f} pp |"
        )
    return lines


def build_report(results_dir: Path, run_name: str = "") -> str:
    root = results_dir / run_name if run_name else results_dir
    off, off_records = _load(root / "thinking_off")
    on, on_records = _load(root / "thinking_on")

    lines = ["# Xiaomi-Lens-0 CV-Bench 评测报告", ""]
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    reference = off or on
    if reference is None:
        return "\n".join(lines + ["未找到任何评测结果。", ""])

    experiment = reference["experiment"]
    lines += [
        "## 实验配置",
        "",
        f"- 模型：`{experiment.get('model_name', 'Xiaomi-Lens-0')}` "
        f"（`{experiment.get('model_path', 'N/A')}`）",
        f"- Attention：`{experiment.get('attn_implementation')}`；dtype：bfloat16",
        f"- max_new_tokens：{experiment.get('max_new_tokens')}；"
        "do_sample：False（贪心，确定性）",
        f"- Prompt 模板：`{experiment.get('prompt_template', '').strip()}`",
        f"- Seed：{experiment.get('seed')}；parser version：{experiment.get('parser_version')}",
        f"- 模型加载耗时：{_num(experiment.get('model_load_seconds'))} s；"
        f"预热：{_num(experiment.get('warmup_seconds'))} s",
        f"- 整卡显存（结束时）：{_num(experiment.get('gpu_used_gib_end'))} GiB 已使用"
        "（包含机器上的其他 GPU 进程，不能与 PyTorch 峰值混为一谈）",
        f"- 运行状态：{experiment.get('status', 'unknown')}，"
        f"完成 {experiment.get('completed_samples', 'N/A')}/"
        f"{experiment.get('expected_samples', 'N/A')}",
        "- 注意：本机 GPU 同时运行其他用户的进程，绝对耗时受资源竞争影响；"
        "Off/On 在同一台机器上顺序运行，相对差异仍可参考，但不宜当作独占 GPU 的基准。",
        "",
    ]

    distribution = reference.get("dataset_distribution", {})
    if distribution:
        lines += ["## 数据集分布", "", "```json",
                  json.dumps(distribution, indent=2, ensure_ascii=False), "```", ""]

    lines += ["## 主对比表", ""]
    if off and on:
        lines += comparison_table(off, on)
        lines += [
            "",
            "> `pp` = percentage points。Thinking Token 按 `</think>` 特殊 Token 精确切分；"
            "Tokens/s 包含 Prefill 与 Decode，不是纯 Decode 速度。",
            "",
        ]
    else:
        lines += ["缺少 Off 或 On 结果，无法生成对比表。", ""]

    if off:
        lines += ["## 分模式明细", ""] + mode_section("Thinking Off", off)
    if on:
        lines += mode_section("Thinking On", on)

    comparison_path = root / "comparison.json"
    comparison = None
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    elif off_records and on_records:
        comparison = compare_runs(off_records, on_records)
    if comparison:
        lines += [
            "## 同题对比（Off vs On）",
            "",
            f"- 共同样本：{comparison['common_samples']}",
            f"- Off 错、On 对：{comparison['off_wrong_on_right']}",
            f"- Off 对、On 错：{comparison['off_right_on_wrong']}",
            f"- 预测发生变化：{comparison['prediction_changed']}",
            "",
            "| 任务 | Off 正确/总数 | On 正确/总数 | Accuracy 变化 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for task, block in comparison["task_deltas"].items():
            delta = block["accuracy_delta"]
            delta_text = "N/A" if delta is None else f"{100 * delta:+.2f} pp"
            lines.append(
                f"| {TASK_LABELS.get(task, task)} | "
                f"{block['off']['correct']}/{block['off']['n']} | "
                f"{block['on']['correct']}/{block['on']['n']} | {delta_text} |"
            )
        lines.append("")

    lines += ["## 数据结论", ""]
    lines += data_conclusions(off, on, comparison)
    lines.append("")

    lines += ["## 深入解读", ""]
    lines += deep_analysis(off, on, comparison)
    lines.append("")

    lines += [
        "## 局限与注意事项", "",
        "- 本机 GPU 同时运行其他用户的 DRL/CityNav 进程，绝对耗时受资源竞争影响；"
        "Off/On 是顺序运行而非独占 GPU 的微基准，耗时倍率应视为近似值。",
        "- 达到 `max_new_tokens=8192` 的样本记为截断，严格解析下通常没有最终答案，"
        "按错误计入分母；没有做“补答”或从 Thinking 中间过程猜答案。",
        "- 模型可能自发输出 `<bbox>` / `<|mask|>`；本评测只做文本选择题，"
        "不执行 Mask 解码或可视化，这些 token 只计入生成量。",
        "- Invalid 与截断均保留在准确率分母中，因此官方分数低于“只统计有效回答”的分数。",
        "",
        "## Adaptive Thinking",
        "",
        "Xiaomi-Lens-0 **不支持原生 Adaptive Thinking**。该 checkpoint 的 chat template "
        "只暴露布尔参数 `enable_thinking`，没有 adaptive/auto 开关，也没有 Thinking "
        "Budget 参数。因此本报告不包含第三组分数，也未用固定 Token 预算或外部规则"
        "冒充原生自适应模式。",
        "",
        "## 结论",
        "",
        "- 所有准确率均来自严格解析后的 `FINAL_ANSWER: (X)` 行；无效输出按错误计入分母。",
        "- 官方总分使用 `(2D_ADE20K + 2D_COCO)/2` 与 `3D_Omni3D` 的加权公式，"
        "不是全样本简单平均。",
        "- 若任一官方分组缺失，对应官方分数标记为不可计算（N/A）。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(HERE / "results"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report = build_report(results_dir, args.run_name)
    output = Path(args.output) if args.output else (
        (results_dir / args.run_name if args.run_name else results_dir)
        / "comparison_report.md"
    )
    output.write_text(report, encoding="utf-8")
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
