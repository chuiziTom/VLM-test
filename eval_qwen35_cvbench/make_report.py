"""从已保存的结果生成中文对比报告。

报告包含两部分：

A. Qwen3.5-4B 自身的 Thinking Off / On 对比。
B. Qwen3.5-4B 与 Xiaomi-Lens-0 的横向对比（只读读取 Xiaomi 目录，
   绝不写入、修改或覆盖对方的任何文件）。

用法::

    micromamba run -n vlm-test python eval_qwen35_cvbench/make_report.py
    # 只跑小样本时指定 run-name：
    micromamba run -n vlm-test python eval_qwen35_cvbench/make_report.py --run-name smoke16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import compare_runs, compute_metrics  # noqa: E402

DEFAULT_XIAOMI_DIR = Path(
    "/home/20261024_1/Workspace/qyh/VLM-test/Xiaomi-Lens-0/eval_xiaomi_cvbench/results"
)

TASK_LABELS = {
    "Count": "物体计数 (Count)",
    "Relation": "二维空间关系 (Relation)",
    "Depth": "深度顺序 (Depth)",
    "Distance": "相对距离 (Distance)",
}
TASK_ORDER = ("Count", "Relation", "Depth", "Distance")
SOURCE_ORDER = ("ADE20K", "COCO", "Omni3D")


# ------------------------------------------------------------------ 读取
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_mode(mode_dir: Path):
    """读取一次运行；若 metrics.json 不存在返回 (None, [])。"""
    metrics_path = mode_dir / "metrics.json"
    if not metrics_path.exists():
        return None, []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    records = _read_jsonl(mode_dir / "predictions.jsonl")
    # 用同一份 metrics.py 从逐样本记录重算一遍，防止报告与原始数据脱节。
    if records:
        metrics["_recomputed"] = compute_metrics(records, metrics.get("experiment"))
    return metrics, records


# ------------------------------------------------------------------ 格式化
def _pct(value, digits=2):
    return "N/A" if value is None else f"{100 * value:.{digits}f}%"


def _num(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _metric(metrics, *keys, default=None):
    current = metrics
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _task_acc(metrics, task):
    return metrics["groups"]["by_task"].get(task, {}).get("accuracy")


def _source_acc(metrics, source):
    return metrics["groups"]["by_source"].get(source, {}).get("accuracy")


def _block_n(metrics, group, key):
    return metrics["groups"][group].get(key, {}).get("n")


# ------------------------------------------------------- A. Off/On 对比表
def off_on_table(off, on):
    lines = [
        "| 指标 | Thinking Off | Thinking On | On - Off |",
        "| --- | ---: | ---: | ---: |",
    ]

    def row(label, off_value, on_value, formatter=_pct, delta_pp=False):
        off_text, on_text = formatter(off_value), formatter(on_value)
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
    for task in TASK_ORDER:
        row(TASK_LABELS[task], _task_acc(off, task), _task_acc(on, task),
            delta_pp=True)
    for source in SOURCE_ORDER:
        row(f"{source} Accuracy", _source_acc(off, source),
            _source_acc(on, source), delta_pp=True)

    row("全部样本普通 Accuracy", off["accuracy"], on["accuracy"], delta_pp=True)
    row("平均端到端耗时 (s)", _metric(off, "timing_seconds", "end_to_end", "mean"),
        _metric(on, "timing_seconds", "end_to_end", "mean"), formatter=_num)
    row("P95 端到端耗时 (s)", _metric(off, "timing_seconds", "end_to_end", "p95"),
        _metric(on, "timing_seconds", "end_to_end", "p95"), formatter=_num)
    off_e2e = _metric(off, "timing_seconds", "end_to_end", "mean")
    on_e2e = _metric(on, "timing_seconds", "end_to_end", "mean")
    lines.append(
        f"| 平均耗时倍率 (On/Off) | 1.00x | {on_e2e / off_e2e:.2f}x | "
        f"{on_e2e / off_e2e - 1:+.2f}x |" if off_e2e and on_e2e
        else "| 平均耗时倍率 (On/Off) | N/A | N/A | N/A |"
    )
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
    experiment = metrics.get("experiment", {})
    lines = [
        f"### {name}",
        "",
        f"- 完成样本：{metrics['total_samples']}"
        f"（状态：{experiment.get('status', 'unknown')}）",
        f"- 官方 CV-Bench：{_pct(official['cvbench_accuracy'])}",
        f"- 官方 2D：{_pct(official['accuracy_2d'])}"
        f"（ADE20K {_pct(official['accuracy_2d_ade20k'])} / "
        f"COCO {_pct(official['accuracy_2d_coco'])}）",
        f"- 官方 3D：{_pct(official['accuracy_3d'])}",
        f"- 全部样本普通 Accuracy：{_pct(metrics['accuracy'])}"
        f"（{metrics['correct']}/{metrics['total_samples']}）",
        f"- Invalid：{metrics['invalid']}（{_pct(metrics['invalid_rate'])}）"
        f"；截断：{metrics['truncated']}（{_pct(metrics['truncation_rate'])}）"
        f"；推理失败：{metrics['errors']}",
        f"- 平均生成 Token：{_num(_metric(metrics, 'tokens', 'generated', 'mean'))}"
        f"（P95 {_num(_metric(metrics, 'tokens', 'generated', 'p95'))}）"
        f"；Thinking Token：{_num(_metric(metrics, 'tokens', 'thinking', 'mean'))}",
        f"- 平均生成耗时：{_num(_metric(metrics, 'timing_seconds', 'generate', 'mean'))} s"
        f"；P95：{_num(_metric(metrics, 'timing_seconds', 'generate', 'p95'))} s",
        f"- Tokens/s（Prefill+Decode）："
        f"{_num(_metric(metrics, 'throughput', 'tokens_per_second_generate'))}",
        f"- 单样本峰值显存（平均/最大）："
        f"{_num(_metric(metrics, 'vram', 'per_sample_peak_allocated_gib_mean'))} / "
        f"{_num(_metric(metrics, 'vram', 'per_sample_peak_allocated_gib_max'))} GiB",
    ]
    # 用逐样本记录重算一遍，和 metrics.json 对照，防止报告数字与原始数据脱节。
    recomputed = metrics.get("_recomputed")
    if recomputed:
        stored_acc = metrics.get("accuracy")
        if stored_acc is not None and abs(stored_acc - recomputed["accuracy"]) > 1e-12:
            lines.append(
                f"- ⚠️ 重算 Accuracy {_pct(recomputed['accuracy'])} 与 "
                f"metrics.json 的 {_pct(stored_acc)} 不一致，请检查结果文件"
            )
    if metrics.get("invalid_reasons"):
        lines.append(f"- Invalid 原因：`{json.dumps(metrics['invalid_reasons'], ensure_ascii=False)}`")
    if metrics.get("subset"):
        lines.append("- **注意：这是子集结果，不是完整测试集成绩。**")
    lines.append("")
    return lines


# ------------------------------------------------- B. 与 Xiaomi 的横向对比
def _check_protocol(qwen, xiaomi) -> list[str]:
    """比对两套运行的协议参数，任何差异都要在报告里说清楚。"""
    notes = []
    q_experiment = qwen.get("experiment", {})
    x_experiment = xiaomi.get("experiment", {})

    def both(key):
        return q_experiment.get(key), x_experiment.get(key)

    q_prompt, x_prompt = both("prompt_template_sha1")
    if q_prompt and x_prompt:
        notes.append(
            "- Prompt 模板：两者 sha1 相同 ✅" if q_prompt == x_prompt
            else f"- Prompt 模板：**不同**（Qwen `{q_prompt[:8]}` / "
                 f"Xiaomi `{x_prompt[:8]}`）⚠️"
        )
    q_parser, x_parser = both("parser_version")
    if q_parser and x_parser:
        notes.append(
            f"- 答案解析协议：版本 {q_parser} vs {x_parser}"
            + ("（相同 ✅）" if q_parser == x_parser else "（**不同** ⚠️）")
        )
    q_max, x_max = both("max_new_tokens")
    if q_max or x_max:
        notes.append(
            f"- max_new_tokens：Qwen {q_max} / Xiaomi {x_max}"
            + ("（相同 ✅）" if q_max == x_max else "（**不同** ⚠️，Thinking On 的截断率不可直接比较）")
        )
    q_attn, x_attn = both("attn_implementation")
    if q_attn or x_attn:
        notes.append(f"- Attention 实现：Qwen `{q_attn}` / Xiaomi `{x_attn}`（不同实现会影响速度，不影响准确率）")
    q_shared = q_experiment.get("gpu_shared")
    x_shared = x_experiment.get("gpu_shared")
    if q_shared is not None or x_shared is not None:
        notes.append(
            f"- 共享 GPU 标记：Qwen `{q_shared}` / Xiaomi `{x_shared}`。"
            "任一为 True 时，耗时/吞吐/显存都属于**共享 GPU 条件下的测量值**，"
            "不能当作独占 GPU 的性能基准。"
        )
    return notes


def cross_model_table(qwen_off, xiaomi_off, qwen_on, xiaomi_on):
    columns = [
        ("Xiaomi-Lens-0 Off", xiaomi_off),
        ("Qwen3.5-4B Off", qwen_off),
        ("Xiaomi-Lens-0 On", xiaomi_on),
        ("Qwen3.5-4B On", qwen_on),
    ]
    lines = [
        "| 指标 | " + " | ".join(name for name, _ in columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]

    def row(label, getter, formatter=_pct, with_counts=False):
        cells = []
        for _, metrics in columns:
            if metrics is None:
                cells.append("N/A")
                continue
            value = getter(metrics)
            text = formatter(value)
            if with_counts:
                counts = getter.counts(metrics) if hasattr(getter, "counts") else None
                if counts:
                    text = f"{counts[0]}/{counts[1]} ({text})"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("CV-Bench 官方总分", lambda m: _metric(m, "official", "cvbench_accuracy"))
    row("2D Accuracy", lambda m: _metric(m, "official", "accuracy_2d"))
    row("3D Accuracy", lambda m: _metric(m, "official", "accuracy_3d"))

    def task_with_counts(task):
        def getter(m):
            return _task_acc(m, task)

        def counts(m):
            return (_block_n(m, "by_task", task) and
                    (m["groups"]["by_task"][task]["correct"],
                     m["groups"]["by_task"][task]["n"]))
        getter.counts = counts
        return getter

    for task in TASK_ORDER:
        row(TASK_LABELS[task], task_with_counts(task), with_counts=True)
    for source in SOURCE_ORDER:
        row(f"{source} Accuracy", lambda m, s=source: _source_acc(m, s))

    row("Invalid Rate", lambda m: m.get("invalid_rate"))
    row("截断率", lambda m: m.get("truncation_rate"))
    row("平均端到端耗时 (s)",
        lambda m: _metric(m, "timing_seconds", "end_to_end", "mean"), _num)
    row("P95 端到端耗时 (s)",
        lambda m: _metric(m, "timing_seconds", "end_to_end", "p95"), _num)
    row("平均生成 Token",
        lambda m: _metric(m, "tokens", "generated", "mean"), _num)
    row("Tokens/s（含 Prefill）",
        lambda m: _metric(m, "throughput", "tokens_per_second_generate"), _num)
    row("单样本峰值显存 (GiB)",
        lambda m: _metric(m, "vram", "per_sample_peak_allocated_gib_mean"), _num)
    row("总样本数", lambda m: m.get("total_samples"), lambda v: "N/A" if v is None else str(v))
    return lines


def cross_sample_diff(qwen_records, xiaomi_records):
    """按 uid 对齐两个模型的逐样本结果，统计谁在哪些题上更强。"""
    q_by_uid = {r["uid"]: r for r in qwen_records if not r.get("error")}
    x_by_uid = {r["uid"]: r for r in xiaomi_records if not r.get("error")}
    common = sorted(set(q_by_uid) & set(x_by_uid))
    if not common:
        return None

    stats = {
        "common": len(common),
        "both_correct": 0, "both_wrong": 0,
        "qwen_only": 0, "xiaomi_only": 0,
        "by_task": {},
    }
    for uid in common:
        q_correct = bool(q_by_uid[uid].get("correct"))
        x_correct = bool(x_by_uid[uid].get("correct"))
        if q_correct and x_correct:
            stats["both_correct"] += 1
        elif not q_correct and not x_correct:
            stats["both_wrong"] += 1
        elif q_correct:
            stats["qwen_only"] += 1
        else:
            stats["xiaomi_only"] += 1
        task = q_by_uid[uid].get("task")
        block = stats["by_task"].setdefault(
            task, {"n": 0, "qwen_only": 0, "xiaomi_only": 0}
        )
        block["n"] += 1
        if q_correct and not x_correct:
            block["qwen_only"] += 1
        elif x_correct and not q_correct:
            block["xiaomi_only"] += 1
    return stats


# ------------------------------------------------------------------ 组装
def build_report(results_root: Path, xiaomi_root: Path | None) -> str:
    qwen_off, qwen_off_records = _load_mode(results_root / "thinking_off")
    qwen_on, qwen_on_records = _load_mode(results_root / "thinking_on")

    lines = ["# Qwen3.5-4B × CV-Bench 评测报告", ""]
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    reference = qwen_off or qwen_on
    if reference is None:
        return "\n".join(lines + ["未找到任何 Qwen 评测结果。", ""])

    experiment = reference.get("experiment", {})
    lines += [
        "## 实验配置",
        "",
        f"- 模型：`{experiment.get('model_name', 'Qwen3.5-4B')}` "
        f"（`{experiment.get('model_resolved_path', 'N/A')}`）",
        f"- Attention：`{experiment.get('attn_implementation')}`；dtype：bfloat16；"
        f"CPU offload：`{experiment.get('cpu_offload')}`",
        f"- max_new_tokens：{experiment.get('max_new_tokens')}；"
        "do_sample：False（贪心，确定性）",
        f"- Prompt 模板：`{str(experiment.get('prompt_template', '')).strip()}`",
        f"- Seed：{experiment.get('seed')}；parser version：{experiment.get('parser_version')}",
        f"- 模型加载耗时：{_num(experiment.get('model_load_seconds'))} s；"
        f"预热：{_num(experiment.get('warmup_seconds'))} s",
        f"- 运行状态：{experiment.get('status', 'unknown')}，"
        f"完成 {experiment.get('completed_samples', 'N/A')}/"
        f"{experiment.get('expected_samples', 'N/A')}",
        f"- 是否与他人共享 GPU：`{experiment.get('gpu_shared')}`",
        "",
    ]

    distribution = reference.get("dataset_distribution", {})
    if distribution:
        lines += ["## 数据集分布", "", "```json",
                  json.dumps(distribution, indent=2, ensure_ascii=False), "```", ""]

    # ---------------------------------------------------------- A. 自身对比
    lines += ["## A. Qwen3.5-4B：Thinking Off vs On", ""]
    if qwen_off and qwen_on:
        lines += off_on_table(qwen_off, qwen_on)
        lines += [
            "",
            "> `pp` = percentage points。Thinking Token 按 `</think>` 特殊 Token 精确切分；",
            "> Tokens/s 包含 Prefill 与 Decode，不是纯 Decode 速度。",
            "",
        ]
    else:
        missing = []
        if not qwen_off:
            missing.append("Thinking Off")
        if not qwen_on:
            missing.append("Thinking On")
        lines += [f"缺少 {', '.join(missing)} 的结果，暂时无法生成 Off/On 对比表。", ""]

    if qwen_off:
        lines += ["## 分模式明细", ""] + mode_section("Thinking Off", qwen_off)
    if qwen_on:
        lines += mode_section("Thinking On", qwen_on)

    comparison = None
    comparison_path = results_root / "comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    elif qwen_off_records and qwen_on_records:
        comparison = compare_runs(qwen_off_records, qwen_on_records)

    if comparison:
        lines += [
            "## 同题对比（Qwen Off vs On）",
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

    # ------------------------------------------------------ B. 横向对比
    lines += ["## B. Qwen3.5-4B vs Xiaomi-Lens-0", ""]
    if xiaomi_root is None:
        lines += ["未指定 Xiaomi 结果目录，跳过横向对比。", ""]
    else:
        x_off, x_off_records = _load_mode(xiaomi_root / "thinking_off")
        x_on, x_on_records = _load_mode(xiaomi_root / "thinking_on")
        if x_off is None and x_on is None:
            lines += [f"在 `{xiaomi_root}` 未找到 Xiaomi 的结果，跳过横向对比。", ""]
        else:
            incomplete = []
            for name, metrics in (("Thinking Off", x_off), ("Thinking On", x_on)):
                if metrics is None:
                    incomplete.append(f"{name} 缺失")
                elif metrics.get("experiment", {}).get("status") != "completed":
                    incomplete.append(
                        f"{name} 状态为 "
                        f"`{metrics.get('experiment', {}).get('status')}`"
                        f"（{metrics['total_samples']} 条）"
                    )
            if incomplete:
                lines += [
                    "> ⚠️ **Xiaomi-Lens-0 的完整评测尚未结束**："
                    + "；".join(incomplete) + "。",
                    "> 下表中的 Xiaomi 数字只是**当前进度下的部分结果**，"
                    "不能当作它的正式成绩，也不能据此判断两个模型谁更强。",
                    "> 等 Xiaomi 跑完后重新执行本脚本即可得到最终对比。",
                    "",
                ]
            lines += ["### 协议一致性检查", ""]
            reference_x = x_off or x_on
            reference_q = qwen_off or qwen_on
            lines += _check_protocol(reference_q, reference_x)
            lines.append("")
            lines += ["### 主对比表", ""]
            lines += cross_model_table(qwen_off, x_off, qwen_on, x_on)
            lines.append("")

            # 逐样本差异：用两侧都使用相同 uid 的样本对齐。
            q_records = qwen_off_records or qwen_on_records
            x_records = x_off_records or x_on_records
            diff = cross_sample_diff(q_records, x_records) if (
                q_records and x_records) else None
            if diff:
                lines += [
                    f"### 逐样本差异（Thinking Off，按 uid 对齐，共同 {diff['common']} 条）",
                    "",
                    f"- 两者都答对：{diff['both_correct']}",
                    f"- 两者都答错：{diff['both_wrong']}",
                    f"- 只有 Qwen 答对：{diff['qwen_only']}",
                    f"- 只有 Xiaomi 答对：{diff['xiaomi_only']}",
                    "",
                    "| 任务 | 共同样本 | 只有 Qwen 对 | 只有 Xiaomi 对 |",
                    "| --- | ---: | ---: | ---: |",
                ]
                for task, block in sorted(diff["by_task"].items()):
                    lines.append(
                        f"| {TASK_LABELS.get(task, task)} | {block['n']} | "
                        f"{block['qwen_only']} | {block['xiaomi_only']} |"
                    )
                lines += [
                    "",
                    "> 该统计只覆盖两边都已完成评测的样本；"
                    "Xiaomi 未跑完时，共同样本数会小于 2638。",
                    "",
                ]

    # ------------------------------------------------------------ 说明
    lines += [
        "## Adaptive Thinking",
        "",
        "Qwen3.5-4B **不支持原生 Adaptive Thinking**。其 chat template 只有布尔参数 "
        "`enable_thinking`，没有 adaptive/auto 开关，也没有 Thinking Budget 参数。"
        "因此本报告不含第三组分数，也没有用固定 Token 预算或外部路由冒充原生自适应模式。",
        "",
        "## 可比较性说明",
        "",
        "- 两个模型的准确率使用**逐字节相同**的解析器与指标代码"
        "（`answer_parser.py`、`metrics.py` 的 sha1 与 Xiaomi 目录一致），"
        "Prompt 格式指令也完全相同，"
        "因此准确率可以直接比较。",
        "- 推理速度、吞吐与显存**不能**直接比较：两个模型架构不同（Qwen3.5-4B 与 "
        "Xiaomi-Lens-0 参数量不同），Attention 实现不同，而且都在与他人共享的 GPU 上运行。",
        "- 所有准确率均来自严格解析后的 `FINAL_ANSWER: (X)` 行；无效输出按错误计入分母。",
        "- 官方总分使用 `(2D_ADE20K + 2D_COCO)/2` 与 `3D_Omni3D` 的加权公式，"
        "不是全样本简单平均；任一官方分组缺失时该分数记为 N/A。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(HERE / "results"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--xiaomi-dir", default=str(DEFAULT_XIAOMI_DIR),
                        help="Xiaomi 结果目录（只读）。传空字符串可跳过横向对比。")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    root = results_dir / args.run_name if args.run_name else results_dir
    xiaomi_root = Path(args.xiaomi_dir) if args.xiaomi_dir else None

    report = build_report(root, xiaomi_root)
    output = Path(args.output) if args.output else root / "comparison_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"报告已写入 {output}")


if __name__ == "__main__":
    main()
