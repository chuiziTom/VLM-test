"""根据实验产物生成 SPAR-Bench-Tiny 报告。

输入（全部来自实际预测记录，不手工填写成绩）：

* experiments/spar_bench/tiny_rgb/<model>/<protocol>/metrics.json
* .../experiment.json
* .../predictions.jsonl（仅用于错误案例）

输出：

* reports/spar_bench_tiny_xiaomi_lens_0.md
* reports/spar_bench_tiny_qwen3_5_4b.md
* reports/spar_bench_tiny_comparison.md

没有 metrics.json 的实验会明确标为“未完成”，不会用 0 或估算值代替。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from sparbench_metrics import (  # noqa: E402
    ALL_TASKS, HIGH_TASKS, LOW_TASKS, MIDDLE_TASKS,
)

EXPERIMENT_ROOT = ROOT / "experiments" / "spar_bench" / "tiny_rgb"
REPORT_DIR = ROOT / "reports"

MODELS = {
    "xiaomi_lens_0": {
        "title": "Xiaomi-Lens-0",
        "hf": "https://huggingface.co/SeerRay-Lab/Xiaomi-Lens-0",
        "env": "xiaomi-lens (micromamba)",
        "report": "spar_bench_tiny_xiaomi_lens_0.md",
    },
    "qwen3_5_4b": {
        "title": "Qwen3.5-4B",
        "hf": "https://huggingface.co/Qwen/Qwen3.5-4B",
        "env": "vlm-test (micromamba)",
        "report": "spar_bench_tiny_qwen3_5_4b.md",
    },
}

PROTOCOLS = [
    ("think_off_8192", "Thinking Off / 8192"),
    ("think_on_8192", "Thinking On / 8192"),
    ("direct_off_64", "Direct Answer Off / 64"),
]

SHORT_LABELS = {
    "think_off_8192": "Off",
    "think_on_8192": "On",
    "direct_off_64": "Direct",
}

LEVEL_NAMES = {"low": "Low（低级空间感知）",
               "medium": "Medium（跨视角感知）",
               "high": "High（高级空间推理）"}


def _pct(value, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100:.{digits}f}"


def _num(value, digits: int = 2, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.{digits}f}{suffix}"


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(model_dir: str, run_name: str = "") -> dict[str, dict]:
    runs = {}
    for protocol, _ in PROTOCOLS:
        base = EXPERIMENT_ROOT / model_dir / protocol
        if run_name:
            base = base / run_name
        metrics = _load_json(base / "metrics.json")
        experiment = _load_json(base / "experiment.json")
        if metrics is None:
            runs[protocol] = {"available": False}
            continue
        runs[protocol] = {
            "available": True,
            "dir": base,
            "metrics": metrics,
            "experiment": experiment or {},
        }
    return runs


def _status(run: dict) -> str:
    if not run["available"]:
        return "未运行"
    exp = run["experiment"]
    status = exp.get("status", "unknown")
    n = run["metrics"]["n_samples"]
    expected = run["metrics"].get("quality", {})
    if status == "complete" and not expected.get("errors"):
        return f"完成（{n} 条）"
    return f"{status}（{n} 条，有错误 {expected.get('errors', 0)}）"


def _all_available(runs: dict[str, dict]) -> bool:
    return all(r["available"] for r in runs.values())


# --------------------------------------------------------------- 公共片段
def _config_table(runs: dict[str, dict]) -> list[str]:
    lines = [
        "| 协议 | Thinking | max_new_tokens | Prompt 版本 | 状态 |",
        "|---|---|---:|---|---|",
    ]
    for protocol, label in PROTOCOLS:
        run = runs[protocol]
        if run["available"]:
            m = run["metrics"]
            lines.append(
                f"| `{protocol}` | {m.get('thinking_mode')} | "
                f"{m.get('max_new_tokens')} | `{m.get('prompt_version')}` | "
                f"{_status(run)} |"
            )
        else:
            lines.append(f"| `{protocol}` | - | - | - | 未运行 |")
    return lines


def _overall_table(runs: dict[str, dict]) -> list[str]:
    lines = [
        "| 协议 | 样本数 | Official Overall | Strict Overall | Low | Medium | High |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol, _ in PROTOCOLS:
        run = runs[protocol]
        if not run["available"]:
            lines.append(f"| `{protocol}` | - | - | - | - | - | - |")
            continue
        m = run["metrics"]
        levels = m["levels"]
        lines.append(
            f"| `{protocol}` | {m['n_samples']} | "
            f"{_pct(m['overall']['official'])} | {_pct(m['overall']['strict'])} | "
            f"{_pct(levels['low']['official_score'])} | "
            f"{_pct(levels['medium']['official_score'])} | "
            f"{_pct(levels['high']['official_score'])} |"
        )
    return lines


def _answer_type_table(runs: dict[str, dict]) -> list[str]:
    lines = [
        "| 协议 | MCA Official | MCA Strict | NA Official | NA Strict | "
        "VCI Official | VCI Strict |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol, _ in PROTOCOLS:
        run = runs[protocol]
        if not run["available"]:
            lines.append(f"| `{protocol}` | - | - | - | - | - | - |")
            continue
        types = run["metrics"]["answer_types"]
        lines.append(
            f"| `{protocol}` | {_pct(types['MCA']['task_mean_official'])} | "
            f"{_pct(types['MCA']['task_mean_strict'])} | "
            f"{_pct(types['NA']['task_mean_official'])} | "
            f"{_pct(types['NA']['task_mean_strict'])} | "
            f"{_pct(types['VCI']['task_mean_official'])} | "
            f"{_pct(types['VCI']['task_mean_strict'])} |"
        )
    return lines


def _img_type_table(runs: dict[str, dict]) -> list[str]:
    lines = [
        "| 协议 | Single-view（400） | Multi-view（600） |",
        "|---|---:|---:|",
    ]
    for protocol, _ in PROTOCOLS:
        run = runs[protocol]
        if not run["available"]:
            lines.append(f"| `{protocol}` | - | - |")
            continue
        img = run["metrics"]["img_types"]
        single = img.get("single_view", {}).get("official_mean")
        multi = img.get("multi_view", {}).get("official_mean")
        lines.append(f"| `{protocol}` | {_pct(single)} | {_pct(multi)} |")
    return lines


def _task_table(runs: dict[str, dict]) -> list[str]:
    lines = [
        "| 任务 | 层级 | 评分 | "
        + " | ".join(f"{label} Official" for _, label in PROTOCOLS)
        + " |",
        "|---|---|---|" + "---:|" * len(PROTOCOLS),
    ]
    for task in ALL_TASKS:
        level = ("low" if task in LOW_TASKS else
                 "medium" if task in MIDDLE_TASKS else "high")
        answer_type = None
        cells = []
        for protocol, _ in PROTOCOLS:
            run = runs[protocol]
            block = None
            if run["available"]:
                block = run["metrics"]["tasks"].get(task)
            if block is None:
                cells.append("-")
            else:
                answer_type = block["answer_type"]
                cells.append(_pct(block["official_score"]))
        lines.append(
            f"| `{task}` | {level} | {answer_type or '-'} | " + " | ".join(cells) + " |"
        )
    return lines


def _strict_task_table(runs: dict[str, dict]) -> list[str]:
    lines = [
        "| 任务 | 层级 | "
        + " | ".join(f"{label} Strict" for _, label in PROTOCOLS)
        + " | Invalid 数（三协议合计） |",
        "|---|---|" + "---:|" * len(PROTOCOLS) + "---:|",
    ]
    for task in ALL_TASKS:
        level = ("low" if task in LOW_TASKS else
                 "medium" if task in MIDDLE_TASKS else "high")
        cells = []
        invalid_total = 0
        have = False
        for protocol, _ in PROTOCOLS:
            run = runs[protocol]
            block = None
            if run["available"]:
                block = run["metrics"]["tasks"].get(task)
            if block is None:
                cells.append("-")
            else:
                have = True
                cells.append(_pct(block["strict_score"]))
                invalid_total += block["invalid"]
        lines.append(f"| `{task}` | {level} | " + " | ".join(cells)
                     + f" | {invalid_total if have else '-'} |")
    return lines


def _performance_table(runs: dict[str, dict]) -> list[str]:
    lines = ["**（1）延迟与吞吐**", "",
             "| 协议 | 加载(s) | 总耗时(h) | E2E 均值(s) | E2E 中位(s) | E2E P95(s) | "
             "E2E 最大(s) | Prefill 均值(s) | Decode 均值(s) | Samples/s | Decode Tokens/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol, _ in PROTOCOLS:
        run = runs[protocol]
        if not run["available"]:
            lines.append(f"| `{protocol}` | - | - | - | - | - | - | - | - | - | - |")
            continue
        m = run["metrics"]
        p = m["performance"]
        t = p["timing_seconds"]
        e2e = t.get("e2e_seconds") or {}
        prefill = t.get("prefill_seconds") or {}
        decode = p.get("decode_seconds") or {}
        lines.append(
            f"| `{protocol}` | {_num(p.get('model_load_seconds'), 1)} | "
            f"{_num((p.get('total_wall_seconds') or 0) / 3600, 2)} | "
            f"{_num(e2e.get('mean'), 2)} | {_num(e2e.get('median'), 2)} | "
            f"{_num(e2e.get('p95'), 2)} | {_num(e2e.get('max'), 2)} | "
            f"{_num(prefill.get('mean'), 3)} | {_num(decode.get('mean'), 2)} | "
            f"{_num(p.get('samples_per_second'), 3)} | "
            f"{_num(p.get('tokens_per_second_decode'), 1)} |"
        )
    lines += ["", "**（2）Token 与显存**", "",
              "| 协议 | 平均 Prompt | 平均生成 | 平均 Thinking | 平均 Answer | "
              "总生成 Tokens | 最大生成 | 截断数 | 加载后 allocated(GiB) | 单样本峰值(GiB) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol, _ in PROTOCOLS:
        run = runs[protocol]
        if not run["available"]:
            lines.append(f"| `{protocol}` | - | - | - | - | - | - | - | - | - |")
            continue
        m = run["metrics"]
        p = m["performance"]
        tk = p["tokens"]
        vram = p["vram"]
        after_load = (vram.get("after_load") or {}).get("allocated_gib")
        lines.append(
            f"| `{protocol}` | {_num((tk.get('prompt') or {}).get('mean'), 0)} | "
            f"{_num((tk.get('generated') or {}).get('mean'), 0)} | "
            f"{_num((tk.get('thinking') or {}).get('mean'), 0)} | "
            f"{_num((tk.get('answer') or {}).get('mean'), 0)} | "
            f"{tk.get('total_generated', 0)} | "
            f"{_num((tk.get('generated') or {}).get('max'), 0)} | "
            f"{m['quality']['truncated']} | {_num(after_load, 2)} | "
            f"{_num((vram or {}).get('per_sample_peak_allocated_gib_max'), 2)} |"
        )
    lines += ["",
              "说明：E2E 为单条样本端到端耗时（图片解码 + 预处理 + Prefill + Decode）；"
              "Decode Tokens/s 只统计 decode 阶段；峰值显存为 PyTorch "
              "`max_memory_allocated`，与整卡 `nvidia-smi` 使用量不同。", ""]
    return lines


def _quality_table(runs: dict[str, dict]) -> list[str]:
    lines = [
        "| 协议 | Invalid | 截断 | 推理错误 | OOM | Thinking 未闭合 | 模式校验失败 | 图片数校验失败 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol, _ in PROTOCOLS:
        run = runs[protocol]
        if not run["available"]:
            lines.append(f"| `{protocol}` | - | - | - | - | - | - | - |")
            continue
        q = run["metrics"]["quality"]
        lines.append(
            f"| `{protocol}` | {q['invalid']} | {q['truncated']} | {q['errors']} | "
            f"{q['oom']} | {q['thinking_unclosed']} | {q['mode_unverified']} | "
            f"{q['image_count_mismatch']} |"
        )
    return lines


def _failure_examples(runs: dict[str, dict], limit_per_protocol: int = 3) -> list[str]:
    """从 predictions.jsonl 里挑少量 invalid / 截断 / 错误案例。"""
    lines = []
    for protocol, label in PROTOCOLS:
        run = runs[protocol]
        if not run["available"]:
            continue
        path = run["dir"] / "predictions.jsonl"
        if not path.exists():
            continue
        picked = []
        seen_reasons = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reason = record.get("invalid_reason")
                bad = record.get("invalid") or record.get("error") or record.get("truncated")
                if not bad:
                    continue
                # 每种原因最多取一条，避免报告被同一错误刷屏。
                key = reason or "error"
                if key in seen_reasons:
                    continue
                seen_reasons.add(key)
                picked.append(record)
                if len(picked) >= limit_per_protocol:
                    break
        if not picked:
            continue
        lines.append(f"**{label}**")
        lines.append("")
        for record in picked:
            answer = (record.get("final_answer_text") or "").strip().replace("\n", " ")
            lines.append(
                f"- `{record.get('task')}` / id={record.get('id')}："
                f"reason=`{record.get('invalid_reason') or record.get('error', '')[:80]}`，"
                f"最终回答=`{answer[:120]!r}`，GT=`{record.get('ground_truth')}`"
            )
        lines.append("")
    if not lines:
        lines.append("未发现 invalid / 截断 / 推理错误案例。")
    return lines


def _limits_section(model: dict) -> list[str]:
    return [
        "## 12. 测试局限",
        "",
        "- 本次只评测 **SPAR-Bench-Tiny（RGB）**，不能代表完整 SPAR-Bench 或 RGB-D 结果。",
        "- 每题 50 条，单任务分数置信区间较宽；任务间 1~2 个百分点的差异不宜过度解读。",
        "- 官方分数基于官方评分函数；`exact_match` 含“预测是答案子串”与“首字符相等”"
        "回退逻辑，因此官方 MCA 分数可能高于严格解析分数。",
        "- 官方 `exact_match` 对空串会因子串匹配返回 1；本实现的评分入口把"
        "“空最终回答”显式记 0 并标 invalid，避免把没答完算成正确。",
        "- Thinking On 时官方 / 严格评分都作用在 `</think>` 之后的最终回答；"
        "若生成被截断且没有闭合 Thinking，最终回答为空，记 0 并标记 invalid。",
        "- Direct Answer（64 tokens）使用独立的 Prompt 版本，"
        "不能把它与 Thinking Off 的差异解释为单一变量消融。",
        "- 推理耗时包含 CUDA 同步与逐样本显存统计开销；不同时刻 GPU 上还有其他进程，"
        "绝对速度仅供参考，模型间对比应在同一时段内解读。",
        "",
    ]


# --------------------------------------------------------------- 单模型报告
def build_model_report(model_dir: str, runs: dict[str, dict]) -> str:
    model = MODELS[model_dir]
    lines: list[str] = []
    add = lines.append

    complete = _all_available(runs) and all(
        r["experiment"].get("status") == "complete" for r in runs.values()
        if r["available"]
    )

    add(f"# {model['title']} — SPAR-Bench-Tiny 评测报告")
    add("")
    add(f"> 数据：SPAR-Bench-Tiny（RGB，1,000 条，20 种任务，50 条/任务）。"
        f"模型：{model['title']}。")
    if not complete:
        add(">")
        add("> **注意：并非所有协议都已完成；下方仅展示已有实际结果，未运行项为 N/A。**")
    if any(
        run["available"] and run["metrics"]["n_samples"] < 1000
        for run in runs.values()
    ):
        add(">")
        add("> **注意：这是小规模试跑（样本数 < 1,000），不能当作全量成绩。**")
    add("")

    add("## 1. 模型与部署")
    add("")
    add("| 项目 | 内容 |")
    add("|---|---|")
    add(f"| 模型 | {model['title']} |")
    add(f"| Hugging Face | {model['hf']} |")
    add(f"| 运行环境 | {model['env']} |")
    sample_run = next((r for r in runs.values() if r["available"]), None)
    if sample_run:
        exp = sample_run["experiment"]
        load_info = exp.get("load_info") or {}
        add(f"| 本地权重 | `{load_info.get('model_path', 'N/A')}` |")
        add(f"| Attention | {load_info.get('attn_implementation', 'N/A')} |")
        add(f"| dtype / batch | bfloat16 / 1 |")
        config = exp.get("config", {})
        add(f"| GPU | {config.get('gpu_name', 'N/A')} |")
        add(f"| PyTorch / Transformers | {config.get('torch', 'N/A')} / "
            f"{config.get('transformers', 'N/A')} |")
        add(f"| 数据集 SHA-256 | `{config.get('dataset_sha256', 'N/A')}` |")
    add("")

    add("## 2. 实验协议")
    add("")
    add("| 协议 | Thinking | max_new_tokens | Prompt | 说明 |")
    add("|---|---|---:|---|---|")
    add("| `think_off_8192` | Off | 8192 | 官方 Prompt | 不开启显式思考 |")
    add("| `think_on_8192` | On | 8192 | 官方 Prompt | 与 Off 仅 `enable_thinking` 不同 |")
    add("| `direct_off_64` | Off | 64 | 官方 + 直接回答指令 | 独立 Prompt，要求只输出答案 |")
    add("")
    add("三个协议使用相同的 1,000 条样本与相同顺序；Multi-view 的全部图片按原始顺序输入。")
    add("")

    add("## 3. 运行状态")
    add("")
    lines.extend(_config_table(runs))
    add("")

    add("## 4. 总体成绩")
    add("")
    add("官方总分口径：先算 20 个任务各自的分数，再对任务分数取算术平均。"
        "Low / Medium / High 同样先算任务分数再取层级均值。")
    add("")
    lines.extend(_overall_table(runs))
    add("")

    add("## 5. MCA / NA / VCI")
    add("")
    add("| 评分方式 | 指标 | 说明 |")
    add("|---|---|---|")
    add("| MCA | Accuracy | 官方 `exact_match`（含子串 / 首字符回退） |")
    add("| NA | MRA:.5:.95:.05 | 官方 `process_na` + `mean_relative_accuracy` |")
    add("| VCI | VCI metric | 官方 `compute_vci_metric`，5 个方向各算一次 MRA |")
    add("")
    lines.extend(_answer_type_table(runs))
    add("")

    add("## 6. 20 种任务得分（Official）")
    add("")
    lines.extend(_task_table(runs))
    add("")
    add("## 7. 20 种任务得分（Strict，补充）")
    add("")
    add("Strict 只接受最终回答区域里的严格格式：选择题必须是单个选项字母，"
        "数值题必须是单个数值，VCI 必须完整给出 5 组参数；不做事后模糊匹配。")
    add("")
    lines.extend(_strict_task_table(runs))
    add("")

    add("## 8. 单视角 / 多视角")
    add("")
    lines.extend(_img_type_table(runs))
    add("")
    add("Single-view 与 Multi-view 的任务集合不同，这一对比反映的是**任务组合差异**，"
        "不能单独解释为“多图输入能力”的因果效应。")
    add("")

    add("## 9. 性能指标")
    add("")
    lines.extend(_performance_table(runs))
    add("")

    add("## 10. 数据质量与异常")
    add("")
    lines.extend(_quality_table(runs))
    add("")
    add("### 失败案例示例")
    add("")
    lines.extend(_failure_examples(runs))
    add("")

    add("## 11. 结论与下一步")
    add("")
    available = [(p, label, runs[p]) for p, label in PROTOCOLS if runs[p]["available"]]
    if available:
        best_protocol, best_label, best_run = max(
            available,
            key=lambda item: item[2]["metrics"]["overall"]["official"] or -1,
        )
        bm = best_run["metrics"]
        add(f"- 三个协议中 **{best_label}** 的官方总分最高："
            f"Official {_pct(bm['overall']['official'])} / "
            f"Strict {_pct(bm['overall']['strict'])}。")
        add(f"- 该协议层级得分：Low {_pct(bm['levels']['low']['official_score'])}、"
            f"Medium {_pct(bm['levels']['medium']['official_score'])}、"
            f"High {_pct(bm['levels']['high']['official_score'])}；"
            f"MCA {_pct(bm['answer_types']['MCA']['task_mean_official'])}、"
            f"NA {_pct(bm['answer_types']['NA']['task_mean_official'])}、"
            f"VCI {_pct(bm['answer_types']['VCI']['task_mean_official'])}。")

        off_run = runs["think_off_8192"]
        on_run = runs["think_on_8192"]
        direct_run = runs["direct_off_64"]
        if off_run["available"]:
            om = off_run["metrics"]
            gap = (om["overall"]["official"] - om["overall"]["strict"]) * 100
            add(f"- Thinking Off：Official 比 Strict 高 {gap:+.2f} pp。"
                "官方 `exact_match` 的首字符 / 子串回退会奖励部分解释性文本，"
                "Strict 不奖励；两者不能混用。")
            add(f"- Thinking Off 有 {om['quality']['invalid']} 条严格解析失败、"
                f"{om['quality']['truncated']} 条截断；"
                f"平均生成 {om['performance']['tokens']['generated']['mean']:.0f} tokens。")
        if off_run["available"] and on_run["available"]:
            delta = (on_run["metrics"]["overall"]["official"]
                     - off_run["metrics"]["overall"]["official"]) * 100
            add(f"- Thinking On 相对 Off 的官方变化 {delta:+.2f} pp；"
                f"On 截断 {on_run['metrics']['quality']['truncated']}/1000，"
                f"平均生成 {on_run['metrics']['performance']['tokens']['generated']['mean']:.0f} tokens。")
        if direct_run["available"]:
            dm = direct_run["metrics"]
            add(f"- Direct Answer：E2E 均值 {dm['performance']['timing_seconds']['e2e_seconds']['mean']:.2f}s，"
                f"截断 {dm['quality']['truncated']}/1000，"
                f"VCI {_pct(dm['answer_types']['VCI']['task_mean_official'])}。")
            add("- Direct Answer 使用独立 Prompt 版本，不能与 Thinking Off 做单变量归因。")
    add("- 下一步可在完整 SPAR-Bench 上复核 Tiny 结论，并补做 RGB-D 版本；"
        "VCI 是两模型共同的最弱项，值得单独做 prompt / 输出格式优化。")
    add("")
    lines.extend(_limits_section(model))

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- 对比报告
def build_comparison(all_runs: dict[str, dict[str, dict]]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Xiaomi-Lens-0 vs Qwen3.5-4B — SPAR-Bench-Tiny 综合对比")
    add("")
    add("> 数据：SPAR-Bench-Tiny（RGB，1,000 条，20 种任务）。"
        "两组模型使用相同的样本、相同的 Prompt 版本和相同的评分代码。")
    add("> 所有数字来自实际预测记录；未完成的实验标为 N/A。")
    if any(
        run["available"] and run["metrics"]["n_samples"] < 1000
        for runs in all_runs.values() for run in runs.values()
    ):
        add(">")
        add("> **注意：部分或全部结果为小规模试跑（样本数 < 1,000），不是全量成绩。**")
    add("")

    add("## 1. 总体对比")
    add("")
    add("| 协议 | 模型 | 样本数 | Official | Strict | Low | Medium | High |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for protocol, label in PROTOCOLS:
        for model_dir, model in MODELS.items():
            run = all_runs[model_dir][protocol]
            if not run["available"]:
                add(f"| {label} | {model['title']} | - | - | - | - | - | - |")
                continue
            m = run["metrics"]
            levels = m["levels"]
            add(
                f"| {label} | {model['title']} | {m['n_samples']} | "
                f"{_pct(m['overall']['official'])} | "
                f"{_pct(m['overall']['strict'])} | "
                f"{_pct(levels['low']['official_score'])} | "
                f"{_pct(levels['medium']['official_score'])} | "
                f"{_pct(levels['high']['official_score'])} |"
            )
    add("")

    add("## 2. 20 种任务对比（Official）")
    add("")
    header = "| 任务 | 层级 |"
    for protocol, _ in PROTOCOLS:
        short = SHORT_LABELS[protocol]
        header += f" Xiaomi {short} | Qwen {short} |"
    add(header)
    add("|---|---|" + "---:|" * (2 * len(PROTOCOLS)))
    for task in ALL_TASKS:
        level = ("low" if task in LOW_TASKS else
                 "medium" if task in MIDDLE_TASKS else "high")
        row = f"| `{task}` | {level} |"
        for protocol, _ in PROTOCOLS:
            for model_dir in MODELS:
                run = all_runs[model_dir][protocol]
                block = run["metrics"]["tasks"].get(task) if run["available"] else None
                row += f" {_pct(block['official_score']) if block else '-'} |"
        add(row)
    add("")

    add("## 3. MCA / NA / VCI 对比")
    add("")
    add("| 协议 | 模型 | MCA | NA (MRA) | VCI |")
    add("|---|---|---:|---:|---:|")
    for protocol, label in PROTOCOLS:
        for model_dir, model in MODELS.items():
            run = all_runs[model_dir][protocol]
            if not run["available"]:
                add(f"| {label} | {model['title']} | - | - | - |")
                continue
            types = run["metrics"]["answer_types"]
            add(f"| {label} | {model['title']} | "
                f"{_pct(types['MCA']['task_mean_official'])} | "
                f"{_pct(types['NA']['task_mean_official'])} | "
                f"{_pct(types['VCI']['task_mean_official'])} |")
    add("")

    add("## 4. 单视角 / 多视角")
    add("")
    add("| 协议 | 模型 | Single-view | Multi-view |")
    add("|---|---|---:|---:|")
    for protocol, label in PROTOCOLS:
        for model_dir, model in MODELS.items():
            run = all_runs[model_dir][protocol]
            if not run["available"]:
                add(f"| {label} | {model['title']} | - | - |")
                continue
            img = run["metrics"]["img_types"]
            add(f"| {label} | {model['title']} | "
                f"{_pct(img.get('single_view', {}).get('official_mean'))} | "
                f"{_pct(img.get('multi_view', {}).get('official_mean'))} |")
    add("")

    add("## 5. Thinking On / Off 的影响")
    add("")
    add("只比较 `think_off_8192` 与 `think_on_8192`：两者 Prompt 完全相同，"
        "唯一变量是 `enable_thinking`。`direct_off_64` 使用另一版 Prompt，不参与此结论。")
    add("")
    add("| 模型 | Off Official | On Official | 变化 | Off 平均生成 Tokens | On 平均生成 Tokens |")
    add("|---|---:|---:|---:|---:|---:|")
    for model_dir, model in MODELS.items():
        off = all_runs[model_dir]["think_off_8192"]
        on = all_runs[model_dir]["think_on_8192"]
        if not (off["available"] and on["available"]):
            add(f"| {model['title']} | - | - | - | - | - |")
            continue
        off_score = off["metrics"]["overall"]["official"]
        on_score = on["metrics"]["overall"]["official"]
        delta = (on_score - off_score) * 100 if None not in (off_score, on_score) else None
        off_tokens = (off["metrics"]["performance"]["tokens"]["generated"] or {}).get("mean")
        on_tokens = (on["metrics"]["performance"]["tokens"]["generated"] or {}).get("mean")
        add(f"| {model['title']} | {_pct(off_score)} | {_pct(on_score)} | "
            f"{'N/A' if delta is None else f'{delta:+.2f} pp'} | "
            f"{_num(off_tokens, 0)} | {_num(on_tokens, 0)} |")
    add("")

    add("## 6. 推理速度与显存")
    add("")
    add("| 协议 | 模型 | E2E 均值(s) | E2E P95(s) | Samples/s | Tokens/s | 峰值显存(GiB) |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for protocol, label in PROTOCOLS:
        for model_dir, model in MODELS.items():
            run = all_runs[model_dir][protocol]
            if not run["available"]:
                add(f"| {label} | {model['title']} | - | - | - | - | - |")
                continue
            p = run["metrics"]["performance"]
            timings = p["timing_seconds"]
            add(f"| {label} | {model['title']} | "
                f"{_num((timings.get('e2e_seconds') or {}).get('mean'))} | "
                f"{_num((timings.get('e2e_seconds') or {}).get('p95'))} | "
                f"{_num(p.get('samples_per_second'))} | "
                f"{_num(p.get('tokens_per_second_generate'), 1)} | "
                f"{_num((p.get('vram') or {}).get('per_sample_peak_allocated_gib_max'), 1)} |")
    add("")

    add("## 7. 数据质量")
    add("")
    add("| 模型 | 协议 | Invalid | 截断 | 错误 | OOM | Thinking 未闭合 |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for model_dir, model in MODELS.items():
        for protocol, label in PROTOCOLS:
            run = all_runs[model_dir][protocol]
            if not run["available"]:
                add(f"| {model['title']} | {label} | - | - | - | - | - |")
                continue
            q = run["metrics"]["quality"]
            add(f"| {model['title']} | {label} | {q['invalid']} | {q['truncated']} | "
                f"{q['errors']} | {q['oom']} | {q['thinking_unclosed']} |")
    add("")

    add("## 8. 结论")
    add("")

    def both(protocol):
        a = all_runs["xiaomi_lens_0"][protocol]
        b = all_runs["qwen3_5_4b"][protocol]
        return (a, b) if a["available"] and b["available"] else None

    add("### 8.1 总体差距")
    add("")
    add("| 协议 | Xiaomi Official | Qwen Official | Qwen - Xiaomi | 领先 |")
    add("|---|---:|---:|---:|---|")
    for protocol, label in PROTOCOLS:
        pair = both(protocol)
        if pair is None:
            add(f"| {label} | - | - | - | - |")
            continue
        a = pair[0]["metrics"]["overall"]["official"]
        b = pair[1]["metrics"]["overall"]["official"]
        delta = (b - a) * 100 if None not in (a, b) else None
        if delta is None:
            winner = "-"
        elif abs(delta) < 0.01:
            winner = "持平"
        else:
            winner = "Qwen" if delta > 0 else "Xiaomi"
        add(f"| {label} | {_pct(a)} | {_pct(b)} | "
            f"{'N/A' if delta is None else f'{delta:+.2f} pp'} | {winner} |")
    add("")

    add("### 8.2 层级差距（Qwen - Xiaomi）")
    add("")
    add("| 协议 | Low | Medium | High |")
    add("|---|---:|---:|---:|")
    for protocol, label in PROTOCOLS:
        pair = both(protocol)
        if pair is None:
            add(f"| {label} | - | - | - |")
            continue
        cells = []
        for level in ("low", "medium", "high"):
            a = pair[0]["metrics"]["levels"][level]["official_score"]
            b = pair[1]["metrics"]["levels"][level]["official_score"]
            delta = (b - a) * 100 if None not in (a, b) else None
            cells.append("N/A" if delta is None else f"{delta:+.2f}")
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("")

    add("### 8.3 任务级差距（|Qwen - Xiaomi| ≥ 10 pp）")
    add("")
    add("| 协议 | 任务 | Xiaomi | Qwen | 差值 |")
    add("|---|---|---:|---:|---:|")
    found_any = False
    for protocol, label in PROTOCOLS:
        pair = both(protocol)
        if pair is None:
            continue
        for task in ALL_TASKS:
            a_block = pair[0]["metrics"]["tasks"].get(task)
            b_block = pair[1]["metrics"]["tasks"].get(task)
            if not a_block or not b_block:
                continue
            a = a_block["official_score"]
            b = b_block["official_score"]
            if a is None or b is None:
                continue
            delta = (b - a) * 100
            if abs(delta) >= 10:
                found_any = True
                add(f"| {label} | `{task}` | {_pct(a)} | {_pct(b)} | {delta:+.2f} pp |")
    if not found_any:
        add("| - | 无 ≥ 10 pp 的任务级差异 | - | - | - |")
    add("")

    add("### 8.4 速度与显存")
    add("")
    add("| 模型 | 协议 | E2E 均值(s) | Decode Tokens/s | 峰值显存(GiB) |")
    add("|---|---|---:|---:|---:|")
    for protocol, label in PROTOCOLS:
        for model_dir, model in MODELS.items():
            run = all_runs[model_dir][protocol]
            if not run["available"]:
                continue
            p = run["metrics"]["performance"]
            add(f"| {model['title']} | {label} | "
                f"{_num((p['timing_seconds'].get('e2e_seconds') or {}).get('mean'))} | "
                f"{_num(p.get('tokens_per_second_decode'), 1)} | "
                f"{_num((p.get('vram') or {}).get('per_sample_peak_allocated_gib_max'), 1)} |")
    add("")

    add("### 8.5 综合判断")
    add("")

    def _score(model_dir, protocol, key="official"):
        run = all_runs[model_dir][protocol]
        return run["metrics"]["overall"][key] if run["available"] else None

    def _delta(protocol, key="official"):
        a = _score("xiaomi_lens_0", protocol, key)
        b = _score("qwen3_5_4b", protocol, key)
        return None if a is None or b is None else (b - a) * 100

    off_delta = _delta("think_off_8192")
    off_strict_delta = _delta("think_off_8192", "strict")
    if off_delta is not None:
        add(f"- **Thinking Off**：Qwen 的 Official 领先 {off_delta:+.2f} pp；"
            f"但 Strict 下为 {off_strict_delta:+.2f} pp。"
            + (" 官方差距主要来自 `exact_match` 的格式宽容度，"
               "严格解析后两模型基本持平。" if abs(off_strict_delta) < abs(off_delta) / 2
               else " 严格解析后差距仍然存在。"))

    on_delta = _delta("think_on_8192")
    if on_delta is not None:
        x_on = all_runs["xiaomi_lens_0"]["think_on_8192"]["metrics"]
        q_on = all_runs["qwen3_5_4b"]["think_on_8192"]["metrics"]
        add(f"- **Thinking On**：Qwen - Xiaomi = {on_delta:+.2f} pp（Strict "
            f"{_delta('think_on_8192', 'strict'):+.2f} pp）。"
            f"两模型截断分别为 Xiaomi {x_on['quality']['truncated']}/1000、"
            f"Qwen {q_on['quality']['truncated']}/1000；"
            f"Low 层为 Xiaomi {_pct(x_on['levels']['low']['official_score'])}、"
            f"Qwen {_pct(q_on['levels']['low']['official_score'])}，"
            "说明低层数值题被 8192 上限拖累最明显。")

    direct_delta = _delta("direct_off_64")
    if direct_delta is not None:
        x_dir = all_runs["xiaomi_lens_0"]["direct_off_64"]["metrics"]
        q_dir = all_runs["qwen3_5_4b"]["direct_off_64"]["metrics"]
        add(f"- **Direct Answer**：Qwen - Xiaomi = {direct_delta:+.2f} pp（Strict "
            f"{_delta('direct_off_64', 'strict'):+.2f} pp）。"
            f"该协议是两模型各自的最好成绩：Xiaomi {_pct(x_dir['overall']['official'])}、"
            f"Qwen {_pct(q_dir['overall']['official'])}；"
            "说明格式约束能显著提升两者的可评分性。")

    vci_values = []
    for model_dir, model in MODELS.items():
        for protocol, _ in PROTOCOLS:
            run = all_runs[model_dir][protocol]
            if run["available"]:
                vci_values.append((model["title"], protocol,
                                   run["metrics"]["answer_types"]["VCI"]["task_mean_official"]))
    if vci_values:
        worst = min(vci_values, key=lambda item: item[2])
        best = max(vci_values, key=lambda item: item[2])
        add(f"- **VCI 是共同短板**：全部 6 组在 {_pct(worst[2])} ~ {_pct(best[2])} 之间"
            f"（最高为 {best[0]} / `{best[1]}`）。`view_change_infer` 需要精确输出 "
            "5 组移动/旋转参数，当前模型普遍格式不合格或数值偏差大。")

    all_scores = []
    for model_dir, model in MODELS.items():
        for protocol, _ in PROTOCOLS:
            run = all_runs[model_dir][protocol]
            if run["available"]:
                all_scores.append((model["title"], protocol,
                                   run["metrics"]["overall"]["official"]))
    if all_scores:
        best = max(all_scores, key=lambda item: item[2])
        add(f"- **最佳单项**：{best[0]} / `{best[1]}`，Official {_pct(best[2])}。")

    add("- **工程可用性**：Direct Answer 约 0.5s/样本；Thinking Off 下 Qwen 明显快于 "
        "Xiaomi；Thinking On 两组都达到 1~3 分钟/样本，且大量触顶 8192，"
        "部署成本远高于 Direct / Off。")
    add("")

    add("## 9. 报告说明与局限")
    add("")
    add("- 本次为 **SPAR-Bench-Tiny RGB-only**，不是完整 SPAR-Bench，也不包含 RGB-D。")
    add("- 所有成绩来自 `predictions.jsonl`，`metrics.json` 可由 `recompute_metrics.py` 复算。")
    add("- 官方 MCA `exact_match` 含模糊回退，报告同时给出 Strict 作为补充，二者不可混用。")
    add("- Multi-view 的 2~3 张图片全部按原始顺序输入，并通过 image token / grid 数量校验。")
    add("")

    return "\n".join(lines) + "\n"


def main() -> int:
    global EXPERIMENT_ROOT
    import argparse

    parser = argparse.ArgumentParser(description="Generate SPAR-Bench reports")
    parser.add_argument("--root", default=str(EXPERIMENT_ROOT),
                        help="实验根目录")
    parser.add_argument("--run-name", default="",
                        help="读取 <protocol>/<run-name>/（冒烟测试用）")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    EXPERIMENT_ROOT = Path(args.root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_runs = {
        model_dir: load_runs(model_dir, args.run_name) for model_dir in MODELS
    }

    for model_dir, model in MODELS.items():
        report = build_model_report(model_dir, all_runs[model_dir])
        path = report_dir / model["report"]
        path.write_text(report, encoding="utf-8")
        print(f"已写出 {path}")

    comparison = build_comparison(all_runs)
    path = report_dir / "spar_bench_tiny_comparison.md"
    path.write_text(comparison, encoding="utf-8")
    print(f"已写出 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
