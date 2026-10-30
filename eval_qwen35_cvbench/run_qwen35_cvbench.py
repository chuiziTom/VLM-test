"""在 CV-Bench 上评测 Qwen3.5-4B（Thinking Off / On）。

本脚本是 Xiaomi 评测 runner 的"换模型"版本：数据集、严格答案解析、指标计算、
断点续跑与结果文件结构全部沿用同一套实现，只有模型适配器换成了 Qwen3.5-4B。
因此两个模型的准确率是在同一协议下得到的，可以直接横向比较。

示例
----
单样本，Thinking Off::

    micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py \
        --mode off --limit 1 --verbose --run-name single

分层小样本（覆盖 6 个 type/task/source）::

    micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py \
        --mode all --limit 16 --stratified --run-name smoke16

完整测试集，两种模式依次跑完（模型只加载一次，可断点续跑）::

    micromamba run -n vlm-test python eval_qwen35_cvbench/run_qwen35_cvbench.py \
        --mode all --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# 必须在 CUDA 初始化前设置，降低显存碎片。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from answer_parser import parse_final_answer  # noqa: E402
from cvbench_dataset import (  # noqa: E402
    CVBenchImages, describe, load_records,
)
from metrics import compare_runs, compute_metrics  # noqa: E402
from model_qwen35 import (  # noqa: E402
    MODEL_NAME, Qwen35Adapter, mode_to_enable_thinking,
)

PARSER_VERSION = "1.0"          # 与 Xiaomi 目录中的 answer_parser.py 相同
SUPPORTED_MODES = ("off", "on")
DEFAULT_DATASET_DIR = "/home/20261024_1/Workspace/qyh/VLM-test/dataset/CV-Bench"
DEFAULT_OUTPUT_DIR = str(HERE / "results")
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

# 与 Xiaomi 评测完全相同的格式指令（逐字符一致，保证可比）。
FORMAT_INSTRUCTION = (
    "\n\nAfter answering the question, put the final choice on a separate line "
    "using exactly this format: FINAL_ANSWER: (X)\n"
    "where X is one of the provided option letters."
)


# --------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CV-Bench evaluation for Qwen3.5-4B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("off", "on", "adaptive", "all"),
                        required=True,
                        help="Thinking 模式。'adaptive' 只做探测，不会被模拟。")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评测按 idx 排序后的前 N 条。")
    parser.add_argument("--stratified", action="store_true",
                        help="与 --limit 配合，按 type/task/source 轮转抽样。")
    parser.add_argument("--split", choices=("2d", "3d", "all"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="生成上限；与 Xiaomi 一致，便于横向比较。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="",
                        help="子目录名，用来隔离不同配置的运行。")
    parser.add_argument("--resume", action="store_true",
                        help="在同一输出目录上续跑。")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn", default="sdpa",
                        choices=("flash_attention_2", "sdpa", "eager"),
                        help="本环境未安装 flash-attn，默认 sdpa。")
    parser.add_argument("--warmup", type=int, default=1,
                        help="预热次数（不计入成绩）。")
    parser.add_argument("--min-free-gib", type=float, default=2.0,
                        help="每个样本前，若 torch.cuda.mem_get_info 报告的空闲显存"
                             "低于该值则等待。")
    parser.add_argument("--memory-wait", type=int, default=120,
                        help="每个样本前等待空闲显存的最长秒数。")
    parser.add_argument("--require-free-gib", type=float, default=12.0,
                        help="加载模型前的硬性门槛：整卡空闲显存不足则直接退出，"
                             "而不是冒险 OOM 干扰其他任务。")
    parser.add_argument("--oom-retries", type=int, default=3)
    parser.add_argument("--oom-wait", type=int, default=30)
    parser.add_argument("--verbose", action="store_true",
                        help="打印每条样本；默认只打印前 5 条。")
    parser.add_argument("--validate", action="store_true",
                        help="只校验已有结果，不加载模型。")
    return parser.parse_args()


# ------------------------------------------------------------------ 工具
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _module_version(name: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return None


def _gpu_processes() -> list[dict]:
    """列出当前 GPU 上的计算进程，用来判断是否与他人共享 GPU。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append({"pid": parts[0], "process_name": parts[1],
                         "used_memory": parts[2]})
    return rows


def _free_gib(device) -> float:
    free, _ = torch.cuda.mem_get_info(device)
    return free / 1024 ** 3


def _make_experiment(mode: str, args: argparse.Namespace, records) -> dict:
    dataset_files = {
        name: _file_fingerprint(Path(args.dataset_dir) / name)
        for name in ("test_2d.parquet", "test_3d.parquet")
        if (Path(args.dataset_dir) / name).exists()
    }
    config = {
        "mode": mode,
        "enable_thinking": mode_to_enable_thinking(mode),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "split": args.split,
        "limit": args.limit,
        "stratified": args.stratified,
        "seed": args.seed,
        "model_path": args.model_path,
        "dataset_dir": args.dataset_dir,
        "dataset_files": dataset_files,
        "prompt_template": FORMAT_INSTRUCTION,
        "prompt_template_sha1": _sha1(FORMAT_INSTRUCTION),
        "parser_version": PARSER_VERSION,
        "selected_uids_sha1": _sha1("|".join(r.uid for r in records)),
        "selected_samples": len(records),
        "torch": torch.__version__,
        "transformers": _module_version("transformers"),
        "python": platform.python_version(),
    }
    config["fingerprint"] = _sha1(
        json.dumps(config, sort_keys=True, ensure_ascii=False)
    )
    return config


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL；崩溃留下的半行会自动修掉。"""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records, last_good = [], 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
            last_good = i + 1
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"JSONL 在第 {i + 1} 行损坏: {path}")
            break
    if last_good < len(lines):
        path.write_text(
            "\n".join(lines[:last_good]) + ("\n" if last_good else ""),
            encoding="utf-8",
        )
    return records


def _select_records(records, limit: int | None, stratified: bool):
    if limit is None:
        return records
    if not stratified:
        return records[:limit]
    buckets: dict[tuple, list] = defaultdict(list)
    for record in records:
        buckets[(record.type, record.task, record.source)].append(record)
    picks = []
    keys = sorted(buckets)
    while len(picks) < limit:
        added = False
        for key in keys:
            if buckets[key] and len(picks) < limit:
                picks.append(buckets[key].pop(0))
                added = True
        if not added:
            break
    return sorted(picks, key=lambda r: r.idx)


def _write_summary_csv(path: Path, metrics: dict) -> None:
    """与 Xiaomi 相同的 5 列结构 group,n,correct,accuracy_percent,invalid。"""
    rows = [("overall", metrics["total_samples"], metrics["correct"],
             metrics["accuracy_percent"], metrics["invalid"])]
    official = metrics["official"]
    for name, value in (
        ("official_cvbench", official["cvbench_accuracy_percent"]),
        ("official_2d", 100 * official["accuracy_2d"]
         if official["accuracy_2d"] is not None else None),
        ("official_3d", 100 * official["accuracy_3d"]
         if official["accuracy_3d"] is not None else None),
    ):
        rows.append((name, metrics["total_samples"], None, value, None))
    for group_name, groups in metrics["groups"].items():
        for key, block in groups.items():
            rows.append((f"{group_name}/{key}", block["n"], block["correct"],
                         block["accuracy_percent"], block["invalid"]))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("group,n,correct,accuracy_percent,invalid\n")
        for row in rows:
            handle.write(",".join("" if v is None else str(v) for v in row) + "\n")


def _print_sample(record: dict, index: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"[{index}/{total}] {record['uid']}  type={record['type']} "
          f"task={record['task']} source={record['source']}")
    print(f"Question: {record['question']}")
    print(f"Choices : {record['choices']}")
    print(f"GT      : {record['ground_truth']}")
    print(f"Mode    : {record['thinking_mode']} "
          f"(enable_thinking={record['enable_thinking']}, "
          f"prompt_verified={record['mode_verified']})")
    print(f"Raw     : {record['raw_generation']!r}")
    print(f"Final   : {record['final_answer_text']!r}")
    print(f"Parsed  : {record['prediction']}  "
          f"{'Correct' if record['correct'] else 'Incorrect'}"
          f"  invalid={record['invalid']} ({record['invalid_reason']})")
    print(f"Tokens  : total={record['generated_tokens']} "
          f"thinking={record['thinking_tokens']} answer={record['answer_tokens']}")
    print(f"Time    : preprocess={record['preprocess_seconds']:.3f}s "
          f"generate={record['generate_seconds']:.3f}s "
          f"e2e={record['e2e_seconds']:.3f}s")
    print(f"VRAM    : peak_allocated={record['vram'].get('peak_allocated_gib', 0):.2f} GiB "
          f"reserved={record['vram'].get('reserved_gib', 0):.2f} GiB")
    print("=" * 72)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or \
        "out of memory" in str(exc).lower()


def _wait_for_memory(device, min_free_gib: float, timeout_s: float) -> bool:
    if min_free_gib <= 0:
        return True
    deadline = time.perf_counter() + timeout_s
    while True:
        if _free_gib(device) >= min_free_gib:
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(10)


# ------------------------------------------------------------------- 运行
def run_mode(
    adapter: Qwen35Adapter,
    records,
    image_store: CVBenchImages,
    mode: str,
    args: argparse.Namespace,
    out_root: Path,
) -> dict:
    mode_dir = out_root / (args.run_name or "") / f"thinking_{mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = mode_dir / "predictions.jsonl"
    experiment_path = mode_dir / "experiment.json"

    experiment = _make_experiment(mode, args, records)

    # 先校验配置指纹，再决定是否读写已有预测，避免不同配置的结果混在一起。
    if experiment_path.exists():
        previous = json.loads(experiment_path.read_text(encoding="utf-8"))
        if previous.get("config", {}).get("fingerprint") != experiment["fingerprint"]:
            raise SystemExit(
                f"{mode_dir} 中的配置与本次参数不一致。请换一个 --run-name，"
                "或删除该目录；不同配置的结果不能混用。"
            )
        experiment = previous["config"]
    else:
        experiment_path.write_text(
            json.dumps({"config": experiment}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    existing = _read_jsonl(predictions_path)
    if predictions_path.exists() and existing and not args.resume:
        raise SystemExit(
            f"{predictions_path} 已有 {len(existing)} 条记录。"
            "请加 --resume 续跑，或换一个 --run-name。"
        )

    # OOM 表示这条样本其实没被评测过：删掉记录，让 --resume 重新评测，
    # 而不是把它当作"答错"计入成绩。
    if args.resume:
        def retryable(record):
            error = (record.get("error") or "").lower()
            return bool(record.get("retryable")) or "out of memory" in error

        retry = [r for r in existing if retryable(r)]
        if retry:
            existing = [r for r in existing if not retryable(r)]
            with predictions_path.open("w", encoding="utf-8") as handle:
                for item in existing:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[{mode}] 丢弃 {len(retry)} 条 OOM 记录，稍后重测")

    done = {r["uid"] for r in existing}
    if len(done) != len(existing):
        raise ValueError(f"{predictions_path} 中存在重复 UID")

    pending = [r for r in records if r.uid not in done]
    print(f"\n[{mode}] 已完成 {len(existing)}，待评测 {len(pending)} -> {mode_dir}")

    # 预热：只为把 CUDA kernel 和显存池预热，结果不计入成绩。
    warmup_seconds = 0.0
    if args.warmup and pending:
        warmup_record = records[0]
        image = image_store.get(warmup_record)
        for _ in range(args.warmup):
            prepared = adapter.build_inputs(
                image, warmup_record.prompt + FORMAT_INSTRUCTION, mode
            )
            if not prepared["mode_verified"]:
                raise RuntimeError(
                    f"chat template 没有按 Thinking {mode} 渲染，拒绝继续运行。"
                )
            start = time.perf_counter()
            adapter.generate(prepared, mode, args.max_new_tokens, reset_peak=False)
            warmup_seconds += time.perf_counter() - start
        torch.cuda.empty_cache()

    global_peak_allocated = 0.0
    global_peak_reserved = 0.0
    started = time.perf_counter()
    aborted = False

    with predictions_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(tqdm(pending, desc=f"thinking_{mode}"), 1):
            sample_start = time.perf_counter()
            error = None
            error_is_oom = False
            image_load_seconds = 0.0
            prepared = {"preprocess_seconds": 0.0, "prompt_tokens": None,
                        "mode_verified": False, "enable_thinking": None}
            generation = {
                "raw_generation": "", "generated_tokens": 0,
                "thinking_tokens": 0, "answer_tokens": 0,
                "finish_reason": "error", "truncated": False,
                "generate_seconds": 0.0, "tokens_per_second": None,
                "vram": {},
            }
            parse = None
            attempt = 0

            while True:
                if not _wait_for_memory(
                    adapter.device, args.min_free_gib, args.memory_wait
                ):
                    print(f"[{mode}] 警告：{record.uid} 之前 GPU 空闲显存仍偏低，仍尝试运行")
                try:
                    image_start = time.perf_counter()
                    image = image_store.get(record)
                    image_load_seconds = time.perf_counter() - image_start

                    instruction = record.prompt + FORMAT_INSTRUCTION
                    prepared = adapter.build_inputs(image, instruction, mode)
                    if not prepared["mode_verified"]:
                        raise RuntimeError("Thinking 模式没有被模板正确应用")

                    generation = adapter.generate(
                        prepared, mode, args.max_new_tokens, reset_peak=True
                    )
                    parse = parse_final_answer(
                        generation["raw_generation"],
                        record.valid_letters,
                        requires_think_close=mode == "on",
                        truncated=generation["truncated"],
                    )
                    error = None
                    break
                except Exception as exc:
                    oom = _is_oom(exc)
                    if oom and attempt < args.oom_retries:
                        attempt += 1
                        torch.cuda.empty_cache()
                        wait = args.oom_wait * attempt
                        print(f"[{mode}] {record.uid} CUDA OOM；第 "
                              f"{attempt}/{args.oom_retries} 次重试，等待 {wait}s")
                        time.sleep(wait)
                        continue
                    if oom:
                        # 不写入这条样本，保证 --resume 会重新评测它。
                        print(f"[{mode}] {record.uid} 持续 OOM，安全停止。"
                              "稍后用 --resume 继续。")
                        aborted = True
                        break

                    traceback.print_exc()
                    error = f"{type(exc).__name__}: {exc}"
                    error_is_oom = False
                    prepared = {"preprocess_seconds": 0.0, "prompt_tokens": None,
                                "mode_verified": False, "enable_thinking": None}
                    generation = {
                        "raw_generation": "", "generated_tokens": 0,
                        "thinking_tokens": 0, "answer_tokens": 0,
                        "finish_reason": "error", "truncated": False,
                        "generate_seconds": 0.0, "tokens_per_second": None,
                        "vram": {},
                    }
                    parse = parse_final_answer(
                        "", record.valid_letters,
                        requires_think_close=mode == "on", truncated=False,
                    )
                    torch.cuda.empty_cache()
                    break

            if aborted:
                break

            prediction = parse.prediction
            correct = prediction is not None and prediction == record.answer_letter
            e2e_seconds = time.perf_counter() - sample_start
            vram = generation["vram"]
            global_peak_allocated = max(
                global_peak_allocated, vram.get("peak_allocated_gib", 0.0)
            )
            global_peak_reserved = max(
                global_peak_reserved, vram.get("peak_reserved_gib", 0.0)
            )

            # 字段名与 Xiaomi 的 predictions.jsonl 保持一致，
            # 只额外增加 model_name（Qwen 特有字段）。
            record_out = {
                "uid": record.uid,
                "idx": record.idx,
                "type": record.type,
                "task": record.task,
                "source": record.source,
                "source_dataset": record.source_dataset,
                "model_name": MODEL_NAME,
                "question": record.question,
                "choices": list(record.choices),
                "ground_truth": record.answer_letter,
                "thinking_mode": mode,
                "enable_thinking": prepared.get("enable_thinking"),
                "mode_verified": prepared.get("mode_verified"),
                "raw_generation": generation["raw_generation"],
                "thinking_text": parse.thinking_text,
                "final_answer_text": parse.final_answer_text,
                "prediction": prediction,
                "correct": correct,
                "invalid": not parse.valid,
                "invalid_reason": parse.invalid_reason,
                "truncated": generation["truncated"],
                "finish_reason": generation["finish_reason"],
                "prompt_tokens": prepared.get("prompt_tokens"),
                "generated_tokens": generation["generated_tokens"],
                "thinking_tokens": generation["thinking_tokens"],
                "answer_tokens": generation["answer_tokens"],
                # Qwen3.5-4B 没有分割掩码输出，保留该字段以对齐表结构。
                "mask_tokens": 0,
                "image_load_seconds": image_load_seconds,
                "preprocess_seconds": prepared.get("preprocess_seconds"),
                "generate_seconds": generation["generate_seconds"],
                "e2e_seconds": e2e_seconds,
                "tokens_per_second": generation["tokens_per_second"],
                "vram": vram,
                "error": error,
                "retryable": error_is_oom,
            }
            if error is not None:
                # 异常不是格式问题，单独标记，方便从统计里区分出来。
                record_out["invalid"] = True
                record_out["invalid_reason"] = "inference_error"
            handle.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            handle.flush()

            if args.verbose or index <= 5:
                _print_sample(record_out, index, len(pending))

    total_seconds = time.perf_counter() - started
    all_records = _read_jsonl(predictions_path)
    if len(all_records) != len(records):
        print(f"警告：已存 {len(all_records)}/{len(records)} 条，本次运行不完整。"
              "用 --resume 继续。")

    gpu_processes = _gpu_processes()
    experiment.update({
        "model_name": MODEL_NAME,
        "model_resolved_path": str(adapter.model_path),
        "model_snapshot": adapter.model_path.name,
        "model_load_seconds": adapter.load_info.get("load_seconds"),
        "vram_after_load": adapter.load_info.get("vram_after_load"),
        "warmup_seconds": warmup_seconds,
        "global_peak_allocated_gib": global_peak_allocated,
        "global_peak_reserved_gib": global_peak_reserved,
        "gpu_used_gib_end": (torch.cuda.mem_get_info(adapter.device)[1]
                             - torch.cuda.mem_get_info(adapter.device)[0]) / 1024 ** 3,
        "gpu_free_gib_end": _free_gib(adapter.device),
        "total_wall_seconds": total_seconds,
        "completed_samples": len(all_records),
        "expected_samples": len(records),
        "status": (
            "completed" if len(all_records) == len(records)
            else ("aborted_oom" if aborted else "partial")
        ),
        "attn_implementation": adapter.load_info.get("attn_implementation"),
        "cpu_offload": adapter.load_info.get("cpu_offload"),
        "hf_device_map": adapter.load_info.get("hf_device_map"),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        # 共享 GPU 标记：若机器上还有别的计算进程，则耗时/吞吐/显存都只能算
        # "共享条件下的测量值"，不能当作独占 GPU 的性能基准。
        "gpu_shared": len(gpu_processes) > 1,
        "gpu_processes_at_end": gpu_processes,
    })
    experiment_path.write_text(
        json.dumps({"config": experiment}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = compute_metrics(all_records, experiment)
    metrics["dataset_distribution"] = describe(records)
    metrics["experiment_mode"] = mode
    metrics["aborted"] = aborted
    (mode_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_summary_csv(mode_dir / "summary.csv", metrics)

    print(f"\n[{mode}] 官方 CV-Bench = "
          f"{metrics['official']['cvbench_accuracy_percent']}")
    print(f"[{mode}] 2D = {100 * (metrics['official']['accuracy_2d'] or 0):.2f}  "
          f"3D = {100 * (metrics['official']['accuracy_3d'] or 0):.2f}  "
          f"invalid = {100 * (metrics['invalid_rate'] or 0):.2f}%")
    return metrics


def validate_run(mode_dir: Path, args: argparse.Namespace) -> bool:
    """检查样本覆盖、重复 UID，以及 metrics.json 与逐样本重算是否一致。"""
    predictions_path = mode_dir / "predictions.jsonl"
    if not predictions_path.exists():
        print(f"FAIL: 缺少 {predictions_path}")
        return False
    records = _read_jsonl(predictions_path)
    expected = _select_records(
        load_records(Path(args.dataset_dir), args.split), args.limit, args.stratified
    )
    expected_uids = {r.uid for r in expected}
    actual_uids = [r["uid"] for r in records]
    ok = True
    if len(actual_uids) != len(set(actual_uids)):
        print(f"FAIL: {predictions_path} 有重复 UID")
        ok = False
    if set(actual_uids) != expected_uids:
        print(f"FAIL: 覆盖不一致（已存 {len(actual_uids)}，应为 {len(expected_uids)}）")
        ok = False
    stored = json.loads((mode_dir / "metrics.json").read_text(encoding="utf-8"))
    recomputed = compute_metrics(records, stored.get("experiment"))
    for key in ("accuracy", "invalid_rate", "truncation_rate"):
        if abs((stored.get(key) or 0) - (recomputed.get(key) or 0)) > 1e-12:
            print(f"FAIL: {key} 与 metrics.json 不一致")
            ok = False
    if stored["official"].get("cvbench_accuracy") != \
            recomputed["official"].get("cvbench_accuracy"):
        print("FAIL: 官方分数与 metrics.json 不一致")
        ok = False
    print(f"{'PASS' if ok else 'FAIL'}: {mode_dir} "
          f"({len(records)} 条, accuracy={recomputed['accuracy']:.4f})")
    return ok


def main() -> int:
    args = parse_args()

    if args.mode == "adaptive":
        print(
            "Adaptive Thinking：Qwen3.5-4B 不支持。\n"
            "该 checkpoint 的 chat template 只暴露布尔参数 `enable_thinking`，"
            "没有 adaptive/auto 开关，也没有 Thinking Budget 参数。\n"
            "为避免把外部规则冒充成原生能力，本程序不会用固定预算或手写路由代替它。"
        )
        return 2

    modes = list(SUPPORTED_MODES) if args.mode == "all" else [args.mode]
    out_root = Path(args.output_dir)

    if args.validate:
        results = []
        for mode in modes:
            results.append(validate_run(
                out_root / (args.run_name or "") / f"thinking_{mode}", args
            ))
        off_dir = out_root / (args.run_name or "") / "thinking_off"
        on_dir = out_root / (args.run_name or "") / "thinking_on"
        if (off_dir / "predictions.jsonl").exists() and \
                (on_dir / "predictions.jsonl").exists():
            comparison = compare_runs(
                _read_jsonl(off_dir / "predictions.jsonl"),
                _read_jsonl(on_dir / "predictions.jsonl"),
            )
            (out_root / (args.run_name or "") / "comparison.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return 0 if all(results) else 1

    if not torch.cuda.is_available():
        print("CUDA 不可用，无法评测。")
        return 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- GPU 安全闸门：显存不足就明确退出，而不是冒着 OOM 影响别人的任务 ----
    device = torch.device(args.device)
    free_before = _free_gib(device)
    total_gib = torch.cuda.mem_get_info(device)[1] / 1024 ** 3
    others = _gpu_processes()
    print(f"GPU: {torch.cuda.get_device_name(device)}  总显存 {total_gib:.1f} GiB  "
          f"空闲 {free_before:.1f} GiB")
    print(f"GPU 上的计算进程数：{len(others)}（>1 表示与他人共享）")
    if free_before < args.require_free_gib:
        print(
            f"空闲显存 {free_before:.1f} GiB < 门槛 {args.require_free_gib} GiB，"
            "本次不加载模型。\n当前服务器有其他 GPU 任务，请等显存释放后再运行；"
            "也可以下调 --require-free-gib，但这会增加 OOM 风险。"
        )
        return 3

    selected_records = _select_records(
        load_records(Path(args.dataset_dir), args.split), args.limit, args.stratified
    )
    print("数据集:", json.dumps(describe(selected_records), indent=2, ensure_ascii=False))

    adapter = Qwen35Adapter(
        model_path=args.model_path, device=args.device, attn_implementation=args.attn
    )
    load_info = adapter.load()
    print(f"模型加载耗时 {load_info['load_seconds']:.2f}s，"
          f"allocated={load_info['vram_after_load']['allocated_gib']:.2f} GiB，"
          f"attn={load_info['attn_implementation']}，"
          f"cpu_offload={load_info['cpu_offload']}")

    image_store = CVBenchImages(Path(args.dataset_dir))
    metrics = {}
    for mode in modes:
        metrics[mode] = run_mode(
            adapter, selected_records, image_store, mode, args, out_root
        )
        if metrics[mode].get("aborted"):
            print(
                "停止：CUDA 显存长时间不足。进度已保存，等 GPU 空闲后用同样的命令"
                "加 --resume 继续。"
            )
            break

    if "off" in metrics and "on" in metrics:
        comparison = compare_runs(
            _read_jsonl(out_root / (args.run_name or "") / "thinking_off" / "predictions.jsonl"),
            _read_jsonl(out_root / (args.run_name or "") / "thinking_on" / "predictions.jsonl"),
        )
        (out_root / (args.run_name or "") / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
