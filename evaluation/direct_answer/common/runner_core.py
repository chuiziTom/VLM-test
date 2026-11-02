"""Direct Answer 统一评测核心。

设计原则：
1. **不修改原有脚本**：数据加载（``cvbench_dataset.py``）、官方指标
   （``metrics.py``）、模型适配器均按文件路径只读导入。
2. **同一套任务内容**：两个模型都使用 ``record.prompt + Direct Answer 指令``，
   只在各自 chat template 中关闭 Thinking。
3. **严格解析**：整段匹配 ``FINAL_ANSWER: (X)``，任何额外文字都算 Invalid。
4. **可断点续跑**：按 uid 跳过已完成样本，配置指纹不一致时拒绝续跑。
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

import torch

from .direct_parser import FORMAT_VIOLATION_REASONS, parse_direct_answer
from .module_loader import load_module

HERE = Path(__file__).resolve().parents[1]      # evaluation/direct_answer
ROOT = HERE.parents[1]                          # 项目根目录
PROMPT_VERSION = "v1"
PARSER_VERSION = "direct-1.0"
EXPERIMENT_NAME = "direct_answer_think_off_max_new_tokens_64_v1"
# 保留 dataset/ 兼容路径：它是指向 datasets/ 的符号链接，历史断点续跑指纹依赖该字符串。
DEFAULT_DATASET_DIR = str(ROOT / "dataset" / "CV-Bench")
DEFAULT_OUTPUT_DIR = str(ROOT / "experiments" / "cv_bench" / "direct_answer_64_v1")
PROMPT_FILE = HERE / "prompts" / f"direct_answer_{PROMPT_VERSION}.txt"


# --------------------------------------------------------------------- CLI
def build_parser(model_label: str, default_model: str, default_attn: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"CV-Bench Direct Answer evaluation ({model_label})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--split", choices=("2d", "3d", "all"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="实验根目录；结果写入 <output-dir>/<model_key>/")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default=default_model)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn", default=default_attn,
                        choices=("flash_attention_2", "sdpa", "eager"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument("--memory-wait", type=int, default=120)
    parser.add_argument("--require-free-gib", type=float, default=12.0)
    parser.add_argument("--oom-retries", type=int, default=3)
    parser.add_argument("--oom-wait", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show", type=int, default=5,
                        help="打印前 N 条样本的完整输出。")
    return parser


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


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL；崩溃留下的最后半行会自动修掉。"""
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
    picks: list = []
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
    print(f"Question : {record['question']}")
    print(f"Choices  : {record['choices']}")
    print(f"GT       : {record['ground_truth']}")
    print(f"Thinking : {record['thinking_mode']} "
          f"(enable_thinking={record['enable_thinking']}, "
          f"prompt_verified={record['mode_verified']})")
    print(f"Raw      : {record['raw_generation']!r}")
    print(f"Parsed   : {record['prediction']}  "
          f"{'Correct' if record['correct'] else 'Incorrect'}  "
          f"invalid={record['invalid']} ({record['invalid_reason']})")
    print(f"Tokens   : {record['generated_tokens']}  "
          f"finish={record['finish_reason']}  truncated={record['truncated']}")
    print(f"Latency  : preprocess={record['preprocess_seconds']:.3f}s "
          f"generate={record['generate_seconds']:.3f}s "
          f"e2e={record['e2e_seconds']:.3f}s")
    print(f"VRAM     : peak_allocated="
          f"{record['vram'].get('peak_allocated_gib', 0):.2f} GiB")
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


def _make_experiment(model_key: str, model_name: str, args, records,
                     instruction_suffix: str, model_dir: Path) -> dict:
    dataset_files = {
        name: _file_fingerprint(Path(args.dataset_dir) / name)
        for name in ("test_2d.parquet", "test_3d.parquet")
        if (Path(args.dataset_dir) / name).exists()
    }
    config = {
        "experiment_name": Path(args.output_dir).name,
        "protocol": "direct_answer",
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "model_key": model_key,
        "model_name": model_name,
        "model_path": args.model_path,
        "thinking_mode": "off",
        "enable_thinking": False,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "split": args.split,
        "limit": args.limit,
        "stratified": args.stratified,
        "seed": args.seed,
        "dataset_dir": args.dataset_dir,
        "dataset_files": dataset_files,
        "prompt_file": str(PROMPT_FILE),
        "prompt_sha1": _sha1(PROMPT_FILE.read_text(encoding="utf-8")),
        "instruction_suffix": instruction_suffix,
        "instruction_suffix_sha1": _sha1(instruction_suffix),
        "selected_uids_sha1": _sha1("|".join(r.uid for r in records)),
        "selected_samples": len(records),
        "eval_code_dir": str(model_dir),
        "attn_requested": args.attn,
        "torch": torch.__version__,
        "transformers": _module_version("transformers"),
        "python": platform.python_version(),
    }
    config["fingerprint"] = _sha1(
        json.dumps(config, sort_keys=True, ensure_ascii=False)
    )
    return config


# ------------------------------------------------------------------- 运行
def _run_model(model_key: str, model_name: str, model_dir: Path,
               adapter_cls, args, dataset_mod, metrics_mod) -> int:
    instruction_suffix = "\n\n" + PROMPT_FILE.read_text(encoding="utf-8").strip()
    out_dir = Path(args.output_dir) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    experiment_path = out_dir / "experiment.json"

    selected = _select_records(
        dataset_mod.load_records(Path(args.dataset_dir), args.split),
        args.limit, args.stratified,
    )
    experiment = _make_experiment(
        model_key, model_name, args, selected, instruction_suffix, model_dir
    )

    if experiment_path.exists():
        previous = json.loads(experiment_path.read_text(encoding="utf-8"))
        if previous.get("config", {}).get("fingerprint") != experiment["fingerprint"]:
            raise SystemExit(
                f"{out_dir} 中的配置与本次参数不一致。请换 --output-dir，"
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
            f"{predictions_path} 已有 {len(existing)} 条记录；"
            "请加 --resume，或换一个 --output-dir。"
        )
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
            print(f"[{model_key}] 丢弃 {len(retry)} 条 OOM 记录，稍后重测")

    done = {r["uid"] for r in existing}
    if len(done) != len(existing):
        raise ValueError(f"{predictions_path} 中存在重复 UID")
    pending = [r for r in selected if r.uid not in done]
    print(f"\n[{model_key}] 已完成 {len(existing)}，待评测 {len(pending)} -> {out_dir}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    free_before = _free_gib(device)
    total_gib = torch.cuda.mem_get_info(device)[1] / 1024 ** 3
    others = _gpu_processes()
    print(f"[{model_key}] GPU: {torch.cuda.get_device_name(device)}  "
          f"总显存 {total_gib:.1f} GiB  空闲 {free_before:.1f} GiB  "
          f"计算进程数 {len(others)}")
    if free_before < args.require_free_gib:
        print(f"[{model_key}] 空闲显存不足 {args.require_free_gib} GiB，停止。")
        return 3

    adapter = adapter_cls(
        model_path=args.model_path, device=args.device,
        attn_implementation=args.attn,
    )
    load_info = adapter.load()
    print(f"[{model_key}] 模型加载 {load_info['load_seconds']:.2f}s  "
          f"attn={load_info['attn_implementation']}  "
          f"allocated={load_info['vram_after_load']['allocated_gib']:.2f} GiB")

    image_store = dataset_mod.CVBenchImages(Path(args.dataset_dir))

    # 预热：不计入成绩，只为了初始化 CUDA kernel 与显存池。
    warmup_seconds = 0.0
    if args.warmup and pending:
        warmup_record = selected[0]
        image = image_store.get(warmup_record)
        for _ in range(args.warmup):
            prepared = adapter.build_inputs(
                image, warmup_record.prompt + instruction_suffix, "off"
            )
            if not prepared["mode_verified"]:
                raise RuntimeError(
                    "chat template 没有按 Thinking Off 渲染，拒绝继续运行。"
                )
            start = time.perf_counter()
            adapter.generate(prepared, "off", args.max_new_tokens, reset_peak=False)
            warmup_seconds += time.perf_counter() - start
        torch.cuda.empty_cache()

    global_peak_allocated = 0.0
    global_peak_reserved = 0.0
    started = time.perf_counter()
    aborted = False
    # 增量计数，避免每 N 条重新读取整个 JSONL。
    completed_this_run = 0
    correct_this_run = 0

    with predictions_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(pending, 1):
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
                "generate_seconds": 0.0, "tokens_per_second": None, "vram": {},
            }
            parse = None
            attempt = 0

            while True:
                if not _wait_for_memory(
                    adapter.device, args.min_free_gib, args.memory_wait
                ):
                    print(f"[{model_key}] 警告：{record.uid} 之前空闲显存偏低，仍尝试运行")
                try:
                    image_start = time.perf_counter()
                    image = image_store.get(record)
                    image_load_seconds = time.perf_counter() - image_start

                    instruction = record.prompt + instruction_suffix
                    prepared = adapter.build_inputs(image, instruction, "off")
                    if not prepared["mode_verified"]:
                        raise RuntimeError("Thinking Off 没有被模板正确应用")

                    generation = adapter.generate(
                        prepared, "off", args.max_new_tokens, reset_peak=True
                    )
                    parse = parse_direct_answer(
                        generation["raw_generation"],
                        record.valid_letters,
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
                        print(f"[{model_key}] {record.uid} CUDA OOM；"
                              f"第 {attempt}/{args.oom_retries} 次重试，等待 {wait}s")
                        time.sleep(wait)
                        continue
                    if oom:
                        print(f"[{model_key}] {record.uid} 持续 OOM，安全停止。")
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
                    parse = parse_direct_answer(
                        "", record.valid_letters, truncated=False
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

            record_out = {
                "uid": record.uid,
                "idx": record.idx,
                "type": record.type,
                "task": record.task,
                "source": record.source,
                "source_dataset": record.source_dataset,
                "model_name": model_name,
                "question": record.question,
                "choices": list(record.choices),
                "ground_truth": record.answer_letter,
                "thinking_mode": "off",
                "enable_thinking": prepared.get("enable_thinking"),
                "mode_verified": prepared.get("mode_verified"),
                "protocol": "direct_answer",
                "prompt_version": PROMPT_VERSION,
                "raw_generation": generation["raw_generation"],
                "normalized_text": parse.normalized_text,
                "thinking_text": "",
                "final_answer_text": parse.normalized_text,
                "prediction": prediction,
                "correct": correct,
                "invalid": not parse.valid,
                "invalid_reason": parse.invalid_reason,
                "format_violation": parse.format_violation,
                "truncated": generation["truncated"],
                "finish_reason": generation["finish_reason"],
                "prompt_tokens": prepared.get("prompt_tokens"),
                "generated_tokens": generation["generated_tokens"],
                "thinking_tokens": generation["thinking_tokens"],
                "answer_tokens": generation["answer_tokens"],
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
                record_out["invalid"] = True
                record_out["invalid_reason"] = "inference_error"
                record_out["format_violation"] = True
            handle.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            handle.flush()
            completed_this_run += 1
            correct_this_run += int(bool(correct))

            if args.verbose or index <= args.show:
                _print_sample(record_out, index, len(pending))
            elif index % 50 == 0:
                print(f"[{model_key}] {index}/{len(pending)}  "
                      f"本次已评测正确率={correct_this_run}/{completed_this_run}",
                      flush=True)

    total_seconds = time.perf_counter() - started
    all_records = _read_jsonl(predictions_path)
    if len(all_records) != len(selected):
        print(f"[{model_key}] 警告：已存 {len(all_records)}/{len(selected)} 条，"
              "本次运行不完整，请用 --resume 继续。")

    gpu_processes = _gpu_processes()
    experiment.update({
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
        "expected_samples": len(selected),
        "status": (
            "completed" if len(all_records) == len(selected)
            else ("aborted_oom" if aborted else "partial")
        ),
        "attn_implementation": adapter.load_info.get("attn_implementation"),
        "cpu_offload": adapter.load_info.get("cpu_offload"),
        "hf_device_map": adapter.load_info.get("hf_device_map"),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "gpu_shared": len(gpu_processes) > 1,
        "gpu_processes_at_end": gpu_processes,
    })
    experiment_path.write_text(
        json.dumps({"config": experiment}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = metrics_mod.compute_metrics(all_records, experiment)
    metrics["dataset_distribution"] = dataset_mod.describe(selected)
    metrics["experiment_mode"] = "direct_answer_off"
    metrics["aborted"] = aborted

    format_violations = sum(1 for r in all_records if r.get("format_violation"))
    metrics["direct_answer"] = {
        "protocol": "direct_answer",
        "prompt_version": PROMPT_VERSION,
        "prompt_file": str(PROMPT_FILE),
        "prompt_sha1": _sha1(PROMPT_FILE.read_text(encoding="utf-8")),
        "instruction_suffix": instruction_suffix,
        "instruction_suffix_sha1": _sha1(instruction_suffix),
        "parser_version": PARSER_VERSION,
        "max_new_tokens": args.max_new_tokens,
        "format_violations": format_violations,
        "format_violation_rate": (
            format_violations / len(all_records) if all_records else None
        ),
        "format_violation_reasons": dict(Counter(
            r.get("invalid_reason") for r in all_records
            if r.get("format_violation")
        )),
        "valid_rate": metrics["valid_rate"],
        "invalid_reason_counts": metrics["invalid_reasons"],
        "note": (
            "整段严格匹配 FINAL_ANSWER: (X)；任何额外文字、额外答案行或格式偏差"
            "都记为 invalid，并保留在准确率分母中。"
        ),
    }
    if metrics["invalid_reasons"]:
        non_format = {
            k: v for k, v in metrics["invalid_reasons"].items()
            if k not in FORMAT_VIOLATION_REASONS
        }
        metrics["direct_answer"]["non_format_invalid_reasons"] = non_format

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_summary_csv(out_dir / "summary.csv", metrics)

    official = metrics["official"]
    print(f"\n[{model_key}] 官方 CV-Bench = {official['cvbench_accuracy_percent']:.4f}%")
    print(f"[{model_key}] 2D = {100 * (official['accuracy_2d'] or 0):.2f}%  "
          f"3D = {100 * (official['accuracy_3d'] or 0):.2f}%  "
          f"invalid = {100 * (metrics['invalid_rate'] or 0):.2f}%  "
          f"format_violation = {100 * (metrics['direct_answer']['format_violation_rate'] or 0):.2f}%")
    print(f"[{model_key}] 平均生成 Token = {metrics['tokens']['generated']['mean']:.2f}  "
          f"平均 e2e = {metrics['timing_seconds']['end_to_end']['mean']:.3f}s")
    return 0


# ----------------------------------------------------------------- 校验
def _validate(model_key: str, model_dir: Path, args, dataset_mod, metrics_mod) -> bool:
    out_dir = Path(args.output_dir) / model_key
    predictions_path = out_dir / "predictions.jsonl"
    if not predictions_path.exists():
        print(f"FAIL: 缺少 {predictions_path}")
        return False
    records = _read_jsonl(predictions_path)
    expected = _select_records(
        dataset_mod.load_records(Path(args.dataset_dir), args.split),
        args.limit, args.stratified,
    )
    expected_uids = {r.uid for r in expected}
    uids = [r["uid"] for r in records]
    ok = True
    if len(uids) != len(set(uids)):
        print(f"FAIL: {predictions_path} 有重复 UID")
        ok = False
    if set(uids) != expected_uids:
        print(f"FAIL: 覆盖不一致（已存 {len(uids)}，应为 {len(expected_uids)}）")
        ok = False

    # 重新解析，确认保存的 prediction 可复现。
    reparsed = 0
    for r in records:
        parse = parse_direct_answer(
            r.get("raw_generation", ""), tuple(
                chr(ord("A") + i) for i in range(len(r["choices"]))
            ), truncated=bool(r.get("truncated")),
        )
        if parse.prediction != r.get("prediction"):
            reparsed += 1
    if reparsed:
        print(f"FAIL: {reparsed} 条重新解析结果与保存值不一致")
        ok = False

    stored = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    recomputed = metrics_mod.compute_metrics(records, stored.get("experiment"))
    for key in ("accuracy", "invalid_rate", "truncation_rate"):
        if abs((stored.get(key) or 0) - (recomputed.get(key) or 0)) > 1e-12:
            print(f"FAIL: {key} 与 metrics.json 不一致")
            ok = False
    if stored["official"].get("cvbench_accuracy") != \
            recomputed["official"].get("cvbench_accuracy"):
        print("FAIL: 官方分数与 metrics.json 不一致")
        ok = False
    print(f"{'PASS' if ok else 'FAIL'}: {out_dir} "
          f"({len(records)} 条, accuracy={recomputed['accuracy']:.4f}, "
          f"official={100 * (recomputed['official']['cvbench_accuracy'] or 0):.4f}%)")
    return ok


# ------------------------------------------------------------------ 入口
def run(model_key: str, model_name: str, model_dir: Path, adapter_cls,
        default_model: str, default_attn: str) -> int:
    parser = build_parser(model_name, default_model, default_attn)
    args = parser.parse_args()

    common_dir = ROOT / "evaluation" / "common"
    dataset_mod = load_module(f"cvbench_dataset_direct_{model_key}",
                              common_dir / "cvbench_dataset.py")
    metrics_mod = load_module(f"metrics_direct_{model_key}",
                              common_dir / "metrics.py")

    if args.validate:
        return 0 if _validate(model_key, model_dir, args, dataset_mod, metrics_mod) else 1

    if not torch.cuda.is_available():
        print("CUDA 不可用，无法评测。")
        return 1
    return _run_model(model_key, model_name, model_dir, adapter_cls, args,
                      dataset_mod, metrics_mod)
