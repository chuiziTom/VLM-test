"""SPAR-Bench-Tiny 全量评测入口。

三个实验协议（每个模型各跑三组）：

===============  ===============  ================  =====================
协议               enable_thinking  max_new_tokens    Prompt 版本
===============  ===============  ================  =====================
think_off_8192   False            8192              official_v1
think_on_8192    True             8192              official_v1
direct_off_64    False            64                direct_answer_v1
===============  ===============  ================  =====================

特性：逐样本落盘、断点续跑、配置指纹校验、OOM 重试、CUDA 同步计时、
Prefill/Decode 分离、逐样本显存峰值、Thinking 后置答案提取。

示例::

    # 冒烟：两种模型各自覆盖 20 个任务
    micromamba run -n vlm-test python evaluation/spar_bench/run_evaluation.py \
        --model qwen35 --protocol think_off_8192 --limit 40 --stratified \
        --run-name smoke

    # 全量
    micromamba run -n vlm-test python evaluation/spar_bench/run_evaluation.py \
        --model qwen35 --protocol think_off_8192 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# 减少显存碎片；必须在 CUDA 初始化前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from sparbench_dataset import decode_images, load_dataset, select_samples  # noqa: E402
from sparbench_metrics import (  # noqa: E402
    ALL_TASKS, answer_type, compute_metrics, format_percent, official_score_for,
    task_level, write_summary_csv,
)
from sparbench_parser import (  # noqa: E402
    parse_strict, split_final_answer, strip_special_tokens,
)
from sparbench_prompts import (  # noqa: E402
    PROMPT_VERSION_DIRECT, PROMPT_VERSION_OFFICIAL, build_prompt, prompt_sha1,
)

PARSER_VERSION = "sparbench_strict_v1"
DEFAULT_DATASET_DIR = str(ROOT / "datasets" / "SPAR-Bench-Tiny")
DEFAULT_OUTPUT_ROOT = str(ROOT / "experiments" / "spar_bench" / "tiny_rgb")

MODEL_REGISTRY = {
    "xiaomi_lens": {
        "dir": "xiaomi_lens_0",
        "id": "SeerRay-Lab/Xiaomi-Lens-0",
        "attn": "flash_attention_2",
        "module": "xiaomi_lens_adapter",
        "class": "SparXiaomiLensAdapter",
    },
    "qwen35": {
        "dir": "qwen3_5_4b",
        "id": "Qwen/Qwen3.5-4B",
        "attn": "sdpa",
        "module": "qwen35_adapter",
        "class": "SparQwen35Adapter",
    },
}

PROTOCOLS = {
    "think_off_8192": {
        "thinking_mode": "off",
        "enable_thinking": False,
        "max_new_tokens": 8192,
        "prompt_version": PROMPT_VERSION_OFFICIAL,
        "direct_answer": False,
    },
    "think_on_8192": {
        "thinking_mode": "on",
        "enable_thinking": True,
        "max_new_tokens": 8192,
        "prompt_version": PROMPT_VERSION_OFFICIAL,
        "direct_answer": False,
    },
    "direct_off_64": {
        "thinking_mode": "off",
        "enable_thinking": False,
        "max_new_tokens": 64,
        "prompt_version": PROMPT_VERSION_DIRECT,
        "direct_answer": True,
    },
}


# ---------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPAR-Bench-Tiny evaluation (RGB, Tiny, 20 tasks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), required=True)
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="覆盖协议默认值（正式实验不建议修改）")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default="",
                        help="放在协议目录下的子目录，用于冒烟测试")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", default=None,
                        help="本地 checkpoint 目录；默认用 HF 缓存")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn", default=None,
                        help="默认 Xiaomi=flash_attention_2, Qwen=sdpa")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-prefill", action="store_true",
                        help="不单独测 Prefill 耗时")
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument("--memory-wait", type=int, default=120)
    parser.add_argument("--oom-retries", type=int, default=3)
    parser.add_argument("--oom-wait", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--recompute", action="store_true",
                        help="不加载模型，只从 predictions.jsonl 重算 metrics.json")
    return parser.parse_args()


# ------------------------------------------------------------------ helpers
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _module_version(name: str) -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL，并修复崩溃留下的半行。"""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict] = []
    last_good = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
            last_good = i + 1
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"{path}:{i + 1} 存在损坏的 JSONL")
            break
    if last_good < len(lines):
        path.write_text(
            "\n".join(lines[:last_good]) + ("\n" if last_good else ""),
            encoding="utf-8",
        )
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _gpu_info() -> dict:
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_gib": total / gib,
        "free_gib": free / gib,
        "used_gib": (total - free) / gib,
        "torch_allocated_gib": torch.cuda.memory_allocated() / gib,
        "torch_reserved_gib": torch.cuda.memory_reserved() / gib,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gib_props": props.total_memory / gib,
    }


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or \
        "out of memory" in str(exc).lower()


def _wait_for_memory(min_free_gib: float, timeout_s: float) -> bool:
    if min_free_gib <= 0:
        return True
    deadline = time.perf_counter() + timeout_s
    while True:
        free, _ = torch.cuda.mem_get_info()
        if free >= min_free_gib * 1024 ** 3:
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(10)


# ------------------------------------------------------------- run config
def _make_config(args, model_info, protocol, samples, parquet_path: Path) -> dict:
    prompts = [
        build_prompt(s.task, s.question, protocol["prompt_version"])
        for s in samples
    ]
    config = {
        "benchmark": "SPAR-Bench-Tiny",
        "modality": "RGB",
        "model_name": model_info["dir"],
        "model_id": model_info["id"],
        "model_path": args.model_path or model_info["id"],
        "protocol": args.protocol,
        "thinking_mode": protocol["thinking_mode"],
        "enable_thinking": protocol["enable_thinking"],
        "max_new_tokens": args.max_new_tokens or protocol["max_new_tokens"],
        "do_sample": False,
        "use_cache": True,
        "batch_size": 1,
        "dtype": "bfloat16",
        "prompt_version": protocol["prompt_version"],
        "prompt_sha1": _sha1("\n".join(prompts)),
        "prompt_count": len(prompts),
        "direct_answer": protocol["direct_answer"],
        "parser_version": PARSER_VERSION,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_file": str(parquet_path),
        "dataset_sha256": _sha256_file(parquet_path),
        "dataset_rows": len(samples),
        "selected_samples": len(samples),
        "selected_uids_sha1": _sha1("|".join(s.uid for s in samples)),
        "limit": args.limit,
        "stratified": args.stratified,
        "seed": 0,
        "image_policy": "all images in original order; no resizing beyond processor",
        "torch": torch.__version__,
        "transformers": _module_version("transformers"),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    config["fingerprint"] = _sha1(json.dumps(config, sort_keys=True, ensure_ascii=False))
    return config


# --------------------------------------------------------------- summary
def _save_outputs(out_dir: Path, metrics: dict, experiment: dict) -> None:
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_csv(out_dir / "summary.csv", metrics)
    (out_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _print_record(record: dict, index: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"[{index}/{total}] id={record['id']} task={record['task']} "
          f"level={record['task_level']} img_type={record['img_type']} "
          f"images={record['n_images']}")
    print(f"Question : {record['question'][:300]}")
    print(f"GT       : {record['ground_truth']}")
    print(f"Thinking : {record['thinking_mode']} "
          f"(enable={record['enable_thinking']}, verified={record['mode_verified']}, "
          f"closed={record['closed_thinking']})")
    print(f"Raw      : {record['raw_generation']!r}")
    print(f"Final    : {record['final_answer_text']!r}")
    print(f"Official : {record['official_metric']} score={record['official_score']:.4f}")
    print(f"Strict   : {record['strict_prediction']!r} "
          f"score={record['strict_score']:.4f} valid={record['strict_valid']} "
          f"({record['invalid_reason']})")
    print(f"Tokens   : prompt={record['prompt_tokens']} gen={record['generated_tokens']} "
          f"think={record['thinking_tokens']} answer={record['answer_tokens']}")
    print(f"Time     : decode={record['image_decode_seconds']:.3f}s "
          f"preprocess={record['preprocess_seconds']:.3f}s "
          f"prefill={record['prefill_seconds']} "
          f"generate={record['generate_seconds']:.3f}s "
          f"e2e={record['e2e_seconds']:.3f}s")
    print("=" * 78)


# ---------------------------------------------------------------- scoring
def _score_and_build_record(
    sample,
    prepared: dict,
    gen: dict,
    protocol: dict,
    decode_seconds: float,
    image_sizes: list[list[int]],
    e2e_seconds: float,
    prefill_seconds: float | None,
    instruction: str,
) -> dict:
    """把一次生成转成完整预测记录（含官方分与严格分）。"""
    enable_thinking = protocol["enable_thinking"]
    thinking_text, final_text, closed = split_final_answer(
        gen["raw_generation"], enable_thinking
    )
    strict = parse_strict(
        sample.task,
        final_text,
        sample.answer,
        enable_thinking=enable_thinking,
        closed_thinking=closed,
        truncated=gen["truncated"],
    )
    official_score, official_reason = official_score_for(
        sample.task, final_text, sample.answer
    )
    invalid = (not strict.valid) or (official_reason is not None)
    invalid_reason = strict.invalid_reason if not strict.valid else official_reason

    kind = answer_type(sample.task)
    record = {
        "idx": sample.idx,
        "id": sample.id,
        "uid": sample.uid,
        "task": sample.task,
        "task_level": task_level(sample.task),
        "answer_type": kind,
        "img_type": sample.img_type,
        "format_type": sample.format_type,
        "source": sample.source,
        "question": sample.question,
        "ground_truth": sample.answer,
        "instruction": instruction,
        "n_images": sample.n_images,
        "image_sizes": image_sizes,
        "thinking_mode": protocol["thinking_mode"],
        "enable_thinking": enable_thinking,
        "mode_verified": prepared["mode_verified"],
        "image_count_ok": prepared.get("image_count_ok"),
        "n_images_prompt": prepared.get("n_images_prompt"),
        "prompt_version": protocol["prompt_version"],
        "prompt_sha1": prompt_sha1(instruction),
        "prompt_tokens": prepared["prompt_tokens"],
        "raw_generation": gen["raw_generation"],
        "thinking_text": thinking_text,
        "final_answer_text": final_text,
        "closed_thinking": closed,
        "finish_reason": gen["finish_reason"],
        "truncated": gen["truncated"],
        "official_metric": (
            "accuracy" if kind == "MCA" else
            "MRA:.5:.95:.05" if kind == "NA" else "vci_metric"
        ),
        "official_score": official_score,
        "official_parse_failed": official_reason,
        "official_correct": (official_score >= 1.0) if kind == "MCA" else None,
        "strict_prediction": strict.prediction,
        "strict_score": strict.score,
        "strict_valid": strict.valid,
        "strict_correct": (strict.score >= 1.0) if kind == "MCA" else None,
        "invalid": bool(invalid),
        "invalid_reason": invalid_reason,
        "generated_tokens": gen["generated_tokens"],
        "thinking_tokens": gen["thinking_tokens"],
        "answer_tokens": gen["answer_tokens"],
        "image_decode_seconds": decode_seconds,
        "preprocess_seconds": prepared["preprocess_seconds"],
        "prefill_seconds": prefill_seconds,
        "generate_seconds": gen["generate_seconds"],
        "e2e_seconds": e2e_seconds,
        "tokens_per_second": gen["tokens_per_second"],
        "vram": gen["vram"],
        "error": None,
        "retryable": False,
    }
    return record


def _error_record(sample, error: str, retryable: bool, protocol: dict) -> dict:
    return {
        "idx": sample.idx,
        "id": sample.id,
        "uid": sample.uid,
        "task": sample.task,
        "task_level": task_level(sample.task),
        "answer_type": answer_type(sample.task),
        "img_type": sample.img_type,
        "format_type": sample.format_type,
        "source": sample.source,
        "question": sample.question,
        "ground_truth": sample.answer,
        "n_images": sample.n_images,
        "thinking_mode": protocol["thinking_mode"],
        "enable_thinking": protocol["enable_thinking"],
        "official_score": 0.0,
        "strict_score": 0.0,
        "strict_valid": False,
        "invalid": True,
        "invalid_reason": "inference_error",
        "truncated": False,
        "error": error,
        "retryable": retryable,
    }


# ------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> int:
    model_info = MODEL_REGISTRY[args.model]
    protocol = PROTOCOLS[args.protocol]
    max_new_tokens = args.max_new_tokens or protocol["max_new_tokens"]
    protocol = dict(protocol, max_new_tokens=max_new_tokens)

    samples, parquet_path = load_dataset(args.dataset_dir)
    samples = select_samples(samples, args.limit, args.stratified)
    if not samples:
        raise SystemExit("没有选中任何样本")

    out_root = Path(args.output_root)
    out_dir = out_root / model_info["dir"] / args.protocol
    if args.run_name:
        out_dir = out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    experiment_path = out_dir / "experiment.json"

    config = _make_config(args, model_info, protocol, samples, parquet_path)

    # ---------------- 配置指纹校验：不同配置绝不混在同一个文件里
    experiment: dict = {
        "config": config,
        "status": "not_started",
        "started_at": _now(),
        "updated_at": _now(),
        "finished_at": None,
        "load_info": None,
        "gpu_start": _gpu_info(),
        "gpu_end": None,
        "wall_seconds": None,
        "global_peak_allocated_gib": None,
        "global_peak_reserved_gib": None,
        "completed_records": 0,
        "expected_records": len(samples),
    }
    if experiment_path.exists():
        previous = json.loads(experiment_path.read_text(encoding="utf-8"))
        prev_fp = previous.get("config", {}).get("fingerprint")
        if prev_fp != config["fingerprint"]:
            raise SystemExit(
                f"配置指纹不一致：{out_dir}\n"
                f"  已存在: {prev_fp}\n  本次: {config['fingerprint']}\n"
                "请换 --run-name，或确认后删除该目录；不同配置的结果不能混用。"
            )
        experiment = previous
        experiment["config"] = config
        experiment["updated_at"] = _now()
        experiment["expected_records"] = len(samples)
    else:
        experiment_path.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    existing = _read_jsonl(predictions_path)
    if existing and not args.resume:
        raise SystemExit(
            f"{predictions_path} 已有 {len(existing)} 条记录；"
            "加 --resume 继续，或换 --run-name。"
        )

    # OOM / 可重试错误不算完成，resume 时删掉重跑。
    if args.resume and existing:
        retry = [r for r in existing if r.get("retryable")]
        if retry:
            existing = [r for r in existing if not r.get("retryable")]
            _write_jsonl(predictions_path, existing)
            print(f"移除 {len(retry)} 条可重试错误记录，稍后重跑")

    done = {r["id"] for r in existing}
    if len(done) != len(existing):
        raise ValueError(f"{predictions_path} 存在重复 id")

    pending = [s for s in samples if s.id not in done]
    print(f"数据集 : {parquet_path}")
    print(f"模型   : {model_info['id']}  ({args.model})")
    print(f"协议   : {args.protocol}  thinking={protocol['thinking_mode']} "
          f"max_new_tokens={max_new_tokens} prompt={protocol['prompt_version']}")
    print(f"输出   : {out_dir}")
    print(f"样本   : 共 {len(samples)}，已完成 {len(done)}，待跑 {len(pending)}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

    def _meta() -> dict:
        return {**config, **experiment, **(experiment.get("load_info") or {})}

    if args.recompute:
        metrics = compute_metrics(existing, meta=_meta())
        _save_outputs(out_dir, metrics, experiment)
        print(f"已重算 metrics.json：{metrics['n_samples']} 条")
        return 0

    # ---------------- 加载模型
    import importlib
    adapter_module = importlib.import_module(model_info["module"])
    adapter_cls = getattr(adapter_module, model_info["class"])
    adapter = adapter_cls(
        args.model_path or model_info["id"],
        device=args.device,
        attn_implementation=args.attn or model_info["attn"],
    )
    load_info = adapter.load(measure_prefill=not args.no_prefill)
    experiment["load_info"] = load_info
    experiment["status"] = "running"
    experiment["updated_at"] = _now()
    experiment["started_at"] = experiment.get("started_at") or _now()
    experiment["gpu_start"] = experiment.get("gpu_start") or _gpu_info()
    experiment["gpu_after_load"] = _gpu_info()
    experiment_path.write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"模型已加载：{load_info['model_path']}  "
          f"attn={load_info['attn_implementation']}  "
          f"load={load_info['load_seconds']:.1f}s  "
          f"cpu_offload={load_info['cpu_offload']}")

    start_wall = time.perf_counter()

    # ---------------- Warmup（不计分）
    if args.warmup > 0 and pending:
        warm_sample = pending[0]
        try:
            images, _ = decode_images(warm_sample)
            instruction = build_prompt(
                warm_sample.task, warm_sample.question, protocol["prompt_version"]
            )
            prepared = adapter.build_inputs(
                images, instruction, protocol["enable_thinking"]
            )
            adapter.generate(
                prepared, protocol["enable_thinking"],
                min(16, max_new_tokens), reset_peak=True,
            )
            print("Warmup 完成")
        except Exception as exc:  # warmup 失败不致命，正式样本会暴露问题
            print(f"Warmup 失败（继续）：{exc}")
            traceback.print_exc()

    # ---------------- 主循环
    metrics_every = 25
    global_peak_allocated = 0.0
    global_peak_reserved = 0.0
    processed_since_save = 0
    error_records: list[dict] = []

    with predictions_path.open("a", encoding="utf-8") as handle:
        progress = tqdm(pending, desc=args.protocol, unit="sample")
        for index, sample in enumerate(progress, start=1):
            if not _wait_for_memory(args.min_free_gib, args.memory_wait):
                print("GPU 可用显存不足，安全停止；已完成的样本可直接 --resume")
                break

            attempt = 0
            record = None
            while True:
                try:
                    e2e_start = time.perf_counter()
                    images, decode_seconds = decode_images(sample)
                    instruction = build_prompt(
                        sample.task, sample.question, protocol["prompt_version"]
                    )
                    prepared = adapter.build_inputs(
                        images, instruction, protocol["enable_thinking"]
                    )
                    image_sizes = [list(image.size) for image in images]

                    prefill_seconds = None
                    if not args.no_prefill:
                        prefill_seconds = adapter.measure_prefill_time(prepared)

                    gen = adapter.generate(
                        prepared,
                        protocol["enable_thinking"],
                        max_new_tokens,
                        reset_peak=True,
                    )
                    e2e_seconds = time.perf_counter() - e2e_start

                    record = _score_and_build_record(
                        sample, prepared, gen, protocol,
                        decode_seconds, image_sizes, e2e_seconds,
                        prefill_seconds, instruction,
                    )
                    record["error"] = None
                    record["retryable"] = False
                    break
                except Exception as exc:
                    is_oom = _is_oom(exc)
                    if is_oom and attempt < args.oom_retries:
                        attempt += 1
                        print(f"\n[OOM] id={sample.id} 第 {attempt} 次重试：{exc}")
                        torch.cuda.empty_cache()
                        time.sleep(args.oom_wait * attempt)
                        continue

                    error_record = _error_record(
                        sample, f"{type(exc).__name__}: {exc}", True, protocol
                    )
                    handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    existing.append(error_record)
                    error_records.append(error_record)
                    print(f"\n[ERROR] id={sample.id}: {type(exc).__name__}: {exc}")
                    if is_oom:
                        traceback.print_exc()
                        experiment["status"] = "error"
                        experiment["updated_at"] = _now()
                        experiment_path.write_text(
                            json.dumps(experiment, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        raise SystemExit(
                            "OOM 重试耗尽，已安全停止；该样本标记为 retryable，"
                            "恢复后会自动重跑。"
                        )
                    record = None
                    break

            if record is None:
                continue

            # 逐条落盘 + fsync，进程被杀也不会丢已完成样本。
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            existing.append(record)
            processed_since_save += 1

            vram = record.get("vram") or {}
            global_peak_allocated = max(
                global_peak_allocated, float(vram.get("peak_allocated_gib") or 0)
            )
            global_peak_reserved = max(
                global_peak_reserved, float(vram.get("peak_reserved_gib") or 0)
            )

            if args.verbose or index <= 3:
                _print_record(record, index, len(pending))

            if processed_since_save >= metrics_every:
                processed_since_save = 0
                experiment["completed_records"] = len(existing)
                experiment["global_peak_allocated_gib"] = global_peak_allocated
                experiment["global_peak_reserved_gib"] = global_peak_reserved
                experiment["updated_at"] = _now()
                metrics = compute_metrics(existing, meta=_meta())
                _save_outputs(out_dir, metrics, experiment)

    # ---------------- 收尾
    wall_seconds = time.perf_counter() - start_wall
    experiment["wall_seconds"] = wall_seconds
    experiment["completed_records"] = len(existing)
    experiment["global_peak_allocated_gib"] = global_peak_allocated
    experiment["global_peak_reserved_gib"] = global_peak_reserved
    experiment["updated_at"] = _now()

    complete = len(existing) >= len(samples) and not error_records
    experiment["status"] = "complete" if complete else "partial"
    experiment["finished_at"] = _now()
    experiment["gpu_end"] = _gpu_info()

    metrics = compute_metrics(existing, meta=_meta())
    _save_outputs(out_dir, metrics, experiment)

    print("\n" + "=" * 78)
    print(f"状态       : {experiment['status']}")
    print(f"完成样本   : {metrics['n_samples']} / {len(samples)}")
    print(f"任务数     : {metrics['n_tasks']} / {len(ALL_TASKS)}")
    print(f"Official   : {format_percent(metrics['overall']['official'])}")
    print(f"Strict     : {format_percent(metrics['overall']['strict'])}")
    for level in ("low", "medium", "high"):
        block = metrics["levels"][level]
        print(f"  {level:<8}: {format_percent(block['official_score'])} "
              f"(strict {format_percent(block['strict_score'])})")
    q = metrics["quality"]
    print(f"Invalid    : {q['invalid']}  Truncated: {q['truncated']}  "
          f"Errors: {q['errors']}  OOM: {q['oom']}")
    print(f"产物       : {out_dir}")
    print("=" * 78)
    return 0


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
