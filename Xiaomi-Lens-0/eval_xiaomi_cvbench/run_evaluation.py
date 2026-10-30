"""Run CV-Bench evaluation for Xiaomi-Lens-0 with Thinking Off / On.

Examples
--------
Single sample, Thinking Off::

    micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
        --mode off --limit 1

Stratified smoke test (all tasks/sources)::

    micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
        --mode off --limit 16 --stratified

Full benchmark, both modes, resumable::

    micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/run_evaluation.py \
        --mode all --resume
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
from collections import Counter, defaultdict
from pathlib import Path

# Reduce allocator fragmentation; must be set before CUDA is initialized.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "src"))

from answer_parser import parse_final_answer  # noqa: E402
from cvbench_dataset import (  # noqa: E402
    CVBenchImages, describe, load_records,
)
from metrics import compare_runs, compute_metrics  # noqa: E402
from xiaomi_lens_adapter import (  # noqa: E402
    MODEL_NAME, XiaomiLensAdapter, mode_to_enable_thinking,
)

PARSER_VERSION = "1.0"
SUPPORTED_MODES = ("off", "on")
DEFAULT_DATASET_DIR = "/home/20261024_1/Workspace/qyh/VLM-test/dataset/CV-Bench"
DEFAULT_OUTPUT_DIR = str(HERE / "results")
DEFAULT_MODEL = "SeerRay-Lab/Xiaomi-Lens-0"

FORMAT_INSTRUCTION = (
    "\n\nAfter answering the question, put the final choice on a separate line "
    "using exactly this format: FINAL_ANSWER: (X)\n"
    "where X is one of the provided option letters."
)


# --------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CV-Bench evaluation for Xiaomi-Lens-0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("off", "on", "adaptive", "all"),
                        required=True,
                        help="Thinking mode. 'adaptive' is probed, not simulated.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N samples (by idx).")
    parser.add_argument("--stratified", action="store_true",
                        help="With --limit, round-robin across type/task/source.")
    parser.add_argument("--split", choices=("2d", "3d", "all"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="",
                        help="Subdirectory name; isolates runs with different configs.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a run using the same output directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn", default="flash_attention_2",
                        choices=("flash_attention_2", "sdpa", "eager"))
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup generations (not scored).")
    parser.add_argument("--min-free-gib", type=float, default=2.0,
                        help="Wait before a sample while CUDA reports less free "
                             "memory (torch.cuda.mem_get_info, not nvidia-smi).")
    parser.add_argument("--memory-wait", type=int, default=120,
                        help="Max seconds to wait for free GPU memory per sample.")
    parser.add_argument("--oom-retries", type=int, default=3,
                        help="Retries after a CUDA OOM before stopping the run.")
    parser.add_argument("--oom-wait", type=int, default=30,
                        help="Base seconds between OOM retries (multiplied by attempt).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every sample; otherwise only the first five.")
    parser.add_argument("--validate", action="store_true",
                        help="Only validate an existing run, do not load the model.")
    return parser.parse_args()


# --------------------------------------------------------------- utilities
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


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


def _module_version(name: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, repairing a partial trailing line after a crash."""
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
                raise ValueError(f"Corrupt JSONL at {path}:{i + 1}")
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
    rows = [("overall", metrics["total_samples"], metrics["correct"],
             metrics["accuracy_percent"], metrics["invalid"])]
    official = metrics["official"]
    for name, value in (
        ("official_cvbench", official["cvbench_accuracy_percent"]),
        ("official_2d", 100 * official["accuracy_2d"] if official["accuracy_2d"] is not None else None),
        ("official_3d", 100 * official["accuracy_3d"] if official["accuracy_3d"] is not None else None),
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
    """Wait until the GPU has ``min_free_gib`` free, or until the timeout."""
    if min_free_gib <= 0:
        return True
    deadline = time.perf_counter() + timeout_s
    while True:
        free, _ = torch.cuda.mem_get_info(device)
        if free >= min_free_gib * 1024 ** 3:
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(10)


# ------------------------------------------------------------------- run
def run_mode(
    adapter: XiaomiLensAdapter,
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

    # Verify the saved configuration before touching any existing predictions.
    if experiment_path.exists():
        previous = json.loads(experiment_path.read_text(encoding="utf-8"))
        if previous.get("config", {}).get("fingerprint") != experiment["fingerprint"]:
            raise SystemExit(
                f"Configuration mismatch in {mode_dir}. Use a new --run-name or "
                "delete the directory; results from different settings must not mix."
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
            f"{predictions_path} already has {len(existing)} records. "
            "Pass --resume to continue or choose another --run-name."
        )

    # An OOM means the sample was never actually evaluated.  Drop such records
    # so --resume retries them instead of silently counting them as wrong.
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
            print(f"[{mode}] dropped {len(retry)} OOM record(s) for retry")

    done = {r["uid"] for r in existing}
    if len(done) != len(existing):
        raise ValueError(f"Duplicate UIDs in {predictions_path}")

    pending = [r for r in records if r.uid not in done]
    print(f"\n[{mode}] {len(existing)} done, {len(pending)} pending "
          f"-> {mode_dir}")

    # Warmup is measured but never scored.
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
                    f"Chat template did not render Thinking {mode}; refusing to run."
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
                "mask_tokens": 0, "vram": {},
            }
            parse = None
            attempt = 0

            while True:
                if not _wait_for_memory(
                    adapter.device, args.min_free_gib, args.memory_wait
                ):
                    print(f"[{mode}] WARNING: GPU still low on memory before "
                          f"{record.uid}; trying anyway")
                try:
                    image_start = time.perf_counter()
                    image = image_store.get(record)
                    image_load_seconds = time.perf_counter() - image_start

                    instruction = record.prompt + FORMAT_INSTRUCTION
                    prepared = adapter.build_inputs(image, instruction, mode)
                    if not prepared["mode_verified"]:
                        raise RuntimeError(
                            "Thinking mode was not applied by the template"
                        )

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
                        print(f"[{mode}] CUDA OOM on {record.uid}; retry "
                              f"{attempt}/{args.oom_retries} in {wait}s")
                        time.sleep(wait)
                        continue
                    if oom:
                        # Do not store this sample: --resume must retry it later.
                        print(f"[{mode}] CUDA OOM persists on {record.uid}; "
                              "stopping safely. Re-run with --resume.")
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
                        "mask_tokens": 0, "vram": {},
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

            record_out = {
                "uid": record.uid,
                "idx": record.idx,
                "type": record.type,
                "task": record.task,
                "source": record.source,
                "source_dataset": record.source_dataset,
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
                "mask_tokens": generation["mask_tokens"],
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
                # An exception is not a formatting failure; label it explicitly.
                record_out["invalid"] = True
                record_out["invalid_reason"] = "inference_error"
            handle.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            handle.flush()

            if args.verbose or index <= 5:
                _print_sample(record_out, index, len(pending))

    total_seconds = time.perf_counter() - started
    all_records = _read_jsonl(predictions_path)
    if len(all_records) != len(records):
        print(f"WARNING: {len(all_records)}/{len(records)} samples stored; "
              "run is partial. Re-run with --resume to finish.")

    gpu = torch.cuda.mem_get_info(adapter.device)
    experiment.update({
        "model_name": MODEL_NAME,
        "model_resolved_path": str(adapter.model_path),
        "model_snapshot": adapter.model_path.name,
        "model_load_seconds": adapter.load_info.get("load_seconds"),
        "vram_after_load": adapter.load_info.get("vram_after_load"),
        "warmup_seconds": warmup_seconds,
        "global_peak_allocated_gib": global_peak_allocated,
        "global_peak_reserved_gib": global_peak_reserved,
        "gpu_used_gib_end": (gpu[1] - gpu[0]) / 1024 ** 3,
        "total_wall_seconds": total_seconds,
        "completed_samples": len(all_records),
        "expected_samples": len(records),
        "status": (
            "completed" if len(all_records) == len(records)
            else ("aborted_oom" if aborted else "partial")
        ),
        "attn_implementation": adapter.load_info.get("attn_implementation"),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
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

    print(f"\n[{mode}] official CV-Bench = "
          f"{metrics['official']['cvbench_accuracy_percent']}")
    print(f"[{mode}] 2D = {100 * (metrics['official']['accuracy_2d'] or 0):.2f}  "
          f"3D = {100 * (metrics['official']['accuracy_3d'] or 0):.2f}  "
          f"invalid = {100 * (metrics['invalid_rate'] or 0):.2f}%")
    return metrics


def validate_run(mode_dir: Path, args: argparse.Namespace) -> bool:
    """Check coverage, duplicates and recomputed metrics for one run."""
    predictions_path = mode_dir / "predictions.jsonl"
    if not predictions_path.exists():
        print(f"FAIL: missing {predictions_path}")
        return False
    records = _read_jsonl(predictions_path)
    expected = _select_records(
        load_records(Path(args.dataset_dir), args.split), args.limit, args.stratified
    )
    expected_uids = {r.uid for r in expected}
    actual_uids = [r["uid"] for r in records]
    ok = True
    if len(actual_uids) != len(set(actual_uids)):
        print(f"FAIL: duplicate UIDs in {predictions_path}")
        ok = False
    if set(actual_uids) != expected_uids:
        print(f"FAIL: coverage mismatch ({len(actual_uids)} stored, "
              f"{len(expected_uids)} expected)")
        ok = False
    stored = json.loads((mode_dir / "metrics.json").read_text(encoding="utf-8"))
    recomputed = compute_metrics(records, stored.get("experiment"))
    for key in ("accuracy", "invalid_rate", "truncation_rate"):
        if abs((stored.get(key) or 0) - (recomputed.get(key) or 0)) > 1e-12:
            print(f"FAIL: {key} differs from metrics.json")
            ok = False
    a = stored["official"].get("cvbench_accuracy")
    b = recomputed["official"].get("cvbench_accuracy")
    if a != b:
        print("FAIL: official score differs from metrics.json")
        ok = False
    print(f"{'PASS' if ok else 'FAIL'}: {mode_dir} "
          f"({len(records)} samples, accuracy={recomputed['accuracy']:.4f})")
    return ok


def main() -> int:
    args = parse_args()

    if args.mode == "adaptive":
        print(
            "Adaptive Thinking: NOT SUPPORTED by Xiaomi-Lens-0.\n"
            "The checkpoint's chat template exposes only the boolean "
            "`enable_thinking`; there is no native adaptive/auto switch and no "
            "Thinking Budget parameter. Refusing to substitute a hand-written "
            "router or a fixed budget for the native mode."
        )
        return 2

    modes = list(SUPPORTED_MODES) if args.mode == "all" else [args.mode]
    out_root = Path(args.output_dir)

    if args.validate:
        results = []
        for mode in modes:
            mode_dir = out_root / (args.run_name or "") / f"thinking_{mode}"
            results.append(validate_run(mode_dir, args))
        off_dir = out_root / (args.run_name or "") / "thinking_off"
        on_dir = out_root / (args.run_name or "") / "thinking_on"
        if off_dir.joinpath("predictions.jsonl").exists() and \
                on_dir.joinpath("predictions.jsonl").exists():
            comparison = compare_runs(
                _read_jsonl(off_dir / "predictions.jsonl"),
                _read_jsonl(on_dir / "predictions.jsonl"),
            )
            (out_root / (args.run_name or "") / "comparison.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return 0 if all(results) else 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    selected_records = _select_records(
        load_records(Path(args.dataset_dir), args.split), args.limit, args.stratified
    )
    print("Dataset:", json.dumps(describe(selected_records), indent=2, ensure_ascii=False))

    adapter = XiaomiLensAdapter(
        model_path=args.model_path, device=args.device, attn_implementation=args.attn
    )
    load_info = adapter.load()
    print("Model loaded in "
          f"{load_info['load_seconds']:.2f}s, "
          f"allocated={load_info['vram_after_load']['allocated_gib']:.2f} GiB, "
          f"attn={load_info['attn_implementation']}")

    image_store = CVBenchImages(Path(args.dataset_dir))
    metrics = {}
    for mode in modes:
        metrics[mode] = run_mode(
            adapter, selected_records, image_store, mode, args, out_root
        )
        if metrics[mode].get("aborted"):
            print(
                "Stopping: CUDA memory was unavailable for too long. "
                "Progress is saved; re-run the same command with --resume "
                "once the GPU has free memory."
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
