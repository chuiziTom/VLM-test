"""Quick image+text speed benchmark using the official infer.py helpers.

Loads the model once and runs a handful of CV-Bench sample images in both
Thinking Off and Thinking On.  It reuses ``infer.load_model_and_processor`` and
``infer.generate_completion``; mask decoding is intentionally skipped because
this is a text-only speed check.

Run:
    micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/bench_speed.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "eval_xiaomi_cvbench"))

from infer import generate_completion, load_model_and_processor  # noqa: E402
from xiaomi_lens.constants import MASK_TOKEN  # noqa: E402

DEFAULT_SAMPLES = Path(
    "/home/20261024_1/Workspace/qyh/VLM-test/dataset/CV-Bench/samples"
)
DEFAULT_DATASET = Path(
    "/home/20261024_1/Workspace/qyh/VLM-test/dataset/CV-Bench"
)
FORMAT_INSTRUCTION = (
    "\n\nAfter answering the question, put the final choice on a separate line "
    "using exactly this format: FINAL_ANSWER: (X)\n"
    "where X is one of the provided option letters."
)
# (split, image file): 3 x 2D + 3 x 3D.
SELECTED = (
    ("2D", "2d_01.png"), ("2D", "2d_02.png"), ("2D", "2d_05.png"),
    ("3D", "3d_03.png"), ("3D", "3d_04.png"), ("3D", "3d_06.png"),
)


def load_annotation(samples_dir: Path, split: str, filename: str) -> str:
    path = samples_dir / split / "annotations.csv"
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["image_file"] == filename:
                return row["question"]
    raise KeyError(f"{filename} not found in {path}")


def build_dataset_prompts(dataset_dir: Path) -> dict[str, str]:
    """Map question text -> original CV-Bench prompt (with choices)."""
    from cvbench_dataset import load_records

    prompts: dict[str, str] = {}
    for record in load_records(dataset_dir):
        prompts.setdefault(record.question.strip(), record.prompt)
    return prompts


def build_inputs(processor, image, text, device, enable_thinking):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    processor.tokenizer.padding_side = "left"
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    inputs = processor(
        text=[prompt_text], images=[image], return_tensors="pt",
        padding=True, add_special_tokens=False,
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    torch.cuda.synchronize(device)
    return inputs, prompt_text, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="SeerRay-Lab/Xiaomi-Lens-0")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path,
                        default=REPO / "eval_xiaomi_cvbench/results/speed_bench.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    prompts = build_dataset_prompts(args.dataset_dir)

    print("Loading model ...")
    torch.cuda.synchronize(device)
    load_start = time.perf_counter()
    model, processor, mask_token_id = load_model_and_processor(
        Path(args.model_path), device, attn_implementation="flash_attention_2"
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    print(f"Model loaded in {load_seconds:.2f}s; <|mask|> id={mask_token_id}")

    results = {"model_load_seconds": load_seconds, "max_new_tokens": args.max_new_tokens,
               "samples": []}

    for mode, enable_thinking in (("off", False), ("on", True)):
        print(f"\n===== Thinking {mode} =====")
        mode_rows = []
        for index, (split, name) in enumerate(SELECTED):
            image_path = args.samples_dir / split / name
            question = load_annotation(args.samples_dir, split, name)
            # Use the original CV-Bench prompt with choices when available.
            instruction = prompts.get(question.strip(), question) + FORMAT_INSTRUCTION
            image = Image.open(image_path).convert("RGB")

            inputs, prompt_text, preprocess_seconds = build_inputs(
                processor, image, instruction, device, enable_thinking
            )
            expected = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
            if not prompt_text.endswith(expected):
                raise RuntimeError(f"Template did not switch to Thinking {mode}")

            prompt_len = inputs["input_ids"].shape[1]
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            completion_ids, generated_text, mask_count = generate_completion(
                model, processor, inputs, args.max_new_tokens, mask_token_id,
                do_sample=False,
            )
            torch.cuda.synchronize(device)
            generate_seconds = time.perf_counter() - start

            tokens = int(completion_ids.shape[1])
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            row = {
                "mode": mode,
                "image": str(image_path),
                "split": split,
                "question": question,
                "prompt_tokens": int(prompt_len),
                "generated_tokens": tokens,
                "generate_seconds": generate_seconds,
                "preprocess_seconds": preprocess_seconds,
                "tokens_per_second": tokens / generate_seconds if generate_seconds else None,
                "mask_tokens": mask_count,
                "truncated": tokens >= args.max_new_tokens,
                "peak_allocated_gib": peak_alloc,
                "raw_generation": generated_text,
            }
            mode_rows.append(row)
            results["samples"].append(row)
            print(
                f"[{index + 1}/{len(SELECTED)}] {split}/{name} "
                f"tokens={tokens} generate={generate_seconds:.2f}s "
                f"preprocess={preprocess_seconds:.3f}s "
                f"tok/s={tokens / generate_seconds:.1f} "
                f"peak={peak_alloc:.2f}GiB"
            )
            print("  output tail:", repr(generated_text[-220:]))

        total_tokens = sum(r["generated_tokens"] for r in mode_rows)
        total_generate = sum(r["generate_seconds"] for r in mode_rows)
        results[f"summary_{mode}"] = {
            "samples": len(mode_rows),
            "mean_generate_seconds": total_generate / len(mode_rows),
            "mean_generated_tokens": total_tokens / len(mode_rows),
            "mean_preprocess_seconds": sum(
                r["preprocess_seconds"] for r in mode_rows
            ) / len(mode_rows),
            "total_tokens": total_tokens,
            "total_generate_seconds": total_generate,
            "tokens_per_second": total_tokens / total_generate if total_generate else None,
            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in mode_rows),
        }
        summary = results[f"summary_{mode}"]
        print(
            f"[{mode}] mean {summary['mean_generate_seconds']:.2f}s/sample, "
            f"{summary['mean_generated_tokens']:.1f} tokens, "
            f"{summary['tokens_per_second']:.1f} tok/s, "
            f"peak {summary['peak_allocated_gib']:.2f} GiB"
        )
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
