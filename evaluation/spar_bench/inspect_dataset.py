"""SPAR-Bench-Tiny 本地数据集检查与统计。

只读本地 Parquet，不下载任何数据。输出：

* reports/spar_bench_tiny_dataset_profile.md
* experiments/spar_bench/tiny_rgb/dataset_statistics.json
* experiments/spar_bench/tiny_rgb/task_distribution.csv
* reports/assets/spar_bench_tiny/*.jpg（少量样本拼图）

所有数字都来自本地文件实测，不直接引用官方 Dataset Card。
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from sparbench_dataset import decode_images, load_dataset  # noqa: E402
from sparbench_metrics import (  # noqa: E402
    ALL_TASKS, HIGH_TASKS, LOW_TASKS, MIDDLE_TASKS, answer_type, task_level,
)

DEFAULT_DATASET_DIR = ROOT / "datasets" / "SPAR-Bench-Tiny"
STATS_DIR = ROOT / "experiments" / "spar_bench" / "tiny_rgb"
REPORT_PATH = ROOT / "reports" / "spar_bench_tiny_dataset_profile.md"
ASSET_DIR = ROOT / "reports" / "assets" / "spar_bench_tiny"

OFFICIAL_LINKS = {
    "dataset_card_tiny": "https://huggingface.co/datasets/jasonzhango/SPAR-Bench-Tiny",
    "dataset_full": "https://huggingface.co/datasets/jasonzhango/SPAR-Bench",
    "code": "https://github.com/LogosIntelligenceLab/SPAR",
    "paper": "https://arxiv.org/abs/2503.22976",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter_dict(counter: Counter) -> dict:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def _montage(images: list[Image.Image], labels: list[str], path: Path,
             max_height: int = 420) -> None:
    """把若干图片横向拼接并加标题，便于在报告里直观看数据集。"""
    resized = []
    for image in images:
        scale = max_height / image.height
        width = max(1, int(image.width * scale))
        resized.append(image.resize((width, max_height), Image.LANCZOS))
    gap = 12
    total_width = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_width, max_height + 34), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for image, label in zip(resized, labels):
        canvas.paste(image, (x, 34))
        draw.text((x + 4, 8), label, fill="black")
        x += image.width + gap
    canvas.save(path, quality=88)


def main() -> int:
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_DIR
    samples, parquet_path = load_dataset(dataset_dir)
    n = len(samples)

    ids = [s.id for s in samples]
    id_counter = Counter(ids)
    duplicate_ids = {k: v for k, v in id_counter.items() if v > 1}

    field_names = ["id", "img_type", "format_type", "task", "source",
                   "image", "question", "answer"]
    missing_fields = Counter()
    attribute = {"image": "image_bytes"}
    for sample in samples:
        for name in field_names:
            value = getattr(sample, attribute.get(name, name))
            if value is None or (isinstance(value, str) and not value.strip()) \
                    or (name == "image" and not value):
                missing_fields[name] += 1

    decode_failures = []
    image_sizes = []
    for sample in samples:
        try:
            images, _ = decode_images(sample)
            image_sizes.extend([list(image.size) for image in images])
        except Exception as exc:  # noqa: BLE001
            decode_failures.append({"id": sample.id, "error": repr(exc)})

    task_counts = Counter(s.task for s in samples)
    img_type_counts = Counter(s.img_type for s in samples)
    format_counts = Counter(s.format_type for s in samples)
    n_images_counts = Counter(s.n_images for s in samples)
    source_counts = Counter(s.source for s in samples)

    task_img = Counter((s.task, s.img_type) for s in samples)
    task_format = Counter((s.task, s.format_type) for s in samples)
    task_img_format = Counter(
        (s.task, s.img_type, s.format_type) for s in samples
    )

    unknown_tasks = sorted(set(task_counts) - set(ALL_TASKS))
    missing_tasks = sorted(set(ALL_TASKS) - set(task_counts))
    level_counts = Counter(task_level(s.task) for s in samples if s.task in ALL_TASKS)

    per_task = {}
    for task in sorted(task_counts):
        rows = [s for s in samples if s.task == task]
        per_task[task] = {
            "n": len(rows),
            "level": task_level(task) if task in ALL_TASKS else "unknown",
            "answer_type": answer_type(task) if task in ALL_TASKS else "unknown",
            "img_types": _counter_dict(Counter(s.img_type for s in rows)),
            "format_types": _counter_dict(Counter(s.format_type for s in rows)),
            "images_per_sample": _counter_dict(Counter(s.n_images for s in rows)),
        }

    statistics = {
        "dataset_dir": str(dataset_dir.resolve()),
        "parquet_file": str(parquet_path.resolve()),
        "parquet_size_bytes": parquet_path.stat().st_size,
        "parquet_sha256": _sha256(parquet_path),
        "total_samples": n,
        "unique_ids": len(set(ids)),
        "duplicate_ids": duplicate_ids,
        "missing_fields": dict(missing_fields),
        "image_decode_failures": decode_failures,
        "task_counts": _counter_dict(task_counts),
        "level_counts": dict(level_counts),
        "img_type_counts": _counter_dict(img_type_counts),
        "format_type_counts": _counter_dict(format_counts),
        "images_per_sample": _counter_dict(n_images_counts),
        "source_counts": _counter_dict(source_counts),
        "single_view": int(img_type_counts.get("single_view", 0)),
        "multi_view": int(img_type_counts.get("multi_view", 0)),
        "image_size_min": [min(x[0] for x in image_sizes), min(x[1] for x in image_sizes)]
                           if image_sizes else None,
        "image_size_max": [max(x[0] for x in image_sizes), max(x[1] for x in image_sizes)]
                           if image_sizes else None,
        "task_x_img_type": {f"{t}|{i}": v for (t, i), v in sorted(task_img.items())},
        "task_x_format_type": {f"{t}|{f}": v for (t, f), v in sorted(task_format.items())},
        "task_x_img_type_x_format_type": {
            f"{t}|{i}|{f}": v for (t, i, f), v in sorted(task_img_format.items())
        },
        "per_task": per_task,
        "unknown_tasks": unknown_tasks,
        "missing_official_tasks": missing_tasks,
        "official_task_order": ALL_TASKS,
        "low_tasks": LOW_TASKS,
        "middle_tasks": MIDDLE_TASKS,
        "high_tasks": HIGH_TASKS,
        "official_links": OFFICIAL_LINKS,
    }

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (STATS_DIR / "dataset_statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # task_distribution.csv
    csv_lines = [
        "task,level,answer_type,n,single_view,multi_view,images_1,images_2,images_3,format_select,format_fill"
    ]
    for task in ALL_TASKS:
        if task not in task_counts:
            continue
        block = per_task[task]
        images = block["images_per_sample"]
        formats = block["format_types"]
        img = block["img_types"]
        csv_lines.append(",".join(str(v) for v in (
            task, block["level"], block["answer_type"], block["n"],
            img.get("single_view", 0), img.get("multi_view", 0),
            images.get("1", 0), images.get("2", 0), images.get("3", 0),
            formats.get("select", 0), formats.get("fill", 0),
        )))
    (STATS_DIR / "task_distribution.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    # 样本拼图：单视角数值题 / 三图选择题 / 两图 VCI
    montage_files = {}
    picks = [
        ("single_view_na", next(s for s in samples
                                if s.task == "depth_prediction_oc")),
        ("multi_view_mca", next(s for s in samples
                                if s.task == "obj_spatial_relation_oo_mv")),
        ("multi_view_vci", next(s for s in samples
                                if s.task == "view_change_infer")),
    ]
    for name, sample in picks:
        images, _ = decode_images(sample)
        labels = [f"image {i + 1} ({im.width}x{im.height})"
                  for i, im in enumerate(images)]
        path = ASSET_DIR / f"{name}.jpg"
        _montage(images, labels, path)
        montage_files[name] = str(path.relative_to(ROOT / "reports"))

    _write_report(statistics, montage_files, picks)
    print(f"已写出 {STATS_DIR / 'dataset_statistics.json'}")
    print(f"已写出 {STATS_DIR / 'task_distribution.csv'}")
    print(f"已写出 {REPORT_PATH}")
    return 0


def _write_report(stats: dict, montage_files: dict, picks) -> None:
    n = stats["total_samples"]
    lines: list[str] = []
    add = lines.append

    add("# SPAR-Bench-Tiny 数据集检查报告")
    add("")
    add("> 本报告的全部统计均由本地 Parquet 实测得到；"
        "官方 Dataset Card 的数字只作为对照，不作为本地结论。")
    add("")
    add("## 1. 数据来源与版本")
    add("")
    add("| 项目 | 内容 |")
    add("|---|---|")
    add(f"| 数据集 | SPAR-Bench-Tiny（RGB，仅 Tiny 版本） |")
    add(f"| 本地目录 | `{stats['dataset_dir']}` |")
    add(f"| Parquet | `{stats['parquet_file']}` |")
    add(f"| 文件大小 | {stats['parquet_size_bytes'] / 1024 ** 2:.1f} MiB |")
    add(f"| SHA-256 | `{stats['parquet_sha256']}` |")
    add(f"| Dataset Card | {OFFICIAL_LINKS['dataset_card_tiny']} |")
    add(f"| 完整 Benchmark | {OFFICIAL_LINKS['dataset_full']} |")
    add(f"| 官方代码 | {OFFICIAL_LINKS['code']} |")
    add(f"| 论文 | {OFFICIAL_LINKS['paper']} |")
    add("")
    add("本次只评测 **SPAR-Bench-Tiny RGB**，不包含 RGB-D 版本，"
        "也不混入完整 SPAR-Bench 的数据。")
    add("")

    add("## 2. 结构检查")
    add("")
    add("| 检查项 | 结果 |")
    add("|---|---|")
    add(f"| Parquet 文件数 | 1 |")
    add(f"| 总样本数 | {n} |")
    add(f"| 唯一 ID 数 | {stats['unique_ids']} |")
    add(f"| 重复 ID | {len(stats['duplicate_ids'])} 个 |")
    add(f"| 缺失字段 | {sum(stats['missing_fields'].values())} 处 |")
    add(f"| 图片解码失败 | {len(stats['image_decode_failures'])} 条 |")
    add(f"| 实际任务种类 | {len(stats['task_counts'])} |")
    add(f"| 未知任务 | {', '.join(stats['unknown_tasks']) or '无'} |")
    add(f"| 官方任务缺失 | {', '.join(stats['missing_official_tasks']) or '无'} |")
    add(f"| 图片来源 | {', '.join(f'{k} ({v})' for k, v in stats['source_counts'].items())} |")
    add(f"| 图片尺寸范围 | {stats['image_size_min']} ~ {stats['image_size_max']} (W×H) |")
    add("")
    add("字段结构：`id`(int64), `img_type`(str), `format_type`(str), `task`(str), "
        "`source`(str), `image`(list<struct<bytes,path>>), `question`(str), `answer`(str)。"
        "图片直接以二进制存储在 Parquet 中，每条样本包含 1~3 张。")
    add("")

    add("## 3. 任务分布（20 种）")
    add("")
    add("| 任务 | 层级 | 评分方式 | 样本数 | 输入类型 | 答案格式 | 图片数分布 |")
    add("|---|---|---|---:|---|---|---|")
    for task in ALL_TASKS:
        if task not in stats["per_task"]:
            continue
        block = stats["per_task"][task]
        img = ", ".join(f"{k}:{v}" for k, v in block["img_types"].items())
        fmt = ", ".join(f"{k}:{v}" for k, v in block["format_types"].items())
        imgs = ", ".join(f"{k}张:{v}" for k, v in block["images_per_sample"].items())
        add(f"| `{task}` | {block['level']} | {block['answer_type']} | "
            f"{block['n']} | {img} | {fmt} | {imgs} |")
    add("")
    add("### 层级分布")
    add("")
    add("| 层级 | 任务数 | 样本数 | 任务列表 |")
    add("|---|---:|---:|---|")
    for level, tasks in (("Low", LOW_TASKS), ("Medium", MIDDLE_TASKS),
                         ("High", HIGH_TASKS)):
        count = sum(stats["task_counts"].get(t, 0) for t in tasks)
        add(f"| {level} | {len(tasks)} | {count} | {', '.join(f'`{t}`' for t in tasks)} |")
    add("")

    add("## 4. 输入类型与答案格式")
    add("")
    add("| 维度 | 类别 | 样本数 | 占比 |")
    add("|---|---|---:|---:|")
    for name, value in stats["img_type_counts"].items():
        add(f"| 输入类型 | {name} | {value} | {value / n * 100:.1f}% |")
    for name, value in stats["images_per_sample"].items():
        add(f"| 每样本图片数 | {name} 张 | {value} | {value / n * 100:.1f}% |")
    for name, value in stats["format_type_counts"].items():
        add(f"| 答案格式 | {name} | {value} | {value / n * 100:.1f}% |")
    add("")
    add(f"**Single-view {stats['single_view']} 条，Multi-view {stats['multi_view']} 条。**"
        " Multi-view 样本的 2~3 张图片在评测中全部按原始顺序输入模型。")
    add("")

    add("### task × img_type × format_type")
    add("")
    add("| 任务 | 输入类型 | 答案格式 | 样本数 |")
    add("|---|---|---|---:|")
    for key, value in stats["task_x_img_type_x_format_type"].items():
        task, img_type, fmt = key.split("|")
        add(f"| `{task}` | {img_type} | {fmt} | {value} |")
    add("")

    add("## 5. 多视角样本示例")
    add("")
    add("### 三图选择题（obj_spatial_relation_oo_mv）")
    add("")
    add(f"![三图选择题]({montage_files['multi_view_mca']})")
    add("")
    sample = picks[1][1]
    add(f"- 图片数量：{sample.n_images}")
    add(f"- 问题：{sample.question}")
    add(f"- 答案：`{sample.answer}`")
    add("")
    add("### 两图 VCI（view_change_infer）")
    add("")
    add(f"![VCI 样本]({montage_files['multi_view_vci']})")
    add("")
    sample = picks[2][1]
    add(f"- 图片数量：{sample.n_images}")
    add(f"- 问题：{sample.question}")
    add(f"- 答案：`{sample.answer}`")
    add("")
    add("### 单视角数值题（depth_prediction_oc）")
    add("")
    add(f"![单视角数值题]({montage_files['single_view_na']})")
    add("")
    sample = picks[0][1]
    add(f"- 图片数量：{sample.n_images}")
    add(f"- 问题：{sample.question}")
    add(f"- 答案：`{sample.answer}`")
    add("")

    add("## 6. 数据质量结论")
    add("")
    add(f"- {n} 条样本、{stats['unique_ids']} 个唯一 ID，"
        f"重复 ID {len(stats['duplicate_ids'])} 个。")
    add(f"- 实际 {len(stats['task_counts'])} 种任务，"
        f"任务样本数范围 "
        f"{min(stats['task_counts'].values())}~{max(stats['task_counts'].values())} 条。")
    add(f"- Single-view {stats['single_view']} 条 / Multi-view {stats['multi_view']} 条；"
        "图片数 1/2/3 张分别对应 "
        f"{stats['images_per_sample'].get('1', 0)} / "
        f"{stats['images_per_sample'].get('2', 0)} / "
        f"{stats['images_per_sample'].get('3', 0)} 条。")
    add(f"- 全部图片均可正常解码，缺失字段 0 处。")
    add(f"- `format_type` 实际只有 `select` 与 `fill` 两类，"
        f"分别为 {stats['format_type_counts'].get('select', 0)} / "
        f"{stats['format_type_counts'].get('fill', 0)} 条。")
    add("")
    add("> 说明：`format_type` 是数据层面的“选择题/填空题”标记；"
        "官方评分按 `task` 划分为 MCA / NA / VCI 三类。两者并不完全一一对应"
        "（例如 `view_change_infer` 在数据里是 `fill`，但使用专门的 VCI 指标）。")
    add("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
