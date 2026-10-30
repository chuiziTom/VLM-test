"""数据集、官方加权分数、Invalid 计分与断点续跑工具链的单元测试。

全部在 CPU 上运行，不加载模型、不占 GPU：

    micromamba run -n vlm-test python eval_qwen35_cvbench/test_metrics.py

解析器的测试在 test_answer_parser.py（与 Xiaomi 评测目录逐字节相同）。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cvbench_dataset import describe, load_records  # noqa: E402
from metrics import compare_runs, compute_metrics  # noqa: E402

DATASET_DIR = Path("/home/20261024_1/Workspace/qyh/VLM-test/dataset/CV-Bench")


def rec(uid, type_, task, source, correct, *, invalid=False,
        invalid_reason=None, generate=1.0, e2e=2.0, tokens=10,
        peak=10.0, truncated=False):
    """构造一条最小的预测记录。"""
    return {
        "uid": uid, "type": type_, "task": task, "source": source,
        "correct": correct, "invalid": invalid,
        "invalid_reason": invalid_reason, "truncated": truncated,
        "finish_reason": "length" if truncated else "eos",
        "generate_seconds": generate, "e2e_seconds": e2e,
        "generated_tokens": tokens, "thinking_tokens": 0, "answer_tokens": tokens,
        "vram": {"peak_allocated_gib": peak}, "error": None,
    }


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records(DATASET_DIR, "all")

    def test_total_matches_official(self):
        self.assertEqual(len(self.records), 2638)

    def test_split_counts(self):
        dist = describe(self.records)
        self.assertEqual(dist["by_type"], {"2D": 1438, "3D": 1200})

    def test_tasks(self):
        dist = describe(self.records)
        self.assertEqual(
            dist["by_task"],
            {"Count": 788, "Relation": 650, "Depth": 600, "Distance": 600},
        )

    def test_sources(self):
        dist = describe(self.records)
        self.assertEqual(dist["by_source"], {"ADE20K": 633, "COCO": 805,
                                             "Omni3D": 1200})

    def test_uid_unique(self):
        uids = [r.uid for r in self.records]
        self.assertEqual(len(uids), len(set(uids)))

    def test_valid_letters_cover_choices(self):
        # 选项数量不固定：A~F 都可能出现，字母集合必须与 choices 长度一致。
        for record in self.records[:200]:
            self.assertEqual(len(record.valid_letters), len(record.choices))
            self.assertEqual(record.valid_letters[0], "A")

    def test_answer_letter_parsed(self):
        for record in self.records:
            self.assertIsNotNone(record.answer_letter)
            self.assertIn(record.answer_letter, record.valid_letters)

    def test_split_subset(self):
        only_3d = load_records(DATASET_DIR, "3d")
        self.assertEqual(len(only_3d), 1200)
        self.assertTrue(all(r.type == "3D" for r in only_3d))


class OfficialScoreTests(unittest.TestCase):
    def test_weighted_formula(self):
        # ADE20K 1/2，COCO 2/4，Omni3D 3/4。
        # 2D = (0.5 + 0.5)/2 = 0.5，3D = 0.75，总分 = (0.5 + 0.75)/2 = 0.625
        records = [
            rec("u1", "2D", "Count", "ADE20K", True),
            rec("u2", "2D", "Count", "ADE20K", False),
            rec("u3", "2D", "Relation", "COCO", True),
            rec("u4", "2D", "Relation", "COCO", True),
            rec("u5", "2D", "Relation", "COCO", False),
            rec("u6", "2D", "Relation", "COCO", False),
            rec("u7", "3D", "Depth", "Omni3D", True),
            rec("u8", "3D", "Depth", "Omni3D", True),
            rec("u9", "3D", "Distance", "Omni3D", True),
            rec("u10", "3D", "Distance", "Omni3D", False),
        ]
        metrics = compute_metrics(records)
        official = metrics["official"]
        self.assertAlmostEqual(official["accuracy_2d_ade20k"], 0.5)
        self.assertAlmostEqual(official["accuracy_2d_coco"], 0.5)
        self.assertAlmostEqual(official["accuracy_2d"], 0.5)
        self.assertAlmostEqual(official["accuracy_3d"], 0.75)
        self.assertAlmostEqual(official["cvbench_accuracy"], 0.625)
        self.assertTrue(official["computable"])
        # 普通样本平均 = 6/10 = 0.6，与官方 0.625 不同，必须分开报告。
        self.assertAlmostEqual(metrics["accuracy"], 0.6)

    def test_invalid_counts_in_denominator(self):
        records = [
            rec("u1", "2D", "Count", "ADE20K", True),
            rec("u2", "2D", "Count", "ADE20K", False, invalid=True,
                invalid_reason="no_final_answer"),
            rec("u3", "2D", "Count", "ADE20K", False, invalid=True,
                invalid_reason="no_final_answer"),
            rec("u4", "2D", "Count", "ADE20K", True),
        ]
        metrics = compute_metrics(records)
        # Invalid 不会被剔除：4 条里有 2 条正确 = 50%。
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["invalid"], 2)
        self.assertAlmostEqual(metrics["invalid_rate"], 0.5)
        self.assertEqual(metrics["invalid_reasons"], {"no_final_answer": 2})

    def test_missing_source_is_not_computable(self):
        # 只有 2D 样本：官方 3D 分数缺失，总分为 None，而不是编造一个值。
        records = [rec("u1", "2D", "Count", "ADE20K", True),
                   rec("u2", "2D", "Count", "COCO", True)]
        official = compute_metrics(records)["official"]
        self.assertIsNone(official["accuracy_3d"])
        self.assertIsNone(official["cvbench_accuracy"])
        self.assertFalse(official["computable"])

    def test_subset_flag(self):
        records = [rec(f"u{i}", "2D", "Count", "ADE20K", True) for i in range(5)]
        self.assertTrue(compute_metrics(records)["subset"])

    def test_truncation_counted(self):
        records = [rec("u1", "2D", "Count", "ADE20K", False,
                       invalid=True, invalid_reason="truncated_no_final_answer",
                       truncated=True)]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["truncated"], 1)
        self.assertAlmostEqual(metrics["truncation_rate"], 1.0)

    def test_timing_and_throughput(self):
        records = [rec(f"u{i}", "2D", "Count", "ADE20K", True,
                       generate=1.0, e2e=2.0, tokens=100) for i in range(4)]
        metrics = compute_metrics(records)
        self.assertAlmostEqual(metrics["timing_seconds"]["generate"]["mean"], 1.0)
        self.assertAlmostEqual(metrics["timing_seconds"]["end_to_end"]["p95"], 2.0)
        # 400 token / 4s = 100 tokens/s（含 prefill）
        self.assertAlmostEqual(metrics["throughput"]["tokens_per_second_generate"], 100.0)
        self.assertAlmostEqual(
            metrics["throughput"]["samples_per_second_generate"], 1.0)


class CompareTests(unittest.TestCase):
    def test_off_on_flip_counting(self):
        off = [rec("u1", "2D", "Count", "ADE20K", True),
               rec("u2", "2D", "Count", "ADE20K", False),
               rec("u3", "2D", "Count", "ADE20K", True)]
        on = [rec("u1", "2D", "Count", "ADE20K", False),   # Off 对 / On 错
              rec("u2", "2D", "Count", "ADE20K", True),    # Off 错 / On 对
              rec("u3", "2D", "Count", "ADE20K", True)]
        comparison = compare_runs(off, on)
        self.assertEqual(comparison["common_samples"], 3)
        self.assertEqual(comparison["off_right_on_wrong"], 1)
        self.assertEqual(comparison["off_wrong_on_right"], 1)

    def test_unmatched_samples_reported(self):
        off = [rec("u1", "2D", "Count", "ADE20K", True),
               rec("u2", "2D", "Count", "ADE20K", True)]
        on = [rec("u1", "2D", "Count", "ADE20K", True)]
        comparison = compare_runs(off, on)
        self.assertEqual(comparison["common_samples"], 1)
        self.assertEqual(comparison["only_in_off"], 1)
        self.assertEqual(comparison["only_in_on"], 0)


class ResumeToolTests(unittest.TestCase):
    """验证断点续跑用到的两个纯函数：选择样本、修复半行 JSONL。"""

    @classmethod
    def setUpClass(cls):
        from run_qwen35_cvbench import _read_jsonl, _select_records
        cls.read_jsonl = staticmethod(_read_jsonl)
        cls.select = staticmethod(_select_records)
        cls.records = load_records(DATASET_DIR, "all")

    def test_limit_is_prefix(self):
        picked = self.select(self.records, 5, False)
        self.assertEqual([r.idx for r in picked], sorted(r.idx for r in self.records)[:5])

    def test_stratified_covers_all_groups(self):
        picked = self.select(self.records, 6, True)
        groups = {(r.type, r.task, r.source) for r in picked}
        self.assertEqual(len(picked), 6)
        self.assertEqual(len(groups), 6)

    def test_partial_line_is_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                json.dumps({"uid": "a"}) + "\n"
                + json.dumps({"uid": "b"}) + "\n"
                + '{"uid": "c", "trunc',            # 模拟崩溃留下的半行
                encoding="utf-8",
            )
            records = self.read_jsonl(path)
            self.assertEqual([r["uid"] for r in records], ["a", "b"])
            # 半行已被截掉，下次续跑不会重复解析失败。
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
