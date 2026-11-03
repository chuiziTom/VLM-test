"""SPAR-Bench 评分 / 解析测试。

核心是 **与官方实现逐函数对拍**：把官方 ``utils.py`` 作为参考模块加载，
在相同输入上比较 ``exact_match`` / ``process_na`` / ``mean_relative_accuracy``
/ ``compute_vci_metric`` 以及 Prompt 构建结果。

运行：
    micromamba run -n vlm-test python -m unittest discover \
        -s evaluation/spar_bench/tests -v
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPAR_DIR = HERE.parent
sys.path.insert(0, str(SPAR_DIR))


def _load_official():
    """加载 vendored 官方模块；仅为缺失的 loguru 注入空壳。"""
    fake_loguru = types.ModuleType("loguru")
    fake_loguru.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    sys.modules.setdefault("loguru", fake_loguru)
    path = SPAR_DIR / "reference" / "sparbench_tiny_utils_official.py"
    spec = importlib.util.spec_from_file_location("spar_official", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official = _load_official()

import sparbench_metrics as metrics  # noqa: E402
import sparbench_parser as parser  # noqa: E402
from sparbench_prompts import build_official_prompt  # noqa: E402


class TestOfficialParity(unittest.TestCase):
    def test_exact_match_parity(self):
        cases = [
            ("A", "A"), ("a", "A"), ("B.", "B"), ("C ", "C"),
            ("The answer is D", "D"), ("", "A"), ("BC", "B"),
            ("d", "D"), ("AA", "A"), (" A", "A"),
        ]
        for pred, target in cases:
            with self.subTest(pred=pred, target=target):
                self.assertEqual(
                    metrics.exact_match(pred, target),
                    official.exact_match(pred, target),
                )

    def test_process_na_parity(self):
        cases = [
            ("3.5", "depth_prediction_oc"),
            ("The depth is 2.4 meters.", "depth_prediction_oc"),
            ("1.0 then 2.0", "depth_prediction_oc"),
            ("1.0 then 2.0", "depth_prediction_oc_mv"),
            ("no numbers", "depth_prediction_oc_mv"),
            ("-3.2", "distance_prediction_oo"),
            ("in 2024 the value 5", "distance_prediction_oo_mv"),
        ]
        for pred, task in cases:
            with self.subTest(pred=pred, task=task):
                self.assertEqual(
                    metrics.process_na(pred, task),
                    official.process_na(pred, task),
                )

    def test_mra_parity(self):
        rng = random.Random(0)
        cases = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
        cases += [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(200)]
        for pred, target in cases:
            with self.subTest(pred=pred, target=target):
                self.assertAlmostEqual(
                    metrics.mean_relative_accuracy(pred, target, .5, .95, .05),
                    official.mean_relative_accuracy(pred, target, .5, .95, .05),
                    places=12,
                )

    def test_vci_parity(self):
        cases = [
            ("move_right:1.6,move_down:0,move_backward:0,rotate_up:15,rotate_left:75",
             "move_right:1.6,move_down:0,move_backward:0,rotate_up:15,rotate_left:75"),
            ("move_left:0.5,move_up:0.2,move_forward:1.0,rotate_down:10,rotate_right:5",
             "move_right:0.2,move_down:0,move_backward:0.5,rotate_up:20,rotate_left:35"),
            ("move_right:2,move_left:1,move_up:0.5,move_down:0.5,move_forward:1,"
             "move_backward:0,rotate_right:10,rotate_left:5,rotate_up:1,rotate_down:1",
             "move_right:1,move_up:0,move_forward:1,rotate_right:5,rotate_up:0"),
        ]
        for pred, answer in cases:
            with self.subTest(pred=pred):
                self.assertAlmostEqual(
                    metrics.compute_vci_metric(pred, answer),
                    official.compute_vci_metric(pred, answer),
                    places=12,
                )

    def test_vci_exception_parity(self):
        for pred in ["no parameters here", "move_right:1.0", ""]:
            with self.subTest(pred=pred):
                ref_error = None
                try:
                    official.compute_vci_metric(pred, "move_right:1.0")
                except Exception as exc:  # noqa: BLE001
                    ref_error = type(exc)
                our_error = None
                try:
                    metrics.compute_vci_metric(pred, "move_right:1.0")
                except Exception as exc:  # noqa: BLE001
                    our_error = type(exc)
                self.assertEqual(ref_error, our_error)

    def test_prompt_parity(self):
        kwargs = {
            "pre_prompt": "",
            "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            "na_post_prompt": "Please answer the question using a single word or phrase.",
        }
        docs = [
            {"task": "depth_prediction_oc", "question": "Q1?"},
            {"task": "depth_prediction_oc_mv", "question": "Q2?"},
            {"task": "obj_spatial_relation_oo", "question": "Q3?"},
            {"task": "position_matching", "question": "Q4?"},
            {"task": "camera_motion_infer", "question": "Q5?"},
            {"task": "view_change_infer", "question": "Q6?"},
        ]
        for doc in docs:
            with self.subTest(task=doc["task"]):
                self.assertEqual(
                    build_official_prompt(doc["task"], doc["question"]),
                    official.sparbench_doc_to_text(doc, kwargs),
                )


class TestStrictParser(unittest.TestCase):
    def test_mca(self):
        cases = {
            "A": "A", "a": "A", "(B)": "B", "C.": "C", "D)": "D",
            "Answer: A": "A", "final answer: (C)": "C",
        }
        for text, expected in cases.items():
            result = parser.parse_strict("obj_spatial_relation_oo", text, expected)
            self.assertTrue(result.valid, text)
            self.assertEqual(result.prediction, expected)

        invalid = ["The answer is B", "**A**", "A because ...", "", "AB", "E"]
        for text in invalid:
            result = parser.parse_strict("obj_spatial_relation_oo", text, "A")
            self.assertFalse(result.valid, text)
            self.assertEqual(result.score, 0.0)

    def test_na(self):
        result = parser.parse_strict("depth_prediction_oc", "3.5", "3.5")
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.score, 1.0)

        result = parser.parse_strict("depth_prediction_oc", "3.5.", "3.5")
        self.assertTrue(result.valid)

        for text in ["3.5 meters", "The depth is 3.5", "3,5", "~3.5", ""]:
            result = parser.parse_strict("depth_prediction_oc", text, "3.5")
            self.assertFalse(result.valid, text)
            self.assertEqual(result.score, 0.0)

    def test_na_mra_is_graded(self):
        result = parser.parse_strict("depth_prediction_oc", "3.3", "3.5")
        self.assertTrue(result.valid)
        self.assertGreater(result.score, 0.0)
        self.assertLess(result.score, 1.0)

    def test_vci(self):
        gt = "move_right:1.6,move_down:0,move_backward:0,rotate_up:15,rotate_left:75"
        result = parser.parse_strict("view_change_infer", gt, gt)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.score, 1.0)

        # 顺序无关，但仍要求五个方向齐全。
        reordered = "rotate_left:75,rotate_up:15,move_backward:0,move_down:0,move_right:1.6"
        result = parser.parse_strict("view_change_infer", reordered, gt)
        self.assertTrue(result.valid)

        invalid = [
            "move_right:1.6,move_down:0,move_backward:0,rotate_up:15",
            gt + ",extra:1",
            "Sure: " + gt,
            "move_right:1.6,move_right:2,move_backward:0,rotate_up:15,rotate_left:75",
            "move_right:1.6,move_down:0,move_back:0,rotate_up:15,rotate_left:75",
        ]
        for text in invalid:
            result = parser.parse_strict("view_change_infer", text, gt)
            self.assertFalse(result.valid, text)

    def test_thinking_split(self):
        # 实际 decode 出来的是“新生成的 token”，不会带 <|im_start|>assistant 前缀。
        raw = "<think>\nreasoning\n</think>\n\nThe answer is 3.5.<|im_end|>"
        thinking, final, closed = parser.split_final_answer(raw, enable_thinking=True)
        self.assertTrue(closed)
        self.assertEqual(thinking, "<think>\nreasoning")
        self.assertEqual(final, "The answer is 3.5.")

        _, final, closed = parser.split_final_answer(raw, enable_thinking=False)
        self.assertTrue(closed)
        self.assertIn("reasoning", final)

        thinking, final, closed = parser.split_final_answer(
            "<think>unfinished", enable_thinking=True
        )
        self.assertFalse(closed)
        self.assertEqual(final, "")

    def test_truncated_thinking(self):
        result = parser.parse_strict(
            "depth_prediction_oc", "", "3.5",
            enable_thinking=True, closed_thinking=False, truncated=True,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_reason, "truncated_before_final_answer")


class TestOfficialAggregateParity(unittest.TestCase):
    def test_aggregate_matches_official(self):
        """用官方 sparbench_aggregate_results 对拍 20 任务平均口径。"""
        rows = []
        records = []
        for ti, task in enumerate(metrics.ALL_TASKS):
            kind = metrics.answer_type(task)
            key = {"MCA": "accuracy", "NA": "MRA:.5:.95:.05",
                   "VCI": "vci_metric"}[kind]
            values = [(ti % 5) / 4.0, 0.5, 0.25]
            for value in values:
                rows.append({"task": task, key: value})
                records.append({
                    "id": len(records),
                    "task": task,
                    "img_type": "single_view",
                    "official_score": value,
                    "strict_score": value,
                    "invalid": False,
                    "truncated": False,
                    "error": None,
                    "final_answer_text": "A",
                    "ground_truth": "A",
                    "strict_valid": True,
                    "n_images": 1,
                })

        ref_overall = official.sparbench_aggregate_results(rows) / 100.0
        ours = metrics.compute_metrics(
            records, meta={"selected_samples": len(records)}
        )
        self.assertAlmostEqual(ref_overall, ours["overall"]["official"], places=12)

        # 官方 Low / Middle / High 是层级内任务分数的均值。
        for level, tasks in (
            ("low", metrics.LOW_TASKS),
            ("medium", metrics.MIDDLE_TASKS),
            ("high", metrics.HIGH_TASKS),
        ):
            # 每条任务 3 个值的均值，再对层级内任务取均值。
            task_means = [
                ((metrics.ALL_TASKS.index(t) % 5) / 4.0 + 0.5 + 0.25) / 3
                for t in tasks
            ]
            expected = sum(task_means) / len(task_means)
            self.assertAlmostEqual(
                ours["levels"][level]["official_score"], expected, places=12
            )


class TestAggregation(unittest.TestCase):
    def _fake_records(self):
        """20 个任务 × 50 条，分数可控，用于验证官方聚合口径。"""
        records = []
        for ti, task in enumerate(metrics.ALL_TASKS):
            kind = metrics.answer_type(task)
            for i in range(50):
                # 每个任务固定分数 ti/20，便于手算任务均值。
                score = ti / 20.0
                records.append({
                    "id": ti * 50 + i,
                    "task": task,
                    "img_type": "multi_view" if "_mv" in task else "single_view",
                    "official_score": score,
                    "strict_score": score,
                    "invalid": False,
                    "truncated": False,
                    "error": None,
                    "final_answer_text": "A",
                    "ground_truth": "A",
                    "official_correct": None if kind != "MCA" else score >= 1,
                    "strict_valid": True,
                    "n_images": 1,
                })
        return records

    def test_overall_is_mean_of_task_scores(self):
        records = self._fake_records()
        result = metrics.compute_metrics(records, meta={"selected_samples": 1000})
        expected_overall = sum(ti / 20.0 for ti in range(20)) / 20
        self.assertAlmostEqual(result["overall"]["official"], expected_overall)
        self.assertEqual(result["n_tasks"], 20)
        self.assertEqual(result["n_samples"], 1000)
        # 每个任务 50 条、分数相同，池化均值应等于任务均值。
        self.assertAlmostEqual(
            result["overall"]["pooled_official_reference"], expected_overall
        )

    def test_level_aggregation(self):
        records = self._fake_records()
        result = metrics.compute_metrics(records, meta={"selected_samples": 1000})
        low = [metrics.ALL_TASKS.index(t) / 20.0 for t in metrics.LOW_TASKS]
        self.assertAlmostEqual(result["levels"]["low"]["official_score"], sum(low) / len(low))
        self.assertEqual(result["levels"]["low"]["n_samples"], 400)
        self.assertEqual(result["levels"]["medium"]["n_samples"], 150)
        self.assertEqual(result["levels"]["high"]["n_samples"], 450)


if __name__ == "__main__":
    unittest.main()
