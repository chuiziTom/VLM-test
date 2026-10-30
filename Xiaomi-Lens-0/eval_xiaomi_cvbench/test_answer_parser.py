"""Unit tests for the strict CV-Bench answer parser.

Run with::

    micromamba run -n xiaomi-lens python eval_xiaomi_cvbench/test_answer_parser.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_parser import parse_final_answer  # noqa: E402


def parse(text, letters=("A", "B", "C", "D"), *, think=False, truncated=False):
    return parse_final_answer(
        text, tuple(letters),
        requires_think_close=think, truncated=truncated,
    )


class ParserTests(unittest.TestCase):
    # 1. Standard format.
    def test_standard_format(self):
        result = parse("FINAL_ANSWER: (B)")
        self.assertEqual(result.prediction, "B")
        self.assertTrue(result.valid)

    # 2. Thinking mentions several options; only the final answer counts.
    def test_thinking_mentions_options_final_is_c(self):
        text = (
            "<think>\nOption A is unlikely.\nOption B may work.\n"
            "Option C matches the image.\n</think>\n\nFINAL_ANSWER: (C)"
        )
        self.assertEqual(parse(text, think=True).prediction, "C")

    # 3. Six-way question, final answer F.
    def test_six_choices_final_f(self):
        text = "</think>\nFINAL_ANSWER: (F)"
        result = parse(text, letters=tuple("ABCDEF"), think=True)
        self.assertEqual(result.prediction, "F")

    # 4. Conflicting final answers are invalid.
    def test_conflicting_answers(self):
        text = "</think>\nFINAL_ANSWER: (B)\nFINAL_ANSWER: (D)"
        result = parse(text, think=True)
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "conflicting_final_answers")

    # 5. No final answer marker.
    def test_no_final_answer(self):
        text = "</think>\nThe answer is probably B."
        result = parse(text, think=True)
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "no_final_answer")

    # 6. Only thinking content, generation ended.
    def test_only_thinking_not_closed(self):
        result = parse("<think>\nConsidering A and B", think=True)
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "missing_think_close")

    # 7. Letter outside the question's option set.
    def test_invalid_option(self):
        result = parse("</think>\nFINAL_ANSWER: (E)", think=True)
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "invalid_option")

    # 8. Truncated before any final answer.
    def test_truncated(self):
        result = parse("<think>\nStill reasoning", think=True, truncated=True)
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "truncated_before_final_answer")

    # 9. Options live in the prompt; the completion contains one marker only.
    def test_completion_only_is_parsed(self):
        completion = "FINAL_ANSWER: (A)"
        result = parse(completion, letters=("A", "B"))
        self.assertEqual(result.prediction, "A")
        self.assertEqual(result.n_matches, 1)

    # 10. A special token after the answer is stripped before matching.
    def test_special_token_after_answer(self):
        result = parse("FINAL_ANSWER: (D)<|im_end|>")
        self.assertEqual(result.prediction, "D")

    # Extra strictness checks.
    def test_lowercase_is_rejected(self):
        result = parse("FINAL_ANSWER: (c)")
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "malformed_final_answer")

    def test_bold_markdown_is_rejected(self):
        result = parse("**FINAL_ANSWER: (C)**")
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "malformed_final_answer")

    def test_marker_inside_thinking_is_ignored(self):
        text = "FINAL_ANSWER: (A)\n</think>\nNo final answer here."
        result = parse(text, think=True)
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "no_final_answer")

    def test_repeated_identical_answer_is_accepted(self):
        text = "</think>\nFINAL_ANSWER: (C)\nFINAL_ANSWER: (C)"
        result = parse(text, think=True)
        self.assertEqual(result.prediction, "C")
        self.assertEqual(result.n_matches, 2)

    def test_empty_answer_after_think(self):
        result = parse("</think>\n", think=True)
        self.assertIsNone(result.prediction)
        self.assertEqual(result.invalid_reason, "empty_final_answer")

    def test_thinking_off_does_not_require_close_tag(self):
        self.assertTrue(parse("FINAL_ANSWER: (B)").valid)

    def test_special_token_on_its_own_line(self):
        result = parse("FINAL_ANSWER: (A)\n<|im_end|>\n")
        self.assertEqual(result.prediction, "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
