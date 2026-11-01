"""Direct Answer 严格解析器的单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.direct_parser import parse_direct_answer

LETTERS3 = ("A", "B", "C")


class DirectParserTest(unittest.TestCase):
    def check(self, text, expected, reason=None, truncated=False,
              letters=LETTERS3):
        result = parse_direct_answer(text, letters, truncated=truncated)
        self.assertEqual(result.prediction, expected, msg=repr(text))
        if reason is not None:
            self.assertEqual(result.invalid_reason, reason, msg=repr(text))
        return result

    def test_exact_line(self):
        self.check("FINAL_ANSWER: (B)", "B")

    def test_trailing_special_tokens(self):
        self.check("FINAL_ANSWER: (C)<|im_end|>\n<|endoftext|>", "C")

    def test_surrounding_whitespace(self):
        self.check("  \n FINAL_ANSWER: (A)  \n ", "A")

    def test_space_after_colon(self):
        self.check("FINAL_ANSWER:   (A)", "A")

    def test_lowercase_is_malformed(self):
        self.check("final_answer: (B)", None, "extra_text")

    def test_bold_is_extra_text(self):
        self.check("**FINAL_ANSWER: (B)**", None, "extra_text")

    def test_extra_analysis_rejected(self):
        self.check(
            "The image shows two beds.\nFINAL_ANSWER: (B)",
            None, "extra_text",
        )

    def test_answer_only_letter_rejected(self):
        self.check("B", None, "extra_text")

    def test_multiple_answers_rejected(self):
        self.check("FINAL_ANSWER: (A)\nFINAL_ANSWER: (B)",
                   None, "multiple_final_answers")

    def test_same_answer_twice_is_still_multiple(self):
        self.check("FINAL_ANSWER: (A)\nFINAL_ANSWER: (A)",
                   None, "multiple_final_answers")

    def test_invalid_option(self):
        self.check("FINAL_ANSWER: (D)", None, "invalid_option")

    def test_empty_output(self):
        self.check("<|im_end|>", None, "empty_output")

    def test_truncated_empty(self):
        self.check("", None, "truncated_empty_output", truncated=True)

    def test_truncated_extra_text(self):
        self.check("FINAL_ANSWER: (A) because", None,
                   "truncated_extra_text", truncated=True)

    def test_no_marker(self):
        self.check("The answer is B.", None, "extra_text")

    def test_thinking_tag_is_extra_text(self):
        self.check("</think>\nFINAL_ANSWER: (A)", None, "extra_text")

    def test_format_violation_flag(self):
        self.assertFalse(self.check("FINAL_ANSWER: (B)", "B").format_violation)
        self.assertTrue(self.check("B", None, "extra_text").format_violation)


if __name__ == "__main__":
    unittest.main()
