"""Direct Answer 模式的严格答案解析。

与历史评测的 ``answer_parser.py`` 不同，本实验要求模型**只输出一行**：

    FINAL_ANSWER: (X)

因此这里采用整段严格匹配，而不是逐行扫描：只要出现任何额外分析文字、
多个答案、或格式不完全一致，都记为 ``invalid``（仍然计入分母）。

只做两件“无害”的清洗：
1. 去掉 ``<|im_end|>`` / ``<|endoftext|>`` / ``<|mask|>`` 这类结构性特殊 Token；
2. 去掉首尾空白。

不做：搜索全文中的零散字母、模糊匹配选项文字、根据 Ground Truth 反推。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# <|im_end|>, <|endoftext|>, <|mask|> ... 这些是解码时保留下来的结构性 Token。
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>\n]*\|>")
# 整段必须恰好是这一行；允许冒号后空格、行首行尾空白。
_STRICT_RE = re.compile(r"FINAL_ANSWER:[ \t]*\(([A-F])\)")
_MARKER = "FINAL_ANSWER"

# 格式类无效原因：用于单独统计“格式违规率”。
FORMAT_VIOLATION_REASONS = frozenset({
    "extra_text",
    "malformed_final_answer",
    "multiple_final_answers",
    "invalid_option",
    "empty_output",
})


@dataclass
class DirectParseResult:
    prediction: str | None
    normalized_text: str
    invalid_reason: str | None
    n_matches: int
    format_violation: bool

    @property
    def valid(self) -> bool:
        return self.prediction is not None


def _clean(raw: str) -> str:
    """只剥离特殊 Token 与首尾空白，不删除普通文字。"""
    return _SPECIAL_TOKEN_RE.sub("", raw or "").strip()


def parse_direct_answer(
    generated_text: str,
    valid_letters: tuple[str, ...],
    *,
    truncated: bool,
) -> DirectParseResult:
    """解析 Direct Answer 输出。

    参数
    ----
    generated_text:
        只包含模型新生成 Token 的解码文本（不含 prompt）。
    valid_letters:
        本题真实存在的选项字母，例如 ``("A", "B", "C")``。
    truncated:
        是否因为达到 ``max_new_tokens`` 而停止。
    """
    text = _clean(generated_text)

    if not text:
        reason = "truncated_empty_output" if truncated else "empty_output"
        return DirectParseResult(None, text, reason, 0, not truncated)

    matches = _STRICT_RE.findall(text)

    if len(matches) > 1:
        return DirectParseResult(
            None, text, "multiple_final_answers", len(matches), True
        )

    if len(matches) == 1:
        # 整段必须就是这一行（允许首尾空白），否则视为额外文本。
        whole = text.replace("\n", " ").strip()
        if not re.fullmatch(r"FINAL_ANSWER:[ \t]*\([A-F]\)", whole):
            reason = "truncated_extra_text" if truncated else "extra_text"
            return DirectParseResult(None, text, reason, 1, True)

        letter = matches[0]
        if letter not in valid_letters:
            return DirectParseResult(None, text, "invalid_option", 1, True)
        return DirectParseResult(letter, text, None, 1, False)

    if _MARKER in text:
        reason = "truncated_malformed" if truncated else "malformed_final_answer"
        return DirectParseResult(None, text, reason, 0, True)

    reason = "truncated_no_final_answer" if truncated else "extra_text"
    return DirectParseResult(None, text, reason, 0, True)
