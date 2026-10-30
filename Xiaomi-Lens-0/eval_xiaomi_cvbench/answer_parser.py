"""Strict extraction of CV-Bench final answers from model generations.

The model is asked to end its reply with a line of the exact form::

    FINAL_ANSWER: (C)

Only that line, and only in the *final answer* region (after ``</think>`` when
the model is reasoning), is accepted.  We never search the whole completion for
a stray ``A/B/C`` and never look at the ground truth.  Anything that cannot be
parsed unambiguously is reported as invalid and still counts as incorrect in
the accuracy denominator.

Failure reasons (``invalid_reason``):
    missing_think_close        Thinking On but no ``</think>`` in the completion.
    truncated_before_final_answer
                               Same, and generation hit the token limit.
    truncated_no_final_answer  Answer region has no marker and hit the limit.
    malformed_final_answer     ``FINAL_ANSWER`` appears but not in the exact form.
    empty_final_answer         Answer region is empty.
    no_final_answer            Answer region has text but no marker.
    conflicting_final_answers  Two or more different option letters.
    invalid_option             Letter is not in the question's option set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# <|im_end|>, <|endoftext|>, <|mask|>, ... are decoded because we keep special
# tokens in the raw text; strip them before matching a line.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>\n]*\|>")
# Exact final-answer line.  Leading/trailing whitespace and spaces after the
# colon are tolerated; anything else (bold, punctuation, lowercase) is not.
_FINAL_LINE_RE = re.compile(r"^[ \t]*FINAL_ANSWER:[ \t]*\(([A-Z])\)[ \t]*$")
_LOOSE_MARKER_RE = re.compile(r"FINAL_ANSWER")


@dataclass
class ParseResult:
    prediction: str | None
    thinking_text: str
    final_answer_text: str
    invalid_reason: str | None
    n_matches: int
    closed_thinking: bool

    @property
    def valid(self) -> bool:
        return self.prediction is not None


def split_thinking(generated_text: str) -> tuple[str, str, bool]:
    """Split at the first ``</think>``.

    Returns ``(thinking_text, final_answer_text, closed_thinking)``.  When the
    tag is absent, the whole completion is treated as answer text and
    ``closed_thinking`` is False.  This matches ``infer.split_answer_content``.
    """
    head, sep, tail = generated_text.partition(THINK_CLOSE)
    if not sep:
        return "", generated_text, False
    return head, tail, True


def parse_final_answer(
    generated_text: str,
    valid_letters: tuple[str, ...],
    *,
    requires_think_close: bool,
    truncated: bool,
) -> ParseResult:
    """Parse one completion under strict rules.

    Parameters
    ----------
    generated_text:
        Decoded *new* tokens only (never the prompt).
    valid_letters:
        Option letters available for this sample, e.g. ``("A", "B", "C")``.
    requires_think_close:
        True for Thinking On: the completion must contain ``</think>`` before
        an answer can be accepted.
    truncated:
        True when generation stopped because ``max_new_tokens`` was reached.
    """
    thinking, answer, closed = split_thinking(generated_text)

    if requires_think_close and not closed:
        reason = (
            "truncated_before_final_answer"
            if truncated
            else "missing_think_close"
        )
        return ParseResult(None, thinking, answer, reason, 0, closed)

    letters: list[str] = []
    for raw_line in answer.splitlines():
        line = _SPECIAL_TOKEN_RE.sub("", raw_line)
        match = _FINAL_LINE_RE.match(line)
        if match:
            letters.append(match.group(1))

    if not letters:
        if truncated:
            reason = "truncated_no_final_answer"
        elif _LOOSE_MARKER_RE.search(answer):
            reason = "malformed_final_answer"
        elif not answer.strip():
            reason = "empty_final_answer"
        else:
            reason = "no_final_answer"
        return ParseResult(None, thinking, answer, reason, 0, closed)

    distinct = sorted(set(letters))
    if len(distinct) > 1:
        return ParseResult(
            None, thinking, answer, "conflicting_final_answers",
            len(letters), closed,
        )

    letter = distinct[0]
    if letter not in valid_letters:
        return ParseResult(
            None, thinking, answer, "invalid_option", len(letters), closed,
        )
    return ParseResult(letter, thinking, answer, None, len(letters), closed)
