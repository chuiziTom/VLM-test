"""SPAR-Bench 回答切分与严格解析。

职责：
1. 把模型原始输出切分成 ``thinking_text`` 与 ``final_answer_text``
   （Thinking On 时只接受 ``</think>`` 之后的内容）；
2. 对最终回答做 Strict 解析：
   * MCA：必须是一个单独的选项字母（允许 ``A`` / ``(A)`` / ``A.`` /
     ``answer: A``，不接受解释性文本、不接受在整段文字里搜索字母）；
   * NA：必须是一个单独的数值，不接受单位或额外文字；
   * VCI：必须完整给出 5 组 ``key:value``（5 个方向各一组），不接受解释；
3. 解析失败返回明确原因，由调用方记 0 分并保留样本，不做删除。

严格解析不使用 Ground Truth，也不从 Thinking 中间过程猜答案。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sparbench_metrics import (
    NA_QUESTION_TYPES,
    MCA_QUESTION_TYPES,
    compute_vci_metric,
    mean_relative_accuracy,
)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# 解码时保留了特殊 token（如 <|im_end|>），匹配前统一去掉。
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>\n]*\|>")

# 严格 MCA：整个最终回答只能是一个选项字母。
# 只接受三种完整形式：``A`` / ``A.`` / ``(A)``；不接受括号不配对。
_OPTION_BARE_RE = re.compile(r"^([A-Da-d])[.)]?$")
_OPTION_PAREN_RE = re.compile(r"^\(([A-Da-d])\)[.)]?$")
_OPTION_LABELED_RE = re.compile(
    r"^(?:final\s+answer|answer)\s*[:：]\s*(?:\(([A-Da-d])\)|([A-Da-d])[.)]?)$",
    re.IGNORECASE,
)

# 严格 NA：整个最终回答只能是一个数值（允许末尾句号）。
_NUMBER_ONLY_RE = re.compile(r"^[ \t]*([-+]?\d+(?:\.\d+)?)[ \t]*\.?[ \t]*$")

# 严格 VCI：逗号分隔的 5 组 key:value，五个方向必须各出现一次。
_VCI_PAIR_RE = re.compile(
    r"^(move_right|move_left|move_up|move_down|move_forward|move_backward|"
    r"rotate_right|rotate_left|rotate_up|rotate_down)"
    r"\s*:\s*([-+]?\d+(?:\.\d+)?)$"
)
_VCI_AXIS = {
    "move_right": "lr", "move_left": "lr",
    "move_up": "ud", "move_down": "ud",
    "move_forward": "fb", "move_backward": "fb",
    "rotate_right": "rlr", "rotate_left": "rlr",
    "rotate_up": "rud", "rotate_down": "rud",
}
_VCI_REQUIRED_AXES = {"lr", "ud", "fb", "rlr", "rud"}


@dataclass
class StrictParse:
    prediction: object | None      # 解析出的答案；失败为 None
    score: float                   # 严格评分，失败记 0
    valid: bool
    invalid_reason: str | None


def strip_special_tokens(text: str) -> str:
    return _SPECIAL_TOKEN_RE.sub("", text)


def split_final_answer(
    raw_generation: str,
    enable_thinking: bool,
) -> tuple[str, str, bool]:
    """切分 Thinking 与最终回答。

    Returns
    -------
    (thinking_text, final_answer_text, closed_thinking)
        Thinking Off 时 ``closed_thinking=True``（不适用），最终回答为全文；
        Thinking On 时以第一个 ``</think>`` 为界，未出现则最终回答为空串。
    """
    text = strip_special_tokens(raw_generation)
    if not enable_thinking:
        return "", text.strip(), True
    head, sep, tail = text.partition(THINK_CLOSE)
    if not sep:
        return head.strip(), "", False
    return head.strip(), tail.strip(), True


def _parse_mca_strict(text: str) -> tuple[str | None, str | None]:
    """严格解析选择题；返回 (字母, 失败原因)。"""
    match = _OPTION_BARE_RE.match(text) or _OPTION_PAREN_RE.match(text)
    if match is not None:
        return match.group(1).upper(), None
    match = _OPTION_LABELED_RE.match(text)
    if match is None:
        return None, "mca_format_mismatch"
    letter = match.group(1) or match.group(2)
    return letter.upper(), None


def _parse_na_strict(text: str) -> tuple[float | None, str | None]:
    """严格解析数值题；返回 (数值, 失败原因)。"""
    match = _NUMBER_ONLY_RE.match(text)
    if match is None:
        return None, "na_format_mismatch"
    return float(match.group(1)), None


def _parse_vci_strict(text: str, answer: str) -> tuple[float | None, str | None]:
    """严格解析 VCI；返回 (VCI 分数, 失败原因)。"""
    segments = [segment.strip() for segment in text.split(",")]
    if len(segments) != 5:
        return None, "vci_field_count_mismatch"

    axis_seen: dict[str, float] = {}
    pairs: list[tuple[str, float]] = []
    for segment in segments:
        match = _VCI_PAIR_RE.match(segment)
        if match is None:
            return None, "vci_format_mismatch"
        key, value = match.group(1), float(match.group(2))
        axis = _VCI_AXIS[key]
        if axis in axis_seen:
            return None, "vci_duplicate_axis"
        axis_seen[axis] = value
        pairs.append((key, value))

    if set(axis_seen) != _VCI_REQUIRED_AXES:
        return None, "vci_missing_axis"

    # 重新拼成规范字符串，复用官方 VCI 计算（只做数值比较，不改变语义）。
    canonical = ",".join(f"{key}:{value:g}" for key, value in pairs)
    try:
        return compute_vci_metric(canonical, answer), None
    except Exception:
        return None, "vci_parse_failed"


def parse_strict(
    task: str,
    final_answer_text: str,
    ground_truth: str,
    *,
    enable_thinking: bool = False,
    closed_thinking: bool = True,
    truncated: bool = False,
) -> StrictParse:
    """按任务类型做严格解析并评分。

    参数说明
    --------
    final_answer_text:
        ``split_final_answer`` 给出的最终回答区域（Thinking On 时已去掉 Thinking）。
    enable_thinking / closed_thinking:
        用于区分“Thinking 没结束”和“回答格式错误”。
    truncated:
        生成是否因为 ``max_new_tokens`` 被截断，用于给出更准确的原因。
    """
    if enable_thinking and not closed_thinking:
        reason = "truncated_before_final_answer" if truncated else "missing_think_close"
        return StrictParse(None, 0.0, False, reason)

    text = (final_answer_text or "").strip()
    if not text:
        reason = "truncated_no_final_answer" if truncated else "empty_final_answer"
        return StrictParse(None, 0.0, False, reason)

    if task in MCA_QUESTION_TYPES:
        letter, reason = _parse_mca_strict(text)
        if letter is None:
            return StrictParse(None, 0.0, False,
                               "truncated_no_final_answer" if truncated else reason)
        score = 1.0 if letter == ground_truth.strip().upper() else 0.0
        return StrictParse(letter, score, True, None)

    if task in NA_QUESTION_TYPES:
        value, reason = _parse_na_strict(text)
        if value is None:
            return StrictParse(None, 0.0, False,
                               "truncated_no_final_answer" if truncated else reason)
        try:
            target = float(ground_truth)
        except (TypeError, ValueError):
            return StrictParse(value, 0.0, False, "invalid_ground_truth")
        score = mean_relative_accuracy(value, target, start=.5, end=.95, interval=.05)
        return StrictParse(value, score, True, None)

    if task == "view_change_infer":
        score, reason = _parse_vci_strict(text, ground_truth)
        if score is None:
            return StrictParse(None, 0.0, False,
                               "truncated_no_final_answer" if truncated else reason)
        return StrictParse(text, score, True, None)

    raise ValueError(f"未知 SPAR-Bench 任务：{task}")
