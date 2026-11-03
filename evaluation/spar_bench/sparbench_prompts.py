"""SPAR-Bench Prompt 构建。

两个 Prompt 版本：

``official_v1``
    逐字复现官方 ``sparbench_doc_to_text``：不同任务类型的 post prompt
    与 ``lmms_eval_specific_kwargs.default`` 保持一致。Thinking Off / On
    使用完全相同的文本，唯一变量是 chat template 的 ``enable_thinking``。

``direct_answer_v1``
    在官方文本后追加一句“只输出答案”的任务级指令，用于
    Direct Answer（``max_new_tokens=64``）实验。它是独立版本，不能与
    Thinking Off 当作同一 Prompt 比较。

注意：不能对所有任务统一追加 ``FINAL_ANSWER: (X)``，数值题和 VCI
任务不是选择题；这里按任务类型分别处理。
"""

from __future__ import annotations

import hashlib

from sparbench_metrics import NA_QUESTION_TYPES, answer_type

PROMPT_VERSION_OFFICIAL = "official_v1"
PROMPT_VERSION_DIRECT = "direct_answer_v1"

NA_POST_PROMPT = "Please answer the question using a single word or phrase."
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
POSITION_POST_PROMPT = (
    "The values represent the bounding box coordinates normalized to a "
    "0-1000 scale, with the top-left corner as the origin of the image."
)

DIRECT_INSTRUCTIONS = {
    "MCA": "Output only the option letter (A, B, C, or D). Do not explain.",
    "NA": "Output only the number. Do not include units or explanation.",
    "VCI": (
        "Output only the five comma-separated parameters in the required format. "
        "Do not explain."
    ),
}


def build_official_prompt(task: str, question: str) -> str:
    """复现官方 ``sparbench_doc_to_text`` 的文本（``pre_prompt=""``）。"""
    if task in NA_QUESTION_TYPES:
        return "\n" + question + "\n" + NA_POST_PROMPT

    kind = answer_type(task)
    if kind == "MCA":
        post = POSITION_POST_PROMPT if task in (
            "position_matching", "camera_motion_infer"
        ) else ""
        return "\n" + question + "\n" + post + "\n" + MCA_POST_PROMPT

    # view_change_infer：官方两个 post prompt 都为空，结果是 question + "\n\n"。
    return "\n" + question + "\n" + "\n"


def build_prompt(task: str, question: str, prompt_version: str) -> str:
    """按版本构建最终 user instruction。"""
    official = build_official_prompt(task, question)
    if prompt_version == PROMPT_VERSION_OFFICIAL:
        return official
    if prompt_version == PROMPT_VERSION_DIRECT:
        kind = answer_type(task)
        return official + "\n\n" + DIRECT_INSTRUCTIONS[kind]
    raise ValueError(f"未知 Prompt 版本：{prompt_version}")


def prompt_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
