"""Prompt templates for Batch 2 generation (no live API calls)."""
from __future__ import annotations

from src.batch2_text import QUESTION_TYPES

SYSTEM_INSTRUCTION = (
    "You generate Bangla QA dataset candidates for retrieval research. "
    "Return strict JSON only. Never fabricate facts. Use only the provided passage/text."
)

PART_A_PROMPT = """Create one answerable Bangla question from the passage.

Rules:
- question_type must be one of: {question_types}
- The gold_answer MUST be an exact contiguous substring copied from the passage text.
- Keep gold_answer short (name/date/number/phrase) and factual.
- Do not write questions that copy a long 4+ word phrase around the answer except necessary function words; prefer natural wording.
- Do not create time-sensitive questions like বর্তমান/এখন without an explicit year in the passage and question.
- Avoid extremely long transliterated foreign names when a simpler question is possible.
- If the passage cannot support a high-quality question, return question="" and gold_answer="".
- Output JSON with keys: question, gold_answer, question_type, reasoning_brief.

passage_id: {passage_id}
title: {title}
passage:
{passage}
"""

PART_B_PROMPT = """Create one naturally unanswerable Bangla question that is plausible given the hint, but cannot be answered from the full Bengali Wikipedia-like corpus.

Rules:
- question_type must be one of: {question_types}
- The question should be specific and realistic, not nonsense.
- It must NOT be answerable from the provided hint passage.
- Avoid private/subjective claims. Prefer missing factual detail, wrong-entity premise, or unavailable specific fact.
- Return JSON with keys: question, question_type, why_unanswerable, near_miss_fact.

passage_id: {passage_id}
title: {title}
hint_passage:
{passage}
"""

PART_C_REWRITE_PROMPT = """Given an existing Bangla question and evidence removal decision, produce metadata for a removed-evidence unanswerable item.

Rules:
- Preserve meaning of the original question; only minimally edit if required for clarity.
- question_type must be one of: {question_types}
- Return JSON with keys: question, question_type, removal_reason, remnant_risk.

original_question: {question}
original_answer: {answer}
question_type: {question_type}
"""


def format_part_a(passage_id: str, title: str, passage: str) -> str:
    return PART_A_PROMPT.format(
        question_types=", ".join(QUESTION_TYPES),
        passage_id=passage_id,
        title=title or "",
        passage=passage,
    )


def format_part_b(passage_id: str, title: str, passage: str) -> str:
    return PART_B_PROMPT.format(
        question_types=", ".join(QUESTION_TYPES),
        passage_id=passage_id,
        title=title or "",
        passage=passage,
    )


def format_part_c(question: str, answer: str, question_type: str) -> str:
    return PART_C_REWRITE_PROMPT.format(
        question_types=", ".join(QUESTION_TYPES),
        question=question,
        answer=answer,
        question_type=question_type,
    )
