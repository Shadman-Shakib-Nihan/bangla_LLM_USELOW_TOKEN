"""Context compression methods for Phase 4.

All methods operate on the same retrieved parent passages. They never generate
or paraphrase text: M2/M3/M4/M5 return sentences copied verbatim from the
passages, restored to their original document and sentence order.

Sentence boundaries: Bangla danda (।), question marks, and newlines. Periods
are intentionally not boundaries, so decimal numbers such as 3.14 stay intact.
Quoted sentences split after the closing quote rather than inside the quote.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

CountTokens = Callable[[str], int]
Truncate = Callable[[str, int], str]

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"
_DIGIT_TABLE = str.maketrans(BANGLA_DIGITS, ASCII_DIGITS)
_WS_RE = re.compile(r"\s+")

OPEN_DOUBLE = "“"
CLOSE_DOUBLE = "”"
OPEN_SINGLE = "‘"
CLOSE_SINGLE = "’"
SENTENCE_END = "।?"


def split_sentences(text: str) -> list[str]:
    """Split on danda/question mark/newline, respecting double/single quotes.

    A sentence ending inside quotes is emitted after the closing quote. Empty
    or whitespace-only pieces are dropped.
    """
    if not text or not text.strip():
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences: list[str] = []
    start = 0
    i = 0
    in_double = False
    in_single = False
    pending = False

    def emit(end_exclusive: int) -> None:
        piece = text[start:end_exclusive].strip()
        if piece:
            sentences.append(piece)

    while i < len(text):
        ch = text[i]
        if ch == "\n":
            emit(i)
            start = i + 1
            pending = False
        elif ch == OPEN_DOUBLE:
            in_double = True
        elif ch == CLOSE_DOUBLE:
            in_double = False
            if pending and not in_double and not in_single:
                emit(i + 1)
                start = i + 1
                pending = False
        elif ch == '"':
            in_double = not in_double
            if pending and not in_double and not in_single:
                emit(i + 1)
                start = i + 1
                pending = False
        elif ch == OPEN_SINGLE:
            in_single = True
        elif ch == CLOSE_SINGLE:
            in_single = False
            if pending and not in_double and not in_single:
                emit(i + 1)
                start = i + 1
                pending = False
        elif ch in SENTENCE_END:
            if in_double or in_single:
                pending = True
            else:
                emit(i + 1)
                start = i + 1
                pending = False
        i += 1
    emit(len(text))
    return sentences


def normalize_for_match(text: str) -> str:
    """NFC + Bangla-to-ASCII digits + lowercase + collapsed whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_DIGIT_TABLE)
    text = text.lower()
    return _WS_RE.sub(" ", text).strip()


def contains_answer(
    context: str, answers: Iterable[str]
) -> tuple[bool, str | None]:
    """Exact normalized substring match; no fuzzy matching.

    Multiple acceptable answers are OR-ed. Empty answers are ignored.
    """
    haystack = normalize_for_match(context)
    if not haystack:
        return False, None
    for answer in answers:
        needle = normalize_for_match(answer)
        if needle and needle in haystack:
            return True, answer
    return False, None


@dataclass(frozen=True)
class CandidateSentence:
    passage_rank: int
    sentence_index: int
    order: int
    text: str


@dataclass
class SentenceCompression:
    context: str
    selected: list[CandidateSentence]
    truncated: bool
    fallback: str | None = None


def build_candidates(passages: Sequence[str]) -> list[CandidateSentence]:
    candidates: list[CandidateSentence] = []
    order = 0
    for rank, passage in enumerate(passages):
        for sent_index, sentence in enumerate(split_sentences(passage)):
            candidates.append(
                CandidateSentence(rank, sent_index, order, sentence)
            )
            order += 1
    return candidates


def join_selected(selected: Sequence[CandidateSentence]) -> str:
    """Restore original passage/sentence order; text is unchanged."""
    if not selected:
        return ""
    ordered = sorted(selected, key=lambda c: c.order)
    pieces: list[str] = []
    current_rank: int | None = None
    current: list[str] = []
    for cand in ordered:
        if current_rank is None:
            current_rank = cand.passage_rank
        if cand.passage_rank != current_rank:
            pieces.append(" ".join(current))
            current_rank = cand.passage_rank
            current = []
        current.append(cand.text)
    if current:
        pieces.append(" ".join(current))
    return "\n\n".join(pieces)


def _select_by_scores(
    candidates: Sequence[CandidateSentence],
    scores: Sequence[float],
    budget: int,
    count_tokens: CountTokens,
    truncate: Truncate,
) -> SentenceCompression:
    if not candidates:
        return SentenceCompression("", [], False, "empty")
    ranked = sorted(
        range(len(candidates)),
        key=lambda i: (-scores[i], candidates[i].order),
    )
    selected: list[CandidateSentence] = []
    used = 0
    for i in ranked:
        n = count_tokens(candidates[i].text)
        if used + n <= budget:
            selected.append(candidates[i])
            used += n

    if not selected:
        top = candidates[ranked[0]]
        return SentenceCompression(
            truncate(top.text, budget), [top], True, "single_sentence_truncated"
        )

    score_by_order = {c.order: scores[i] for i, c in enumerate(candidates)}
    while selected and count_tokens(join_selected(selected)) > budget:
        worst = min(
            selected,
            key=lambda c: (score_by_order[c.order], -c.order),
        )
        selected.remove(worst)

    if not selected:
        top = candidates[ranked[0]]
        return SentenceCompression(
            truncate(top.text, budget), [top], True, "separator_budget_fallback"
        )
    return SentenceCompression(join_selected(selected), selected, False, None)


def m0_top_k(passages: Sequence[str], k: int) -> str:
    """No compression: first k passages in retrieval order."""
    return "\n\n".join(passages[:k])


def m1_hard_budget(
    passages: Sequence[str],
    budget: int,
    count_tokens: CountTokens,
    truncate: Truncate,
) -> str:
    """Concatenate in retrieval order, then truncate at ``budget`` tokens."""
    context = "\n\n".join(passages)
    if count_tokens(context) <= budget:
        return context
    return truncate(context, budget)


def m2_e5(
    candidates: Sequence[CandidateSentence],
    sentence_scores: Sequence[float],
    budget: int,
    count_tokens: CountTokens,
    truncate: Truncate,
) -> SentenceCompression:
    return _select_by_scores(
        candidates, sentence_scores, budget, count_tokens, truncate
    )


def m3_bm25(
    candidates: Sequence[CandidateSentence],
    sentence_scores: Sequence[float],
    budget: int,
    count_tokens: CountTokens,
    truncate: Truncate,
) -> SentenceCompression:
    return _select_by_scores(
        candidates, sentence_scores, budget, count_tokens, truncate
    )


def m4_oracle(
    candidates: Sequence[CandidateSentence],
    answers: Iterable[str],
) -> SentenceCompression:
    """TyDi oracle lower bound: minimum unique answer-bearing sentences.

    Identical sentences are deduplicated after NFC normalization and digit
    mapping, keeping the first occurrence's original text and order. Distinct
    answer-bearing sentences are all kept, because more than one sentence can
    be needed to cover a question.
    """
    answers = list(answers)
    selected: list[CandidateSentence] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not contains_answer(candidate.text, answers)[0]:
            continue
        key = normalize_for_match(candidate.text)
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
    fallback = "oracle_unique_answer_sentences" if selected else "oracle_no_match"
    return SentenceCompression(
        join_selected(selected), selected, False, fallback
    )


def m5_random(
    candidates: Sequence[CandidateSentence],
    budget: int,
    count_tokens: CountTokens,
    truncate: Truncate,
    rng: random.Random,
) -> SentenceCompression:
    """Random sentence baseline; preserve original order after selection."""
    if not candidates:
        return SentenceCompression("", [], False, "empty")
    order = list(range(len(candidates)))
    rng.shuffle(order)
    selected: list[CandidateSentence] = []
    used = 0
    for i in order:
        n = count_tokens(candidates[i].text)
        if used + n <= budget:
            selected.append(candidates[i])
            used += n
    if not selected:
        top = candidates[order[0]]
        return SentenceCompression(
            truncate(top.text, budget), [top], True, "single_sentence_truncated"
        )
    while selected and count_tokens(join_selected(selected)) > budget:
        selected.remove(selected[-1])
    if not selected:
        top = candidates[order[0]]
        return SentenceCompression(
            truncate(top.text, budget), [top], True, "separator_budget_fallback"
        )
    return SentenceCompression(join_selected(selected), selected, False, None)
