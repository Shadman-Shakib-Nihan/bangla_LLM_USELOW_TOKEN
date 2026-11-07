"""Sentence-aware chunking for over-length passages.

Only passages whose e5 token length exceeds 512 are split (see
``results/phase3_over512_ids.json``). Splitting is on Bangla danda (।) and
newlines, with one-sentence overlap, targeting <= 400 e5 tokens per chunk
(including the ``passage: `` prefix and special tokens). A single sentence that
alone exceeds the limit is split by words as a fallback.

Every chunk keeps the original ``parent_id`` so evaluation can count a parent
once, at its best-ranked chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

DANDA = "।"
SENTENCE_SPLIT_RE = re.compile(r"(?<=।)\s*|\n+")
WORD_RE = re.compile(r"\S+")

CountTokens = Callable[[str], int]


def split_sentences(text: str) -> list[str]:
    """Split on Bangla danda (।) and newlines; strip; drop empties."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class ChunkStats:
    sentences: int = 0
    chunks: int = 0
    oversized_sentences: int = 0
    no_overlap_boundaries: int = 0
    post_fix_splits: int = 0


def _split_oversized(
    sentence: str, count_tokens: CountTokens, max_tokens: int
) -> list[str]:
    """Greedy word split for a sentence that alone exceeds max_tokens."""
    words = WORD_RE.findall(sentence)
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))

    # A single word can still exceed the limit (very rare): split by chars.
    safe: list[str] = []
    for piece in pieces:
        if count_tokens(piece) <= max_tokens:
            safe.append(piece)
            continue
        lo, hi = 1, len(piece)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(piece[:mid]) <= max_tokens:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        safe.append(piece[:best])
        safe.extend(_split_oversized(piece[best:], count_tokens, max_tokens))
    return [p for p in safe if p.strip()]


def chunk_text(
    text: str,
    count_tokens: CountTokens,
    max_tokens: int = 400,
    overlap_sentences: int = 1,
) -> tuple[list[str], ChunkStats]:
    """Split ``text`` into sentence-aware chunks with sentence overlap."""
    if overlap_sentences != 1:
        raise ValueError("only one-sentence overlap is implemented")

    sentences = split_sentences(text)
    stats = ChunkStats(sentences=len(sentences))
    if not sentences:
        return ([text] if text.strip() else []), stats

    chunks: list[str] = []
    current: list[str] = []
    i = 0

    while i < len(sentences):
        sentence = sentences[i]

        if not current:
            if count_tokens(sentence) <= max_tokens:
                current = [sentence]
                i += 1
            else:
                chunks.extend(_split_oversized(sentence, count_tokens, max_tokens))
                stats.oversized_sentences += 1
                i += 1
            continue

        if count_tokens(" ".join(current + [sentence])) <= max_tokens:
            current.append(sentence)
            i += 1
            continue

        # Current chunk is full: close it, then start the next with overlap.
        chunks.append(" ".join(current))
        overlap = current[-1]
        if count_tokens(overlap + " " + sentence) <= max_tokens:
            current = [overlap, sentence]
        elif count_tokens(sentence) <= max_tokens:
            current = [sentence]
            stats.no_overlap_boundaries += 1
        else:
            current = []
            chunks.extend(_split_oversized(sentence, count_tokens, max_tokens))
            stats.oversized_sentences += 1
        i += 1

    if current:
        chunks.append(" ".join(current))

    # Guarantee the 400-token ceiling even after boundary effects.
    final: list[str] = []
    for chunk in chunks:
        if count_tokens(chunk) <= max_tokens:
            final.append(chunk)
        else:
            final.extend(_split_oversized(chunk, count_tokens, max_tokens))
            stats.post_fix_splits += 1

    stats.chunks = len(final)
    return final, stats
