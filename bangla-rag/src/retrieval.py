"""Retrieval primitives: BM25 over chunks and parent-level aggregation.

BM25 tokenization (reported in results):
  1. Unicode NFC normalization
  2. lowercase
  3. maximal runs of Unicode letters (L*), combining marks (M*), and digits (N*)

Step 3 keeps Bangla conjuncts and vowel signs intact (a plain ``\\w+`` split
would break them at the virama).

Scoring uses the Okapi BM25 formula with
``idf = ln(1 + (N - df + 0.5) / (df + 0.5))`` and
``score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``.

Chunks are the retrieval units. A parent passage is scored at its best chunk
(``max`` over its chunks) and counted once.
"""

from __future__ import annotations

import math
import pickle
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def tokenize(text: str) -> list[str]:
    """NFC + lowercase + maximal runs of L*/M*/N* Unicode characters."""
    tokens: list[str] = []
    buf: list[str] = []
    for ch in normalize_text(text):
        category = unicodedata.category(ch)
        if category[0] in ("L", "M", "N"):
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


@dataclass
class ParentIndex:
    parent_ids: list[str]
    unit_to_parent: np.ndarray
    order: np.ndarray
    starts: np.ndarray

    @property
    def n_parents(self) -> int:
        return len(self.parent_ids)


def build_parent_index(unit_parent_ids: list[str]) -> ParentIndex:
    parent_ids = sorted(set(unit_parent_ids))
    mapping = {pid: i for i, pid in enumerate(parent_ids)}
    unit_to_parent = np.fromiter(
        (mapping[pid] for pid in unit_parent_ids),
        dtype=np.int32,
        count=len(unit_parent_ids),
    )
    order = np.argsort(unit_to_parent, kind="stable")
    sorted_parents = unit_to_parent[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_parents[1:] != sorted_parents[:-1]]
    )
    return ParentIndex(parent_ids, unit_to_parent, order, starts)


def aggregate_parent_scores(scores: np.ndarray, pidx: ParentIndex) -> np.ndarray:
    """Max over each parent's units; returns scores aligned with pidx.parent_ids."""
    return np.maximum.reduceat(scores[pidx.order], pidx.starts)


def rank_parents(
    parent_scores: np.ndarray,
    pidx: ParentIndex,
    top_k: int | None = None,
    only_positive: bool = True,
) -> list[str]:
    if only_positive:
        candidates = np.flatnonzero(parent_scores > 0)
    else:
        candidates = np.arange(parent_scores.size)
    if candidates.size == 0:
        return []
    # Primary: score descending. Secondary: parent index ascending (equivalent
    # to lexical order because parent_ids is sorted).
    order = np.lexsort((candidates, -parent_scores[candidates]))
    ranked = candidates[order]
    if top_k is not None:
        ranked = ranked[:top_k]
    return [pidx.parent_ids[i] for i in ranked]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(texts)
        self.doc_len = np.zeros(self.n_docs, dtype=np.int32)

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings[term].append((doc_id, tf))

        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        if self.avgdl == 0.0:
            self.avgdl = 1.0

        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for term, plist in postings.items():
            docs = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((c for _, c in plist), dtype=np.int32, count=len(plist))
            self.postings[term] = (docs, tfs)
            df = len(plist)
            self.idf[term] = math.log(
                1.0 + (self.n_docs - df + 0.5) / (df + 0.5)
            )

    def score(
        self, query_tokens: list[str], k1: float | None = None, b: float | None = None
    ) -> np.ndarray:
        k1 = self.k1 if k1 is None else k1
        b = self.b if b is None else b
        scores = np.zeros(self.n_docs, dtype=np.float32)
        seen: set[str] = set()
        for term in query_tokens:
            if term in seen:
                continue
            seen.add(term)
            posting = self.postings.get(term)
            if posting is None:
                continue
            docs, tfs = posting
            dl = self.doc_len[docs]
            denom = tfs + k1 * (1.0 - b + b * dl / self.avgdl)
            scores[docs] += self.idf[term] * tfs * (k1 + 1.0) / denom
        return scores

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj
