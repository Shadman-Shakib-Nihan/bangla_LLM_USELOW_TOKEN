"""Reciprocal Rank Fusion (RRF) for combining retrieval rankings."""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[str]:
    """Weighted RRF.

    ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks. Documents
    missing from a ranking contribute nothing. Ties are broken by document ID so
    the output is deterministic.
    """
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        seen: set[str] = set()
        for rank, doc in enumerate(ranking, start=1):
            if doc in seen:
                continue
            seen.add(doc)
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)

    return [
        doc for doc, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def weighted_rrf(
    first: Sequence[str],
    second: Sequence[str],
    first_weight: float,
    k: int = 60,
) -> list[str]:
    """Convenience wrapper: fuse two rankings with weights (w, 1 - w)."""
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("first_weight must be in [0, 1]")
    return rrf_fuse([first, second], [first_weight, 1.0 - first_weight], k=k)
