"""Retrieval metrics: Recall@k (hit rate), MRR, nDCG@k.

Conventions used throughout the project:
  * Rankings are lists of parent-passage IDs, best first, already deduplicated.
  * ``gold`` is a set of relevant parent-passage IDs.
  * Recall@k is the retrieval hit rate: 1.0 if any gold ID is in the top k.
  * nDCG@k uses binary relevance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def dedupe(ranking: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc in ranking:
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def recall_at_k(ranking: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    return 1.0 if gold.intersection(ranking[:k]) else 0.0


def recall_fraction_at_k(ranking: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        return 0.0
    return len(gold.intersection(ranking[:k])) / len(gold)


def mrr(ranking: Sequence[str], gold: Iterable[str]) -> float:
    gold = set(gold)
    for rank, doc in enumerate(ranking, start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    dcg = 0.0
    for rank, doc in enumerate(ranking[:k], start=1):
        if doc in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    gold: Mapping[str, Iterable[str]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    qids = sorted(set(rankings) & set(gold))
    if not qids:
        raise ValueError("no overlapping query IDs between rankings and gold")

    rows = []
    for qid in qids:
        ranking = dedupe(rankings[qid])
        g = set(gold[qid])
        row = {"qid": qid}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(ranking, g, k)
        row["recall_frac@10"] = recall_fraction_at_k(ranking, g, 10)
        row["mrr"] = mrr(ranking, g)
        row["ndcg@10"] = ndcg_at_k(ranking, g, 10)
        rows.append(row)

    summary = {"n": len(qids)}
    for key in rows[0]:
        if key == "qid":
            continue
        summary[key] = sum(r[key] for r in rows) / len(rows)
    return summary
