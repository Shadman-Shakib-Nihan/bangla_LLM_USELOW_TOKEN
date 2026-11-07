"""Phase 3: BM25 retrieval over chunks, tuned on MIRACL train only.

- Index units: data/processed/corpus_chunks.jsonl (whole passages + chunks).
- Tokenization: NFC -> lowercase -> maximal L*/M*/N* runs.
- Tuning grid: only data/processed/miracl_train_questions.jsonl.
- Final reporting: MIRACL dev and TyDi validation (held-out).
- Parent-level evaluation: a parent counts once, at its best chunk.
- Saves top-100 parent rankings for later dense/hybrid fusion.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/phase3_bm25.py
"""

from __future__ import annotations

import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (  # noqa: E402
    mrr,
    ndcg_at_k,
    recall_at_k,
    recall_fraction_at_k,
)
from src.retrieval import (  # noqa: E402
    BM25Index,
    aggregate_parent_scores,
    build_parent_index,
    rank_parents,
    tokenize,
)

CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus_chunks.jsonl"
CACHE = PROJECT_ROOT / "data" / "cache" / "bm25_index.pkl"
RESULTS = PROJECT_ROOT / "results"
QUESTIONS = {
    "miracl_train": PROJECT_ROOT / "data" / "processed" / "miracl_train_questions.jsonl",
    "miracl_dev": PROJECT_ROOT / "data" / "processed" / "miracl_dev_questions.jsonl",
    "tydi_validation": PROJECT_ROOT / "data" / "processed" / "tydi_validation_questions.jsonl",
}
K_VALUES = (1, 3, 5, 10)
GRID_K1 = (0.3, 0.6, 0.9, 1.2, 1.5, 2.0)
GRID_B = (0.2, 0.35, 0.5, 0.65, 0.8)


def peak_rss_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3), 2)


def load_units(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    parents: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            parents.append(rec["parent_id"])
    return texts, parents


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate(
    index: BM25Index,
    pidx,
    rows: list[dict],
    k1: float,
    b: float,
    keep_top: int | None = None,
) -> tuple[dict, dict[str, list[str]]]:
    totals = {f"recall@{k}": 0.0 for k in K_VALUES}
    totals["recall_frac@10"] = 0.0
    totals["mrr"] = 0.0
    totals["ndcg@10"] = 0.0
    rankings: dict[str, list[str]] = {}

    for row in rows:
        query_tokens = tokenize(row["question"])
        scores = index.score(query_tokens, k1=k1, b=b)
        parent_scores = aggregate_parent_scores(scores, pidx)
        ranking = rank_parents(parent_scores, pidx, only_positive=True)
        gold = set(row["gold_passage_ids"])

        for k in K_VALUES:
            totals[f"recall@{k}"] += recall_at_k(ranking, gold, k)
        totals["recall_frac@10"] += recall_fraction_at_k(ranking, gold, 10)
        totals["mrr"] += mrr(ranking, gold)
        totals["ndcg@10"] += ndcg_at_k(ranking, gold, 10)

        if keep_top is not None:
            rankings[row["question_id"]] = ranking[:keep_top]

    n = len(rows)
    summary = {"n": n}
    for key, value in totals.items():
        summary[key] = value / n if n else 0.0
    return summary, rankings


def main() -> int:
    start = time.time()
    texts, unit_parents = load_units(CORPUS)
    pidx = build_parent_index(unit_parents)
    print(f"[ok] units={len(texts):,} parents={pidx.n_parents:,}")

    if CACHE.exists() and CACHE.stat().st_mtime >= CORPUS.stat().st_mtime:
        print(f"[ok] loading BM25 index from {CACHE.relative_to(PROJECT_ROOT)}")
        index = BM25Index.load(CACHE)
        build_seconds = 0.0
    else:
        t0 = time.time()
        index = BM25Index(texts)
        build_seconds = time.time() - t0
        index.save(CACHE)
        print(f"[ok] built BM25 index in {build_seconds:.1f}s; cached")

    print(f"     docs={index.n_docs:,} vocab={len(index.postings):,} "
          f"avgdl={index.avgdl:.1f} peak_rss={peak_rss_gib()} GiB")

    train = load_questions(QUESTIONS["miracl_train"])
    dev = load_questions(QUESTIONS["miracl_dev"])
    tydi_val = load_questions(QUESTIONS["tydi_validation"])

    grid_rows = []
    best = None
    for k1 in GRID_K1:
        for b in GRID_B:
            t0 = time.time()
            summary, _ = evaluate(index, pidx, train, k1, b)
            summary.update({"k1": k1, "b": b})
            grid_rows.append(summary)
            key = (summary["ndcg@10"], summary["mrr"], summary["recall@10"])
            if best is None or key > best[0]:
                best = (key, k1, b)
            print(f"[tune] k1={k1:.1f} b={b:.2f} "
                  f"ndcg@10={summary['ndcg@10']:.4f} mrr={summary['mrr']:.4f} "
                  f"recall@10={summary['recall@10']:.4f} ({time.time()-t0:.1f}s)")

    _, best_k1, best_b = best
    print(f"[ok] best on MIRACL train: k1={best_k1} b={best_b}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "phase3_bm25_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)

    final_rows = []
    all_rankings = []
    for name, rows in [("miracl_dev", dev), ("tydi_validation", tydi_val)]:
        summary, rankings = evaluate(
            index, pidx, rows, best_k1, best_b, keep_top=100
        )
        summary.update({"dataset": name, "k1": best_k1, "b": best_b})
        final_rows.append(summary)
        for qid, ranking in rankings.items():
            all_rankings.append({
                "question_id": qid, "dataset": name, "ranking": ranking,
            })
        print(f"[eval] {name}: n={summary['n']} "
              f"R@1={summary['recall@1']:.4f} R@3={summary['recall@3']:.4f} "
              f"R@5={summary['recall@5']:.4f} R@10={summary['recall@10']:.4f} "
              f"MRR={summary['mrr']:.4f} nDCG@10={summary['ndcg@10']:.4f}")

    # Also keep train top-100 rankings (best params) for hybrid weight tuning.
    _, train_rankings = evaluate(index, pidx, train, best_k1, best_b, keep_top=100)
    for qid, ranking in train_rankings.items():
        all_rankings.append({
            "question_id": qid, "dataset": "miracl_train", "ranking": ranking,
        })

    fields = ["dataset", "n", *[f"recall@{k}" for k in K_VALUES],
              "recall_frac@10", "mrr", "ndcg@10", "k1", "b"]
    with (RESULTS / "phase3_bm25_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    with (RESULTS / "phase3_bm25_rankings.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rankings:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "units_file": str(CORPUS.relative_to(PROJECT_ROOT)),
        "n_units": len(texts),
        "n_parents": pidx.n_parents,
        "vocab_size": len(index.postings),
        "avgdl": round(index.avgdl, 3),
        "tokenization": "Unicode NFC -> lowercase -> maximal runs of L*/M*/N*",
        "indexed_field": "text only (titles not indexed)",
        "bm25": {
            "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
            "score": "idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))",
            "query_term_frequency": "ignored (deduplicated)",
            "zero_score_docs": "not ranked",
        },
        "selection_metric": "ndcg@10 on MIRACL train (ties: mrr, recall@10)",
        "best": {"k1": best_k1, "b": best_b},
        "grid": {"k1": list(GRID_K1), "b": list(GRID_B)},
        "build_seconds": round(build_seconds, 1),
        "total_seconds": round(time.time() - start, 1),
        "peak_rss_gib": peak_rss_gib(),
        "reporting_sets": ["miracl_dev", "tydi_validation"],
        "tuning_set": "miracl_train",
        "parent_evaluation": "max over chunks; parent counted once",
    }
    (RESULTS / "phase3_bm25_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"\n[ok] peak RSS {peak_rss_gib()} GiB; total {time.time()-start:.1f}s")
    print("[ok] wrote phase3_bm25_{grid,metrics,rankings,config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
