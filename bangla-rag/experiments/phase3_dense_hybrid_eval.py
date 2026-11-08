"""Phase 3: dense + hybrid (RRF) retrieval, parent-level evaluation.

Retrieval units are exactly the shared chunks in
data/processed/corpus_chunks.jsonl (the same units BM25 uses). Dense retrieval
uses a FAISS flat inner-product index over L2-normalized multilingual-e5-small
embeddings (``query: `` / ``passage: `` prefixes, mean pooling). Parent scores
are the max over the parent's units; a parent is counted once.

Tuning:
  * BM25 k1/b were tuned earlier on MIRACL train (best: k1=0.9, b=0.35).
  * RRF constant and BM25/dense weights are tuned here on MIRACL train only.
    Selection: nDCG@10, tie-break MRR, then Recall@10.

Reporting: MIRACL dev and TyDi QA validation, strict gold-passage IDs.
TyDi additionally gets an answer-string hit metric: a query is a hit if any
retrieved unit (all chunks of the top-k parents) contains a gold answer string.

Bootstrap 95% CIs resample queries with replacement (1,000 draws, seed 42).

Also:
  * truncated-vs-split dense Recall@10 ablation on queries with an over-length
    gold passage;
  * count of retrieval units under 20 e5 tokens.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/phase3_dense_hybrid_eval.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # noqa: E402

from src.fusion import rrf_fuse  # noqa: E402
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

PROCESSED = PROJECT_ROOT / "data" / "processed"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
RESULTS = PROJECT_ROOT / "results"

UNITS = PROCESSED / "corpus_chunks.jsonl"
BM25_CACHE = PROJECT_ROOT / "data" / "cache" / "bm25_index.pkl"
BM25_CONFIG = RESULTS / "phase3_bm25_config.json"
OVER_QUERIES = PROCESSED / "overlength_gold_queries.jsonl"
OVER_TRUNCATED = PROCESSED / "overlength_parents_truncated.jsonl"
TRUNC_EMB = EMB_DIR / "overlength_truncated.f16.npy"
TRUNC_IDS = EMB_DIR / "overlength_truncated_ids.json"
UNITS_EMB = EMB_DIR / "corpus_units.f16.npy"
PER_QUERY = RESULTS / "phase3_per_query.jsonl"

QUERY_SETS = {
    "miracl_train": PROCESSED / "miracl_train_questions.jsonl",
    "miracl_dev": PROCESSED / "miracl_dev_questions.jsonl",
    "tydi_train": PROCESSED / "tydi_train_questions.jsonl",
    "tydi_validation": PROCESSED / "tydi_validation_questions.jsonl",
}

K_VALUES = (1, 3, 5, 10)
METRICS = [f"recall@{k}" for k in K_VALUES] + ["mrr", "ndcg@10"]
N_BOOT = 1000
BOOT_SEED = 42
RRF_K_GRID = (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100)
RRF_W_GRID = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 21))
FUSE_DEPTH = 100
DENSE_FULL_SEARCH = True
DENSE_SEARCH_K = 2000


def norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def load_units() -> tuple[list[dict], list[str], list[str]]:
    recs: list[dict] = []
    texts: list[str] = []
    parents: list[str] = []
    with UNITS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            recs.append(rec)
            texts.append(rec["text"])
            parents.append(rec["parent_id"])
    return recs, texts, parents


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_query_embeddings(name: str, rows: list[dict]) -> np.ndarray:
    npy = EMB_DIR / f"queries_{name}.f16.npy"
    meta = EMB_DIR / f"queries_{name}.json"
    ids = json.loads(meta.read_text(encoding="utf-8"))["question_ids"]
    expected = [r["question_id"] for r in rows]
    if ids != expected:
        raise RuntimeError(f"query embedding order mismatch for {name}")
    vec = np.load(npy).astype(np.float32)
    if not np.isfinite(vec).all():
        raise RuntimeError(f"non-finite query embeddings for {name}")
    return vec


def per_query(ranking: list[str], gold: set[str]) -> dict:
    row = {}
    for k in K_VALUES:
        row[f"recall@{k}"] = recall_at_k(ranking, gold, k)
    row["mrr"] = mrr(ranking, gold)
    row["ndcg@10"] = ndcg_at_k(ranking, gold, 10)
    return row


def first_relevant_rank(ranking: list[str], gold: set[str]) -> int | None:
    for rank, pid in enumerate(ranking, start=1):
        if pid in gold:
            return rank
    return None


def answer_hit_rank(ranking: list[str], qid: str,
                    p2u: dict[str, list[int]],
                    unit_norm: list[str],
                    answers: dict[str, list[str]]) -> int | None:
    gold = answers.get(qid, [])
    if not gold:
        return None
    for rank, pid in enumerate(ranking, start=1):
        units = p2u.get(pid, [])
        if any(any(a in unit_norm[u] for a in gold) for u in units):
            return rank
    return None


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = BOOT_SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(per_query_rows: list[dict], extra: dict | None = None) -> dict:
    summary = {"n": len(per_query_rows)}
    for metric in METRICS:
        vals = np.array([r[metric] for r in per_query_rows], dtype=float)
        lo, hi = bootstrap_ci(vals)
        summary[metric] = float(vals.mean())
        summary[f"{metric}_ci_low"] = lo
        summary[f"{metric}_ci_high"] = hi
    if extra:
        summary.update(extra)
    return summary


def bm25_rankings(index: BM25Index, pidx, rows: list[dict],
                  k1: float, b: float, top_k: int | None = None,
                  full: bool = False) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        scores = index.score(tokenize(row["question"]), k1=k1, b=b)
        parent_scores = aggregate_parent_scores(scores, pidx)
        out[row["question_id"]] = rank_parents(
            parent_scores, pidx, top_k=None if full else top_k,
            only_positive=True,
        )
    return out


def dense_rankings_fast(index, pidx, query_vec: np.ndarray, rows: list[dict],
                        full: bool = True, top_k: int | None = None
                        ) -> dict[str, list[str]]:
    """FAISS search + max-over-units parent aggregation."""
    n_units = index.ntotal
    k = n_units if full else min(DENSE_SEARCH_K, n_units)
    out: dict[str, list[str]] = {}
    for i, row in enumerate(rows):
        scores, idx = index.search(query_vec[i:i + 1], k)
        scores = scores[0]
        idx = idx[0]
        valid = idx >= 0
        parent_scores = np.full(pidx.n_parents, -np.inf, dtype=np.float32)
        np.maximum.at(parent_scores, pidx.unit_to_parent[idx[valid]], scores[valid])
        out[row["question_id"]] = rank_parents(
            parent_scores, pidx, top_k=None if full else top_k,
            only_positive=False,
        )
    return out


def tune_hybrid(bm25_train: dict[str, list[str]],
                dense_train: dict[str, list[str]],
                train_rows: list[dict]) -> tuple[dict, list[dict]]:
    gold = {r["question_id"]: set(r["gold_passage_ids"]) for r in train_rows}
    qids = [r["question_id"] for r in train_rows]
    grid = []
    best = None
    for rrf_k in RRF_K_GRID:
        for w in RRF_W_GRID:
            f1 = f3 = f5 = f10 = fmrr = fndcg = 0.0
            for qid in qids:
                fused = rrf_fuse(
                    [bm25_train[qid], dense_train[qid]],
                    weights=[w, 1.0 - w],
                    k=rrf_k,
                )
                g = gold[qid]
                f1 += recall_at_k(fused, g, 1)
                f3 += recall_at_k(fused, g, 3)
                f5 += recall_at_k(fused, g, 5)
                f10 += recall_at_k(fused, g, 10)
                fmrr += mrr(fused, g)
                fndcg += ndcg_at_k(fused, g, 10)
            n = len(qids)
            row = {
                "rrf_k": rrf_k, "w_bm25": w, "w_dense": round(1.0 - w, 1),
                "n": n,
                "recall@1": f1 / n, "recall@3": f3 / n, "recall@5": f5 / n,
                "recall@10": f10 / n, "mrr": fmrr / n, "ndcg@10": fndcg / n,
            }
            grid.append(row)
            key = (row["ndcg@10"], row["mrr"], row["recall@10"])
            if best is None or key > best[0]:
                best = (key, row)
    return best[1], grid


def hybrid_rankings(bm25: dict[str, list[str]], dense: dict[str, list[str]],
                    rows: list[dict], rrf_k: int, w: float) -> dict[str, list[str]]:
    out = {}
    for row in rows:
        qid = row["question_id"]
        out[qid] = rrf_fuse(
            [bm25[qid], dense[qid]], weights=[w, 1.0 - w], k=rrf_k
        )
    return out


def parent_to_units(parents: list[str]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(parents):
        mapping[pid].append(i)
    return mapping


def ensure_bm25(texts: list[str]) -> BM25Index:
    if BM25_CACHE.exists() and BM25_CACHE.stat().st_mtime >= UNITS.stat().st_mtime:
        return BM25Index.load(BM25_CACHE)
    index = BM25Index(texts)
    index.save(BM25_CACHE)
    return index


def answer_string_metrics(ranking: list[str], qid: str,
                          p2u: dict[str, list[int]],
                          unit_norm: list[str],
                          answers: dict[str, list[str]]) -> dict:
    gold = answers.get(qid, [])
    hit_rank = None
    if gold:
        for rank, pid in enumerate(ranking, start=1):
            units = p2u.get(pid, [])
            if any(any(a in unit_norm[u] for a in gold) for u in units):
                hit_rank = rank
                break
    row = {}
    for k in K_VALUES:
        row[f"recall@{k}"] = 1.0 if hit_rank is not None and hit_rank <= k else 0.0
    row["mrr"] = 1.0 / hit_rank if hit_rank is not None else 0.0
    row["ndcg@10"] = (
        1.0 / math.log2(hit_rank + 1)
        if hit_rank is not None and hit_rank <= 10 else 0.0
    )
    return row


def count_tokens_for_units(tokenizer, recs: list[dict]) -> np.ndarray:
    lengths = np.zeros(len(recs), dtype=np.int32)
    batch = 512
    for start in range(0, len(recs), batch):
        texts = ["passage: " + r["text"] for r in recs[start:start + batch]]
        enc = tokenizer(texts, add_special_tokens=True, truncation=False,
                        return_length=True)
        lengths[start:start + len(texts)] = np.asarray(enc["length"], dtype=np.int32)
    return lengths


def fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def print_table(title: str, rows: list[dict], systems: list[str]) -> None:
    print(f"\n### {title}")
    header = "| system | n | " + " | ".join(METRICS) + " |"
    print(header)
    print("|" + "---|" * (len(METRICS) + 2))
    for row in rows:
        cells = [fmt(row[m], row[f"{m}_ci_low"], row[f"{m}_ci_high"])
                 for m in METRICS]
        print(f"| {row['system']} | {row['n']} | " + " | ".join(cells) + " |")


def main() -> int:
    start_time = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("[ok] loading units and embeddings")
    recs, texts, parents = load_units()
    pidx = build_parent_index(parents)
    unit_emb = np.load(UNITS_EMB).astype(np.float32)
    if unit_emb.shape[0] != len(texts):
        raise RuntimeError(
            f"embedding rows {unit_emb.shape[0]} != units {len(texts)}"
        )
    if not np.isfinite(unit_emb).all():
        raise RuntimeError("non-finite corpus embeddings")
    dim = unit_emb.shape[1]
    print(f"[ok] units={len(texts):,} parents={pidx.n_parents:,} dim={dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(unit_emb)
    print(f"[ok] FAISS flat IP index: ntotal={index.ntotal:,}")

    bm25 = ensure_bm25(texts)
    best_k1, best_b = 0.9, 0.35
    if BM25_CONFIG.exists():
        cfg = json.loads(BM25_CONFIG.read_text(encoding="utf-8"))
        best_k1 = cfg["best"]["k1"]
        best_b = cfg["best"]["b"]
    print(f"[ok] BM25 k1={best_k1} b={best_b}")

    rows_by_set = {name: load_questions(path) for name, path in QUERY_SETS.items()}
    qvec = {name: load_query_embeddings(name, rows)
            for name, rows in rows_by_set.items()}

    # ---- BM25 and dense rankings -----------------------------------------
    t0 = time.time()
    bm25_full = {}
    dense_full = {}
    for name in ["miracl_train", "miracl_dev", "tydi_validation"]:
        rows = rows_by_set[name]
        bm25_full[name] = bm25_rankings(bm25, pidx, rows, best_k1, best_b, full=True)
        dense_full[name] = dense_rankings_fast(
            index, pidx, qvec[name], rows, full=DENSE_FULL_SEARCH
        )
        print(f"[ok] {name}: rankings done ({time.time()-t0:.1f}s)")

    bm25_train_top = {q: r[:FUSE_DEPTH] for q, r in bm25_full["miracl_train"].items()}
    dense_train_top = {q: r[:FUSE_DEPTH] for q, r in dense_full["miracl_train"].items()}

    # ---- Hybrid tuning on MIRACL train only ------------------------------
    print("[tune] RRF grid on MIRACL train")
    best_hybrid, grid = tune_hybrid(
        bm25_train_top, dense_train_top, rows_by_set["miracl_train"]
    )
    print(f"[ok] best: rrf_k={best_hybrid['rrf_k']} "
          f"w_bm25={best_hybrid['w_bm25']} "
          f"ndcg@10={best_hybrid['ndcg@10']:.4f}")
    with (RESULTS / "phase3_hybrid_tuning.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        writer.writeheader()
        writer.writerows(grid)

    # ---- Final reporting: MIRACL dev, TyDi validation --------------------
    all_per_query = []
    per_query_records = []
    final_rows = []
    tydi_answer_rows = []
    p2u = parent_to_units(parents)
    unit_norm = [norm_text(t) for t in texts]

    short_parents = {
        parents[i] for i, rec in enumerate(recs) if rec.get("is_short")
    }
    over_q = load_questions(OVER_QUERIES)
    over_map = {
        r["question_id"]: set(r["overlength_gold_ids"]) for r in over_q
    }

    answer_answers: dict[str, list[str]] = {}
    for r in rows_by_set["tydi_validation"]:
        answer_answers[r["question_id"]] = [
            norm_text(a) for a in r.get("gold_answers", []) if norm_text(a)
        ]

    for name in ["miracl_dev", "tydi_validation"]:
        rows = rows_by_set[name]
        gold = {r["question_id"]: set(r["gold_passage_ids"]) for r in rows}
        bm25_rank = bm25_full[name]
        dense_rank = dense_full[name]
        hybrid_rank = hybrid_rankings(
            {q: r[:FUSE_DEPTH] for q, r in bm25_rank.items()},
            {q: r[:FUSE_DEPTH] for q, r in dense_rank.items()},
            rows, best_hybrid["rrf_k"], best_hybrid["w_bm25"],
        )

        for system, rankings in [
            ("BM25", bm25_rank),
            ("Dense (e5-small)", dense_rank),
            ("Hybrid (RRF)", hybrid_rank),
        ]:
            pq = []
            for r in rows:
                qid = r["question_id"]
                ranking = rankings[qid]
                g = gold[qid]
                row = {"qid": qid, **per_query(ranking, g)}
                pq.append(row)
                all_per_query.append({
                    "dataset": name, "system": system, **row,
                })
                ans_rank = None
                if name == "tydi_validation":
                    ans_rank = answer_hit_rank(
                        ranking, qid, p2u, unit_norm, answer_answers
                    )
                per_query_records.append({
                    "dataset": name,
                    "system": system,
                    "qid": qid,
                    "question": r["question"],
                    "gold_passage_ids": list(r["gold_passage_ids"]),
                    "overlength_gold_ids": sorted(over_map.get(qid, set())),
                    "has_overlength_gold": bool(over_map.get(qid)),
                    "short_gold_ids": sorted(g & short_parents),
                    "is_short_gold": bool(g & short_parents),
                    "answer_hit_rank": ans_rank,
                    **row,
                    "first_gold_rank": first_relevant_rank(ranking, g),
                    "top10": ranking[:10],
                    "top20": ranking[:20],
                })
            summary = summarize(pq, {"dataset": name, "system": system})
            final_rows.append(summary)

        if name == "tydi_validation":
            for system, rankings in [
                ("BM25", bm25_rank),
                ("Dense (e5-small)", dense_rank),
                ("Hybrid (RRF)", hybrid_rank),
            ]:
                pq = []
                for r in rows:
                    qid = r["question_id"]
                    row = {"qid": qid, **answer_string_metrics(
                        rankings[qid], qid, p2u, unit_norm, answer_answers
                    )}
                    pq.append(row)
                    all_per_query.append({
                        "dataset": name + "_answerstring", "system": system, **row,
                    })
                tydi_answer_rows.append(summarize(
                    pq, {"dataset": name + "_answerstring", "system": system}
                ))

    with PER_QUERY.open("w", encoding="utf-8") as f:
        for rec in per_query_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ok] wrote per-query records: {PER_QUERY.name} "
          f"({len(per_query_records)})")

    # ---- Chunk token statistics ------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "intfloat/multilingual-e5-small"
    )
    lengths = count_tokens_for_units(tokenizer, recs)
    is_chunked = np.array([bool(r.get("is_chunked")) for r in recs])
    chunk_stats = {
        "tokenization": "e5 tokenizer, 'passage: ' prefix + special tokens",
        "total_units": int(len(lengths)),
        "split_chunks": int(is_chunked.sum()),
        "whole_passages": int((~is_chunked).sum()),
        "all_units": {
            "min": int(lengths.min()),
            "p5": float(np.percentile(lengths, 5)),
            "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 1),
            "lt_5": int((lengths < 5).sum()),
            "lt_10": int((lengths < 10).sum()),
            "lt_20": int((lengths < 20).sum()),
            "le_20": int((lengths <= 20).sum()),
        },
        "split_chunks_only": {
            "min": int(lengths[is_chunked].min()) if is_chunked.any() else 0,
            "median": float(np.median(lengths[is_chunked])) if is_chunked.any() else 0,
            "lt_5": int((lengths[is_chunked] < 5).sum()),
            "lt_10": int((lengths[is_chunked] < 10).sum()),
            "lt_20": int((lengths[is_chunked] < 20).sum()),
            "le_20": int((lengths[is_chunked] <= 20).sum()),
        },
    }
    (RESULTS / "phase3_chunk_token_stats.json").write_text(
        json.dumps(chunk_stats, indent=2), encoding="utf-8"
    )

    # ---- Truncated-vs-split dense Recall@10 ablation ---------------------
    print("[ablation] truncated vs split dense R@10")
    over_ids = set(json.loads(
        (EMB_DIR / "overlength_truncated_ids.json").read_text(encoding="utf-8")
    ))
    over_ids_set = set(over_ids)

    base_keep = np.array([p not in over_ids_set for p in parents])
    base_emb = unit_emb[base_keep]
    trunc_emb = np.load(TRUNC_EMB).astype(np.float32)
    trunc_ids = json.loads(TRUNC_IDS.read_text(encoding="utf-8"))
    alt_emb = np.vstack([base_emb, trunc_emb])
    alt_parents = [p for p, keep in zip(parents, base_keep) if keep] + trunc_ids
    alt_pidx = build_parent_index(alt_parents)
    alt_index = faiss.IndexFlatIP(dim)
    alt_index.add(alt_emb)
    print(f"[ok] ablation split units={len(texts):,} "
          f"truncated index units={alt_index.ntotal:,}")

    # Split-condition dense rankings for ablation queries (reuse main where
    # possible; recompute the TyDi train subset).
    ablation_rows = []
    for dataset in ["miracl_train", "miracl_dev", "tydi_train", "tydi_validation"]:
        rows = [r for r in over_q if r["dataset"] == dataset]
        if not rows:
            continue
        name = dataset
        qrows = rows_by_set[name]
        id_to_pos = {r["question_id"]: i for i, r in enumerate(qrows)}
        vec = np.stack([qvec[name][id_to_pos[r["question_id"]]] for r in rows])
        # reconstruct rows with gold
        gold_source = {r["question_id"]: r for r in qrows}
        split_rank = dense_rankings_fast(index, pidx, vec, rows, full=True)
        trunc_rank = dense_rankings_fast(alt_index, alt_pidx, vec, rows, full=True)

        split_full, split_over = [], []
        trunc_full, trunc_over = [], []
        for r in rows:
            qid = r["question_id"]
            g_full = set(gold_source[qid]["gold_passage_ids"])
            g_over = set(r["overlength_gold_ids"])
            split_full.append(recall_at_k(split_rank[qid], g_full, 10))
            split_over.append(recall_at_k(split_rank[qid], g_over, 10))
            trunc_full.append(recall_at_k(trunc_rank[qid], g_full, 10))
            trunc_over.append(recall_at_k(trunc_rank[qid], g_over, 10))

        split_full_ci = bootstrap_ci(np.array(split_full, dtype=float))
        trunc_full_ci = bootstrap_ci(np.array(trunc_full, dtype=float))
        split_over_ci = bootstrap_ci(np.array(split_over, dtype=float))
        trunc_over_ci = bootstrap_ci(np.array(trunc_over, dtype=float))

        ablation_rows.append({
            "dataset": dataset,
            "n_queries": len(rows),
            "split_R@10_full_gold": float(np.mean(split_full)),
            "split_R@10_full_gold_ci_low": split_full_ci[0],
            "split_R@10_full_gold_ci_high": split_full_ci[1],
            "trunc_R@10_full_gold": float(np.mean(trunc_full)),
            "trunc_R@10_full_gold_ci_low": trunc_full_ci[0],
            "trunc_R@10_full_gold_ci_high": trunc_full_ci[1],
            "delta_full_gold": float(np.mean(trunc_full) - np.mean(split_full)),
            "split_R@10_over_gold": float(np.mean(split_over)),
            "split_R@10_over_gold_ci_low": split_over_ci[0],
            "split_R@10_over_gold_ci_high": split_over_ci[1],
            "trunc_R@10_over_gold": float(np.mean(trunc_over)),
            "trunc_R@10_over_gold_ci_low": trunc_over_ci[0],
            "trunc_R@10_over_gold_ci_high": trunc_over_ci[1],
            "delta_over_gold": float(np.mean(trunc_over) - np.mean(split_over)),
        })
        print(f"  {dataset:16s} n={len(rows):4d} "
              f"split={np.mean(split_over):.4f} trunc={np.mean(trunc_over):.4f} "
              f"(over-length gold)")

    # ---- Save outputs -----------------------------------------------------
    with (RESULTS / "phase3_dense_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = ["dataset", "system", "n"] + METRICS + [
            f"{m}_{side}" for m in METRICS for side in ("ci_low", "ci_high")
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    with (RESULTS / "phase3_tydi_answerstring.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = ["dataset", "system", "n"] + METRICS + [
            f"{m}_{side}" for m in METRICS for side in ("ci_low", "ci_high")
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tydi_answer_rows)

    with (RESULTS / "phase3_ablation_truncated_vs_split.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(ablation_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(ablation_rows)

    (RESULTS / "phase3_dense_hybrid_summary.json").write_text(
        json.dumps({
            "model": "intfloat/multilingual-e5-small",
            "dtype": "fp16",
            "batch_size": 16,
            "prefixes": {"units": "passage: ", "queries": "query: "},
            "pooling": "mean",
            "normalization": "L2",
            "index": "FAISS IndexFlatIP",
            "units": len(texts),
            "parents": pidx.n_parents,
            "bm25": {"k1": best_k1, "b": best_b},
            "hybrid": {
                "rrf_k": best_hybrid["rrf_k"],
                "w_bm25": best_hybrid["w_bm25"],
                "w_dense": best_hybrid["w_dense"],
                "tuned_on": "miracl_train",
                "fuse_depth": FUSE_DEPTH,
            },
            "main_results": final_rows,
            "tydi_answerstring": tydi_answer_rows,
            "ablation": ablation_rows,
            "chunk_token_stats": chunk_stats,
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "ci": 0.95},
            "total_seconds": round(time.time() - start_time, 1),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- Print tables -----------------------------------------------------
    print("\n## Main results (parent-level, strict gold passage IDs)")
    for name in ["miracl_dev", "tydi_validation"]:
        print_table(name, [r for r in final_rows if r["dataset"] == name], [])
        print("MRR is over the full ranking for BM25/dense; hybrid fuses top-"
              f"{FUSE_DEPTH} lists.")

    print("\n## TyDi validation: answer-string hit")
    print_table("tydi_validation (answer-string)", tydi_answer_rows, [])

    print("\n## Truncated-vs-split dense Recall@10 (queries with over-length gold)")
    print("| dataset | n | split R@10 (over gold) | truncated R@10 (over gold) | delta | split R@10 (full gold) | truncated R@10 (full gold) |")
    print("|---|---|---|---|---|---|---|")
    for r in ablation_rows:
        print(f"| {r['dataset']} | {r['n_queries']} | "
              f"{r['split_R@10_over_gold']:.4f} [{r['split_R@10_over_gold_ci_low']:.4f}, {r['split_R@10_over_gold_ci_high']:.4f}] | "
              f"{r['trunc_R@10_over_gold']:.4f} [{r['trunc_R@10_over_gold_ci_low']:.4f}, {r['trunc_R@10_over_gold_ci_high']:.4f}] | "
              f"{r['delta_over_gold']:+.4f} | "
              f"{r['split_R@10_full_gold']:.4f} | {r['trunc_R@10_full_gold']:.4f} |")

    print("\n## Unit token lengths")
    a = chunk_stats["all_units"]
    c = chunk_stats["split_chunks_only"]
    print(f"total units: {chunk_stats['total_units']:,} "
          f"(split chunks: {chunk_stats['split_chunks']:,})")
    print(f"all units <20 tokens: {a['lt_20']:,} "
          f"({a['lt_20']/chunk_stats['total_units']:.2%}); "
          f"<10: {a['lt_10']:,}; <5: {a['lt_5']:,}")
    print(f"split chunks <20 tokens: {c['lt_20']:,} "
          f"({c['lt_20']/max(chunk_stats['split_chunks'],1):.2%}); "
          f"<10: {c['lt_10']:,}; <5: {c['lt_5']:,}")

    print(f"\n[done] {time.time()-start_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
