"""Read-only access to frozen Phase 3 retriever artifacts.

Nothing in this module re-embeds text, retrains a retriever, or changes a
retriever parameter. It reconstructs deterministic rankings from the artifacts
Phase 3 already produced (the shared chunk file, the cached BM25 index, and the
saved e5 unit/query vectors) so later phases can reuse them without recomputing
embeddings.

The main hybrid configuration is always the MIRACL-train-tuned setting recorded
in ``results/phase3_dense_hybrid_summary.json``. A separate sensitivity module
may tune a different setting, but it must not write here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .retrieval import (
    BM25Index,
    ParentIndex,
    aggregate_parent_scores,
    build_parent_index,
    rank_parents,
    tokenize,
)
from .fusion import rrf_fuse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
RESULTS = PROJECT_ROOT / "results"
CACHE = PROJECT_ROOT / "data" / "cache"

UNITS = PROCESSED / "corpus_chunks.jsonl"
PARENTS = PROCESSED / "corpus.jsonl"
BM25_CACHE = CACHE / "bm25_index.pkl"
UNITS_EMB = EMB_DIR / "corpus_units.f16.npy"
SUMMARY = RESULTS / "phase3_dense_hybrid_summary.json"
BM25_RANKINGS = RESULTS / "phase3_bm25_rankings.jsonl"
PER_QUERY = RESULTS / "phase3_per_query.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-small"
FUSE_DEPTH = 100

QUERY_SETS = {
    "miracl_train": PROCESSED / "miracl_train_questions.jsonl",
    "miracl_dev": PROCESSED / "miracl_dev_questions.jsonl",
    "tydi_train": PROCESSED / "tydi_train_questions.jsonl",
    "tydi_validation": PROCESSED / "tydi_validation_questions.jsonl",
}


def load_questions(name: str) -> list[dict]:
    path = QUERY_SETS[name]
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


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


def load_parent_texts() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with PARENTS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mapping[rec["parent_id"]] = rec["text"]
    return mapping


def load_query_embeddings(name: str, rows: Iterable[dict]) -> np.ndarray:
    rows = list(rows)
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


def load_bm25(texts: list[str]) -> BM25Index:
    if not BM25_CACHE.exists():
        raise FileNotFoundError(f"missing cached BM25 index: {BM25_CACHE}")
    return BM25Index.load(BM25_CACHE)


def build_dense_index() -> tuple["faiss.Index", int]:
    import faiss

    unit_emb = np.load(UNITS_EMB).astype(np.float32)
    if not np.isfinite(unit_emb).all():
        raise RuntimeError("non-finite cached corpus embeddings")
    dim = unit_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(unit_emb)
    return index, unit_emb.shape[0]


def main_hybrid_params() -> dict:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    hybrid = payload["hybrid"]
    if hybrid.get("tuned_on") != "miracl_train":
        raise RuntimeError("main hybrid is not the MIRACL-train-tuned setting")
    return hybrid


def bm25_rankings(
    index: BM25Index,
    pidx: ParentIndex,
    rows: list[dict],
    k1: float,
    b: float,
    top_k: int | None = 100,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        scores = index.score(tokenize(row["question"]), k1=k1, b=b)
        parent_scores = aggregate_parent_scores(scores, pidx)
        out[row["question_id"]] = rank_parents(
            parent_scores, pidx, top_k=top_k, only_positive=True
        )
    return out


def dense_rankings(
    index,
    unit_emb: np.ndarray,
    pidx: ParentIndex,
    query_vec: np.ndarray,
    rows: list[dict],
    top_k: int | None = 100,
    full_search: bool = True,
) -> dict[str, list[str]]:
    """Reproduce Phase 3 dense ranking: FAISS flat IP, max over parent units.

    Phase 3 searched every unit before aggregating to parents. Keep that exact
    behaviour (``full_search=True``) when reproducing its rankings, because a
    shallow unit search can miss a parent's best chunk.
    """
    n_units = index.ntotal
    k = n_units if full_search else min(max(top_k or 100, 1), n_units)
    out: dict[str, list[str]] = {}
    for i, row in enumerate(rows):
        scores, idx = index.search(query_vec[i:i + 1], k)
        scores = scores[0]
        idx = idx[0]
        valid = idx >= 0
        parent_scores = np.full(pidx.n_parents, -np.inf, dtype=np.float32)
        np.maximum.at(parent_scores, pidx.unit_to_parent[idx[valid]], scores[valid])
        out[row["question_id"]] = rank_parents(
            parent_scores, pidx, top_k=top_k, only_positive=False
        )
    return out


def hybrid_rankings(
    bm25: dict[str, list[str]],
    dense: dict[str, list[str]],
    rows: list[dict],
    rrf_k: int,
    w_bm25: float,
    fuse_depth: int = FUSE_DEPTH,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        qid = row["question_id"]
        out[qid] = rrf_fuse(
            [bm25[qid][:fuse_depth], dense[qid][:fuse_depth]],
            weights=[w_bm25, 1.0 - w_bm25],
            k=rrf_k,
        )
    return out


def load_saved_bm25_rankings() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with BM25_RANKINGS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["question_id"]] = rec["ranking"]
    return out


def load_phase3_per_query() -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    with PER_QUERY.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[(rec["dataset"], rec["system"], rec["qid"])] = rec
    return out


def build_units_and_parent_index() -> tuple[list[dict], list[str], list[str], ParentIndex]:
    recs, texts, parents = load_units()
    return recs, texts, parents, build_parent_index(parents)
