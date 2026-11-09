"""Phase 4: context compression on shared top-10 retrieval results.

Methods
  M0  no compression, top-k passages (k = 1, 3, 5, 10)
  M1  hard token budget, concatenate in rank order then truncate
  M2  extractive, e5-small sentence similarity, restore original order
  M3  extractive, BM25 sentence scores, restore original order
  M4  oracle lower bound (TyDi only; not deployable)
  M5  random sentence baseline (fixed per-query seed)

Retrievers: Dense (primary) and the main MIRACL-train-tuned Hybrid (secondary).
Top-10 parent passages come from the frozen Phase 3 ranking artifact. No text
is re-embedded for retrieval, no retriever is retrained, and no external API is
called. The only embedding model used is the existing multilingual-e5-small.

TyDi train is split with a fixed seed into 1,000 tuning questions and the rest
held out. Phase 4 method parameters are fixed by the protocol, so nothing is
fitted; evaluation uses the held-out training questions and TyDi validation.
MIRACL dev reports only gold-passage retention and context token counts because
MIRACL has no answer strings.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/phase4_context_compression.py
"""

from __future__ import annotations

import csv
import importlib.metadata as metadata
import json
import os
import platform
import random
import re
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from src.compression import (  # noqa: E402
    build_candidates,
    contains_answer,
    m0_top_k,
    m1_hard_budget,
    m2_e5,
    m3_bm25,
    m4_oracle,
    m5_random,
    normalize_for_match,
    split_sentences,
)
from src.retrieval import BM25Index, tokenize  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
RESULTS = PROJECT_ROOT / "results"
RANKINGS = RESULTS / "phase3_final_rankings_top100.jsonl"
CORPUS = PROCESSED / "corpus.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-small"
QUERY_SETS = {
    "miracl_dev": PROCESSED / "miracl_dev_questions.jsonl",
    "tydi_train": PROCESSED / "tydi_train_questions.jsonl",
    "tydi_validation": PROCESSED / "tydi_validation_questions.jsonl",
}
SOURCE_RANKINGS = {
    "tydi_heldout": "tydi_train",
    "tydi_validation": "tydi_validation",
    "miracl_dev": "miracl_dev",
}
SYSTEMS = ["Dense (e5-small)", "Hybrid (RRF)"]
SYSTEM_LABELS = {
    "Dense (e5-small)": "Dense (primary)",
    "Hybrid (RRF)": "Hybrid (secondary, MIRACL-tuned)",
}
METHOD_LABELS = {
    "M0": "M0 no compression",
    "M1": "M1 hard token budget",
    "M2": "M2 extractive e5",
    "M3": "M3 extractive BM25",
    "M4": "M4 oracle (not deployable)",
    "M5": "M5 random sentence baseline",
}
BUDGETS = [128, 256, 512, 1024]
M0_KS = [1, 3, 5, 10]
TYDI_SPLIT_SEED = 42
TYDI_TUNING_N = 1000
M5_SEED = 2026
N_BOOT = 10_000
BOOT_SEED = 42
CHUNK_QUERIES = 16
EMBED_BATCH = 64
BM25_K1 = 0.9
BM25_B = 0.35
MAX_SEQ_LEN = 512
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
START = time.time()


class O200k:
    def __init__(self) -> None:
        import tiktoken

        os.environ.setdefault(
            "TIKTOKEN_CACHE_DIR", str(PROJECT_ROOT / "data" / "cache" / "tiktoken")
        )
        self.enc = tiktoken.get_encoding("o200k_base")
        self.name = "o200k_base"
        self.version = metadata.version("tiktoken")

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))

    def truncate(self, text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        ids = self.enc.encode(text)[:budget]
        out = self.enc.decode(ids)
        while ids and len(self.enc.encode(out)) > budget:
            ids = ids[:-1]
            out = self.enc.decode(ids)
        return out


class E5Encoder:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, dtype=torch.float16
        ).to(DEVICE).eval()
        self.device = DEVICE

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_SEQ_LEN, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(F.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_questions(name: str) -> list[dict]:
    return load_jsonl(QUERY_SETS[name])


def load_rankings() -> dict[str, dict[str, dict[str, list[str]]]]:
    out: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with RANKINGS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["system"] not in SYSTEMS:
                continue
            if rec["dataset"] not in SOURCE_RANKINGS.values():
                continue
            out[rec["dataset"]][rec["system"]][rec["question_id"]] = rec[
                "ranking"
            ][:10]
    return out


def load_parent_texts(needed: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["parent_id"]
            if pid in needed:
                texts[pid] = rec["text"]
    return texts


def load_query_vectors(name: str, rows: list[dict]) -> dict[str, np.ndarray]:
    npy = EMB_DIR / f"queries_{name}.f16.npy"
    meta = EMB_DIR / f"queries_{name}.json"
    ids = json.loads(meta.read_text(encoding="utf-8"))["question_ids"]
    expected = [r["question_id"] for r in rows]
    if ids != expected:
        raise RuntimeError(f"query embedding order mismatch for {name}")
    vectors = np.load(npy).astype(np.float32)
    return {r["question_id"]: vectors[i] for i, r in enumerate(rows)}


def bootstrap_mean(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
    }


def m1_with_spans(
    passages: list[str], budget: int, o200k: O200k
) -> tuple[str, set[int]]:
    sep = "\n\n"
    full = sep.join(passages)
    if o200k.count(full) <= budget:
        ids = o200k.enc.encode(full)
        context = full
    else:
        ids = o200k.enc.encode(full)[:budget]
        context = o200k.enc.decode(ids)
        while ids and o200k.count(context) > budget:
            ids = ids[:-1]
            context = o200k.enc.decode(ids)
    prefix = ""
    starts: list[int] = []
    for i, passage in enumerate(passages):
        if i > 0:
            prefix += sep
        starts.append(len(o200k.enc.encode(prefix)))
        prefix += passage
    retained = {i for i, start in enumerate(starts) if start < len(ids)}
    return context, retained


def mean_ci_text(mean, lo, hi, digits: int = 4) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_jsonl_line(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def add_stat(
    stats: dict,
    per_query_handle,
    o200k: O200k,
    e5: E5Encoder,
    dataset: str,
    kind: str,
    system: str,
    method: str,
    setting: str,
    budget_tokens: int | None,
    top_k: int | None,
    qid: str,
    question: str,
    context: str,
    retained_ranks: set[int],
    top10: list[str],
    answers: list[str],
    gold_ids: set[str],
    m0_has_answer: bool | None,
    gold_in_top10: bool,
) -> None:
    o_tokens = o200k.count(context)
    e_tokens = e5.count(context)
    has_answer = None
    loss = None
    if kind == "tydi":
        has_answer = contains_answer(context, answers)[0] if answers else False
        if m0_has_answer:
            loss = 0.0 if has_answer else 1.0
    retained_gold = bool(gold_ids & {top10[r] for r in retained_ranks})

    key = (dataset, system, method, setting, budget_tokens, top_k)
    bucket = stats[key]
    bucket["o200k"].append(o_tokens)
    bucket["e5"].append(e_tokens)
    if has_answer is not None:
        bucket["recall"].append(1.0 if has_answer else 0.0)
    if loss is not None:
        bucket["loss"].append(1.0 if loss else 0.0)
    if gold_in_top10:
        bucket["gold"].append(1.0 if retained_gold else 0.0)

    write_jsonl_line(per_query_handle, {
        "dataset": dataset,
        "dataset_kind": kind,
        "retriever": system,
        "method": method,
        "setting": setting,
        "budget_tokens": budget_tokens,
        "top_k": top_k,
        "question_id": qid,
        "question": question,
        "o200k_tokens": o_tokens,
        "e5_tokens": e_tokens,
        "answer_evidence_recall": has_answer,
        "gold_passage_retained": retained_gold if gold_in_top10 else None,
        "gold_in_top10": gold_in_top10,
        "m0_top10_has_answer": m0_has_answer,
        "evidence_loss": loss,
    })


def capture_m2_example(
    examples: list[dict],
    o200k: O200k,
    dataset: str,
    system: str,
    budget: int,
    row: dict,
    answers: list[str],
    top10: list[str],
    candidates,
    scores: np.ndarray,
    result,
    m0_context: str,
    m0_has_answer: bool,
) -> None:
    if dataset != "tydi_heldout" and dataset != "tydi_validation":
        return
    if not m0_has_answer:
        return
    if contains_answer(result.context, answers)[0]:
        return
    if len([e for e in examples if e["question_id"] == row["question_id"]]) > 0:
        return
    if len([e for e in examples if (e["dataset"], e["retriever"], e["budget"])
            == (dataset, system, budget)]) >= 25:
        return

    answer_sents = [
        (i, c) for i, c in enumerate(candidates)
        if contains_answer(c.text, answers)[0]
    ]
    if not answer_sents:
        examples.append({
            "dataset": dataset,
            "retriever": system,
            "budget": budget,
            "question_id": row["question_id"],
            "question": row["question"],
            "gold_answers": answers,
            "answer_sentence": None,
            "answer_passage_id": None,
            "answer_passage_rank": None,
            "answer_sentence_o200k_tokens": None,
            "answer_sentence_score": None,
            "answer_sentence_score_rank": None,
            "n_selected_sentences": len(result.selected),
            "m0_top10_context": m0_context,
            "m2_context": result.context,
            "reason": (
                "The normalized gold answer spans a sentence boundary, so no "
                "single candidate sentence contained the full answer string."
            ),
        })
        return

    order = sorted(
        range(len(candidates)), key=lambda i: (-scores[i], candidates[i].order)
    )
    rank_by_index = {idx: pos + 1 for pos, idx in enumerate(order)}
    best_i, best = max(
        answer_sents, key=lambda item: (scores[item[0]], -item[1].order)
    )
    n_tok = o200k.count(best.text)
    selected = {c.order for c in result.selected}
    if best.order in selected:
        return
    if n_tok > budget:
        reason = (
            f"The answer-bearing sentence is {n_tok} o200k tokens, longer than "
            f"the {budget}-token budget, so it could not be selected intact."
        )
    else:
        reason = (
            f"None of the {len(answer_sents)} answer-bearing sentence(s) made "
            f"the budget cut. The best one ranked {rank_by_index[best_i]} by "
            f"e5 similarity ({scores[best_i]:.3f}) and needed {n_tok} o200k "
            f"tokens; higher-scoring sentences filled the {budget}-token budget."
        )

    examples.append({
        "dataset": dataset,
        "retriever": system,
        "budget": budget,
        "question_id": row["question_id"],
        "question": row["question"],
        "gold_answers": answers,
        "answer_sentence": best.text,
        "answer_passage_id": top10[best.passage_rank],
        "answer_passage_rank": best.passage_rank + 1,
        "answer_sentence_o200k_tokens": n_tok,
        "answer_sentence_score": float(scores[best_i]),
        "answer_sentence_score_rank": rank_by_index[best_i],
        "n_selected_sentences": len(result.selected),
        "m0_top10_context": m0_context,
        "m2_context": result.context,
        "reason": reason,
    })


def process_item(
    item: dict,
    dataset: str,
    kind: str,
    o200k: O200k,
    e5: E5Encoder,
    parent_texts: dict[str, str],
    stats: dict,
    per_query_handle,
    examples: list[dict],
    runtime: dict,
    counters: dict,
) -> None:
    row = item["row"]
    qid = row["question_id"]
    question = row["question"]
    qvec = item["qvec"]
    answers = [a for a in row.get("gold_answers", []) if normalize_for_match(a)]
    gold_ids = set(row["gold_passage_ids"])

    for system in SYSTEMS:
        data = item["systems"][system]
        top10 = data["top10"]
        passages = data["passages"]
        candidates = data["candidates"]
        if not candidates:
            counters["empty_candidates"] += 1
            continue

        m0_context = m0_top_k(passages, 10)
        m0_has_answer = (
            contains_answer(m0_context, answers)[0] if kind == "tydi" else None
        )
        gold_in_top10 = bool(gold_ids & set(top10))
        counters["questions_processed"] += 1

        t0 = time.perf_counter()
        for k in M0_KS:
            context = m0_top_k(passages, k)
            retained = set(range(min(k, len(passages))))
            add_stat(
                stats, per_query_handle, o200k, e5, dataset, kind, system,
                "M0", f"top{k}", None, k, qid, question, context, retained,
                top10, answers, gold_ids, m0_has_answer, gold_in_top10,
            )
        runtime["M0"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for budget in BUDGETS:
            context, retained = m1_with_spans(passages, budget, o200k)
            add_stat(
                stats, per_query_handle, o200k, e5, dataset, kind, system,
                "M1", f"B{budget}", budget, None, qid, question, context,
                retained, top10, answers, gold_ids, m0_has_answer,
                gold_in_top10,
            )
        runtime["M1"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        matrix = np.stack([item["vec_map"][c.text] for c in candidates])
        scores = matrix @ qvec
        runtime["M2_sentence_scoring"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for budget in BUDGETS:
            result = m2_e5(
                candidates, scores, budget, o200k.count, o200k.truncate
            )
            retained = {c.passage_rank for c in result.selected}
            add_stat(
                stats, per_query_handle, o200k, e5, dataset, kind, system,
                "M2", f"B{budget}", budget, None, qid, question,
                result.context, retained, top10, answers, gold_ids,
                m0_has_answer, gold_in_top10,
            )
            capture_m2_example(
                examples, o200k, dataset, system, budget, row, answers,
                top10, candidates, scores, result, m0_context, m0_has_answer,
            )
        runtime["M2"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        bm25 = BM25Index([c.text for c in candidates], k1=BM25_K1, b=BM25_B)
        bm25_scores = bm25.score(tokenize(question))
        runtime["M3_sentence_scoring"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for budget in BUDGETS:
            result = m3_bm25(
                candidates, bm25_scores, budget, o200k.count, o200k.truncate
            )
            retained = {c.passage_rank for c in result.selected}
            add_stat(
                stats, per_query_handle, o200k, e5, dataset, kind, system,
                "M3", f"B{budget}", budget, None, qid, question,
                result.context, retained, top10, answers, gold_ids,
                m0_has_answer, gold_in_top10,
            )
        runtime["M3"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for budget in BUDGETS:
            rng = random.Random(f"{M5_SEED}|{qid}|{budget}")
            result = m5_random(
                candidates, budget, o200k.count, o200k.truncate, rng
            )
            retained = {c.passage_rank for c in result.selected}
            add_stat(
                stats, per_query_handle, o200k, e5, dataset, kind, system,
                "M5", f"B{budget}", budget, None, qid, question,
                result.context, retained, top10, answers, gold_ids,
                m0_has_answer, gold_in_top10,
            )
        runtime["M5"] += time.perf_counter() - t0

        if kind == "tydi":
            t0 = time.perf_counter()
            result = m4_oracle(candidates, answers)
            retained = {c.passage_rank for c in result.selected}
            add_stat(
                stats, per_query_handle, o200k, e5, dataset, kind, system,
                "M4", "oracle", None, None, qid, question, result.context,
                retained, top10, answers, gold_ids, m0_has_answer,
                gold_in_top10,
            )
            runtime["M4"] += time.perf_counter() - t0


def rows_from_stats(stats: dict) -> list[dict]:
    out = []
    for (dataset, system, method, setting, budget_tokens, top_k), bucket in sorted(
        stats.items(),
        key=lambda kv: (
            kv[0][0], kv[0][1], kv[0][2], kv[0][4] or 0, kv[0][5] or 0,
        ),
    ):
        o = bootstrap_mean(bucket["o200k"])
        e = bootstrap_mean(bucket["e5"])
        recall = bootstrap_mean(bucket["recall"]) if bucket["recall"] else None
        gold = bootstrap_mean(bucket["gold"]) if bucket["gold"] else None
        loss = bootstrap_mean(bucket["loss"]) if bucket["loss"] else None
        kind = "miracl" if dataset == "miracl_dev" else "tydi"
        out.append({
            "dataset": dataset,
            "dataset_kind": kind,
            "retriever": system,
            "retriever_label": SYSTEM_LABELS[system],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "setting": setting,
            "budget_tokens": budget_tokens,
            "top_k": top_k,
            "n_questions": o["n"],
            "n_answer_denominator": recall["n"] if recall else None,
            "n_gold_denominator": gold["n"] if gold else None,
            "n_loss_denominator": loss["n"] if loss else None,
            "mean_o200k_tokens": o["mean"],
            "o200k_ci_low": o["ci_low"],
            "o200k_ci_high": o["ci_high"],
            "mean_e5_tokens": e["mean"],
            "e5_ci_low": e["ci_low"],
            "e5_ci_high": e["ci_high"],
            "answer_evidence_recall": recall["mean"] if recall else None,
            "recall_ci_low": recall["ci_low"] if recall else None,
            "recall_ci_high": recall["ci_high"] if recall else None,
            "gold_passage_retention": gold["mean"] if gold else None,
            "gold_ci_low": gold["ci_low"] if gold else None,
            "gold_ci_high": gold["ci_high"] if gold else None,
            "evidence_loss_rate": loss["mean"] if loss else None,
            "loss_ci_low": loss["ci_low"] if loss else None,
            "loss_ci_high": loss["ci_high"] if loss else None,
        })
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_long_ci(rows: list[dict]) -> None:
    metric_defs = [
        ("mean_o200k_tokens", "o200k_ci_low", "o200k_ci_high", "n_questions"),
        ("mean_e5_tokens", "e5_ci_low", "e5_ci_high", "n_questions"),
        ("answer_evidence_recall", "recall_ci_low", "recall_ci_high",
         "n_answer_denominator"),
        ("gold_passage_retention", "gold_ci_low", "gold_ci_high",
         "n_gold_denominator"),
        ("evidence_loss_rate", "loss_ci_low", "loss_ci_high",
         "n_loss_denominator"),
    ]
    out = []
    for row in rows:
        for metric, lo_key, hi_key, n_key in metric_defs:
            if row[metric] is None:
                continue
            out.append({
                "dataset": row["dataset"],
                "retriever": row["retriever"],
                "method": row["method"],
                "setting": row["setting"],
                "metric": metric,
                "value": row[metric],
                "ci95_low": row[lo_key],
                "ci95_high": row[hi_key],
                "n": row[n_key],
                "bootstrap_resamples": N_BOOT,
                "bootstrap_seed": BOOT_SEED,
            })
    write_csv(RESULTS / "phase4_bootstrap_ci.csv", out)


def write_plot(rows: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    styles = {
        "M0": ("#1f77b4", "o", "-"),
        "M1": ("#ff7f0e", "s", "-"),
        "M2": ("#2ca02c", "^", "-"),
        "M3": ("#d62728", "D", "-"),
        "M5": ("#7f7f7f", "v", "--"),
        "M4": ("#9467bd", "*", ":"),
    }
    for ax, system in zip(axes, SYSTEMS):
        for method in ["M0", "M1", "M2", "M3", "M5", "M4"]:
            pts = [
                r for r in rows
                if r["dataset"] == "tydi_heldout"
                and r["retriever"] == system
                and r["method"] == method
                and r["answer_evidence_recall"] is not None
            ]
            if not pts:
                continue
            pts = sorted(pts, key=lambda r: r["mean_o200k_tokens"])
            xs = [r["mean_o200k_tokens"] for r in pts]
            ys = [r["answer_evidence_recall"] for r in pts]
            color, marker, ls = styles[method]
            label = METHOD_LABELS[method]
            if method == "M4":
                label += " (TyDi)"
            ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                    label=label, markersize=7, linewidth=1.8)
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xlabel("Mean context tokens (o200k_base)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Answer-evidence recall (TyDi held-out)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Phase 4: answer-evidence recall vs context cost "
        f"(TyDi held-out train, n={next((r['n_questions'] for r in rows if r['dataset']=='tydi_heldout'), 0)})"
    )
    fig.tight_layout()
    path = RESULTS / "phase4_recall_vs_tokens.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def select_examples(examples: list[dict]) -> list[dict]:
    preferences = [
        ("tydi_heldout", "Dense (e5-small)", 128),
        ("tydi_heldout", "Dense (e5-small)", 256),
        ("tydi_heldout", "Hybrid (RRF)", 128),
        ("tydi_heldout", "Hybrid (RRF)", 256),
        ("tydi_validation", "Dense (e5-small)", 128),
        ("tydi_validation", "Hybrid (RRF)", 128),
    ]
    selected: list[dict] = []
    seen: set[str] = set()
    for dataset, system, budget in preferences:
        for ex in sorted(examples, key=lambda e: e["question_id"]):
            if ex["question_id"] in seen:
                continue
            if (ex["dataset"], ex["retriever"], ex["budget"]) != (
                dataset, system, budget
            ):
                continue
            selected.append(ex)
            seen.add(ex["question_id"])
            if len(selected) == 5:
                return selected
            break
    for ex in sorted(examples, key=lambda e: e["question_id"]):
        if ex["question_id"] in seen:
            continue
        selected.append(ex)
        seen.add(ex["question_id"])
        if len(selected) == 5:
            return selected
    return selected


def write_examples(examples: list[dict]) -> None:
    selected = select_examples(examples)
    (RESULTS / "phase4_m2_answer_loss_examples.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Phase 4: five verbatim M2 answer-loss examples",
        "",
        "M2 uses e5-small sentence similarity. Each example is a TyDi question "
        "where the M0 top-10 context contained a gold answer but the M2 context "
        "did not. Text below is copied verbatim from the dataset and the "
        "experiment output.",
        "",
    ]
    for i, ex in enumerate(selected, start=1):
        lines += [
            f"## Example {i}",
            "",
            f"- Dataset: `{ex['dataset']}`",
            f"- Retriever: `{ex['retriever']}`",
            f"- Budget: {ex['budget']} o200k_base tokens",
            f"- Question ID: `{ex['question_id']}`",
            "",
            "**Question**",
            "",
            f"> {ex['question']}",
            "",
            "**Gold answer(s)**",
            "",
            f"> {', '.join(ex['gold_answers'])}",
            "",
            "**Original answer-bearing sentence (verbatim)**",
            "",
            f"> {ex['answer_sentence'] if ex['answer_sentence'] else 'n/a'}",
            "",
            f"Parent passage: `{ex['answer_passage_id']}` at retrieval rank "
            f"{ex['answer_passage_rank']}.",
            "",
            "**M2 compressed context (verbatim output)**",
            "",
            "```text",
            ex["m2_context"],
            "```",
            "",
            "**Why the answer was lost**",
            "",
            ex["reason"],
            "",
        ]
    (RESULTS / "phase4_m2_answer_loss_examples.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def markdown_table(rows: list[dict], include_loss: bool = True) -> str:
    header = (
        "| method | setting | o200k tokens | e5 tokens | answer recall | "
        "gold retention | evidence loss |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        recall = (
            mean_ci_text(r["answer_evidence_recall"], r["recall_ci_low"],
                         r["recall_ci_high"])
            if r["answer_evidence_recall"] is not None else "n/a"
        )
        gold = (
            mean_ci_text(r["gold_passage_retention"], r["gold_ci_low"],
                         r["gold_ci_high"])
            if r["gold_passage_retention"] is not None else "n/a"
        )
        loss = (
            mean_ci_text(r["evidence_loss_rate"], r["loss_ci_low"],
                         r["loss_ci_high"])
            if include_loss and r["evidence_loss_rate"] is not None else "n/a"
        )
        lines.append(
            f"| {r['method_label']} | {r['setting']} | "
            f"{mean_ci_text(r['mean_o200k_tokens'], r['o200k_ci_low'], r['o200k_ci_high'], 1)} | "
            f"{mean_ci_text(r['mean_e5_tokens'], r['e5_ci_low'], r['e5_ci_high'], 1)} | "
            f"{recall} | {gold} | {loss} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict], config: dict, runtime: dict,
                 counters: dict, memory: dict, examples_path: Path,
                 plot_path: Path) -> None:
    def subset(dataset: str, system: str, methods: list[str] | None = None):
        out = [
            r for r in rows
            if r["dataset"] == dataset and r["retriever"] == system
            and (methods is None or r["method"] in methods)
        ]
        return out

    lines = [
        "# Phase 3 finalization and Phase 4 context compression",
        "",
        "## Phase 3 freeze",
        "",
        f"- Tag `phase3-complete` -> commit `{config['phase3_commit']}` "
        f"(`{config['phase3_commit_message']}`).",
        f"- Working tree changes preserved: {config['working_tree_changes_preserved']}.",
        "- Phase 3 result checksums: `results/phase3_freeze_manifest.json`.",
        "",
        "## Phase 3 final MRR@100 (metric-only recomputation)",
        "",
        "Every ranking is capped at 100 candidates before MRR is computed. "
        "Rankings were replayed from frozen Phase 3 artifacts; the replay "
        "matched all Phase 3 top-20 records exactly. No embeddings were "
        "recomputed and no retriever was reconfigured.",
        "",
    ]
    mrr_payload = json.loads(
        (RESULTS / "phase3_final_mrr100.json").read_text(encoding="utf-8")
    )
    lines += [
        "| dataset | system | n | MRR@100 | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for r in mrr_payload["results"]:
        lines.append(
            f"| {r['dataset']} | {r['system']} | {r['n']} | "
            f"{r['mrr@100']:.4f} | [{r['mrr@100_ci_low']:.4f}, "
            f"{r['mrr@100_ci_high']:.4f}] |"
        )
    lines += [
        "",
        "## TyDi-train-tuned hybrid sensitivity (separate experiment)",
        "",
        "This is **not** the primary hybrid. The main hybrid remains the "
        "MIRACL-train-tuned setting. See "
        "`results/phase3_tydi_hybrid_sensitivity.md` for the full table.",
        "",
    ]
    sens = json.loads(
        (RESULTS / "phase3_tydi_hybrid_sensitivity.json").read_text(
            encoding="utf-8"
        )
    )
    sv = sens["validation_result"]
    lines += [
        "| setting | w_BM25 | w_dense | R@10 | MRR@100 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {sens['label']} | {sv['w_bm25']:.2f} | {sv['w_dense']:.2f} | "
        f"{sv['recall@10']:.4f} | {sv['mrr@100']:.4f} | {sv['ndcg@10']:.4f} |",
        "",
        "## Phase 4 scope and integrity",
        "",
        "- Dense is the primary Phase 4 retriever; Hybrid is secondary and "
        "always uses the main MIRACL-train-tuned RRF configuration.",
        "- All methods use exactly the same top-10 parent passages per query "
        "and retriever.",
        "- The custom Bengali QA review dataset was not read, filtered, "
        "overwritten, or mixed into these benchmarks. Manual `keep`/`reject`/"
        "`fix` decisions were preserved.",
        "- No external API was called and no new model was introduced. The only "
        "embedding model is the existing `intfloat/multilingual-e5-small`.",
        f"- TyDi split seed {config['tydi_split']['seed']}: "
        f"{config['tydi_split']['n_tuning']} tuning questions and "
        f"{config['tydi_split']['n_heldout']} held-out training questions. "
        "Method parameters are fixed, so no fitting was performed; evaluation "
        "uses the held-out split and validation.",
        f"- MIRACL reports only gold-passage retention and token counts; no "
        "answer recall or answer-correctness claim is made.",
        "",
        "## Token counting",
        "",
        f"- Budget enforcement: `{config['tokenizers']['budget']}` "
        f"(tiktoken {config['tokenizers']['tiktoken_version']}); special tokens "
        "excluded.",
        f"- Comparison tokenizer: `{config['tokenizers']['e5']}` "
        f"(transformers {config['tokenizers']['transformers_version']}); "
        "special tokens included, no `passage: ` prefix. Empty context counts "
        "as 0.",
        "- M1 truncates the concatenated string at the token budget, decoding "
        "the retained token IDs and trimming until re-encoding is <= budget. "
        "M2/M3/M5 include a sentence only if it fits the remaining budget; a "
        "single sentence longer than the whole budget is truncated to the "
        "budget as a documented fallback.",
        "",
        "## Primary results: TyDi held-out (dense, primary retriever)",
        "",
        markdown_table(subset("tydi_heldout", "Dense (e5-small)")),
        "",
        "## Secondary results: TyDi held-out (hybrid, MIRACL-tuned)",
        "",
        markdown_table(subset("tydi_heldout", "Hybrid (RRF)")),
        "",
        "## TyDi validation (dense and hybrid)",
        "",
        "### Dense",
        "",
        markdown_table(subset("tydi_validation", "Dense (e5-small)")),
        "",
        "### Hybrid (main MIRACL-tuned configuration)",
        "",
        markdown_table(subset("tydi_validation", "Hybrid (RRF)")),
        "",
        "## MIRACL dev (retention and tokens only)",
        "",
        "### Dense",
        "",
        markdown_table([
            r for r in rows
            if r["dataset"] == "miracl_dev"
            and r["retriever"] == "Dense (e5-small)"
        ], include_loss=False),
        "",
        "### Hybrid (main MIRACL-tuned configuration)",
        "",
        markdown_table([
            r for r in rows
            if r["dataset"] == "miracl_dev"
            and r["retriever"] == "Hybrid (RRF)"
        ], include_loss=False),
        "",
        "MIRACL has no answer strings, so answer-evidence recall and "
        "evidence-loss rate are intentionally not reported.",
        "",
        "## Bootstrap",
        "",
        f"- {N_BOOT:,} question-level resamples, seed {BOOT_SEED}, percentile "
        "95% CI.",
        "- The sampling unit is the question. Methods for a given dataset and "
        "retriever use the same question subset; evidence-loss CIs resample "
        "only questions where M0 top-10 contained the answer.",
        "- Denominators are reported in the tables as `n` for answer metrics "
        "and in the CSV columns `n_*_denominator`.",
        "",
        "## Required examples",
        "",
        f"Five verbatim M2 answer-loss examples: `{examples_path.name}` "
        f"(JSON: `phase4_m2_answer_loss_examples.json`).",
        "",
        "## Tests",
        "",
        "- Sentence splitter and answer matcher tests: "
        "`results/phase4_test_results.txt` (19 tests) and full suite "
        "`results/phase4_test_results_all.txt` (110 tests).",
        "",
        "## Runtime, memory, and hardware",
        "",
        f"- Total runtime: {config['runtime_seconds']:.1f}s.",
        f"- Peak RSS: {memory['peak_rss_mib']:.1f} MiB "
        f"({memory['measurement']}).",
        f"- Device: {config['hardware']['device']}; "
        f"{config['hardware']['platform']}.",
        f"- Questions processed: {counters['questions_processed']} "
        "query-retriever pairs; failed: "
        f"{counters['failed']}; skipped/empty candidate sets: "
        f"{counters['empty_candidates']}.",
        "",
        "| stage | seconds |",
        "|---|---:|",
    ]
    for key in sorted(runtime):
        lines.append(f"| {key} | {runtime[key]:.2f} |")
    lines += [
        "",
        "## Plot",
        "",
        f"![Answer-evidence recall vs context tokens]({plot_path.name})",
        "",
        "## Result labels",
        "",
        "- **Primary**: TyDi held-out, dense retriever.",
        "- **Secondary**: TyDi held-out hybrid, and TyDi validation for both "
        "retrievers.",
        "- **Sensitivity**: TyDi-train-tuned hybrid RRF weights, kept separate "
        "from the main hybrid.",
        "- **Oracle lower bound**: M4 retains gold-answer sentences; it is not "
        "a deployable method and was not used to tune M1/M2/M3.",
        "- **Random baseline**: M5 is a sanity baseline, not a competitive "
        "method.",
        "",
    ]
    (RESULTS / "phase4_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"[start] device={DEVICE}")

    tydi_train_rows = load_questions("tydi_train")
    tydi_val_rows = load_questions("tydi_validation")
    miracl_rows = load_questions("miracl_dev")

    indices = list(range(len(tydi_train_rows)))
    rng = random.Random(TYDI_SPLIT_SEED)
    tuning_idx = sorted(rng.sample(indices, TYDI_TUNING_N))
    tuning_set = set(tuning_idx)
    heldout_idx = [i for i in indices if i not in tuning_set]
    tydi_heldout_rows = [tydi_train_rows[i] for i in heldout_idx]
    (RESULTS / "phase4_tydi_split.json").write_text(json.dumps({
        "seed": TYDI_SPLIT_SEED,
        "n_total": len(tydi_train_rows),
        "n_tuning": len(tuning_idx),
        "n_heldout": len(heldout_idx),
        "tuning_question_ids": [tydi_train_rows[i]["question_id"] for i in tuning_idx],
        "heldout_question_ids": [r["question_id"] for r in tydi_heldout_rows],
        "note": "Phase 4 method parameters are fixed; no fitting used this split.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[split] TyDi train {len(tydi_train_rows)} -> tuning {len(tuning_idx)}, "
          f"held-out {len(heldout_idx)}")

    rankings = load_rankings()
    dataset_rows = {
        "tydi_heldout": tydi_heldout_rows,
        "tydi_validation": tydi_val_rows,
        "miracl_dev": miracl_rows,
    }
    dataset_kinds = {"tydi_heldout": "tydi", "tydi_validation": "tydi",
                     "miracl_dev": "miracl"}

    needed_parents: set[str] = set()
    for dataset, rows in dataset_rows.items():
        source = SOURCE_RANKINGS[dataset]
        for system in SYSTEMS:
            for row in rows:
                needed_parents.update(
                    rankings[source][system].get(row["question_id"], [])[:10]
                )
    print(f"[load] needed parent passages: {len(needed_parents):,}")
    parent_texts = load_parent_texts(needed_parents)
    missing = needed_parents - set(parent_texts)
    if missing:
        raise RuntimeError(f"missing parent texts for {len(missing)} IDs")

    qvec_maps = {}
    for source in ["tydi_train", "tydi_validation", "miracl_dev"]:
        rows = load_questions(source)
        qvec_maps[source] = load_query_vectors(source, rows)

    o200k = O200k()
    print(f"[load] o200k_base (tiktoken {o200k.version})")
    e5 = E5Encoder()
    print(f"[load] e5 encoder on {e5.device}")

    stats: dict = defaultdict(lambda: {
        "o200k": [], "e5": [], "recall": [], "gold": [], "loss": [],
    })
    runtime: dict = defaultdict(float)
    counters = {"questions_processed": 0, "failed": 0,
                "empty_candidates": 0}
    examples: list[dict] = []

    per_query_path = RESULTS / "phase4_per_query_metrics.jsonl"
    with per_query_path.open("w", encoding="utf-8") as per_query_handle:
        for dataset, rows in dataset_rows.items():
            kind = dataset_kinds[dataset]
            source = SOURCE_RANKINGS[dataset]
            print(f"[run] {dataset}: {len(rows)} questions")
            for start in range(0, len(rows), CHUNK_QUERIES):
                chunk = rows[start:start + CHUNK_QUERIES]
                items = []
                unique_sentences: set[str] = set()
                for row in chunk:
                    qid = row["question_id"]
                    item = {
                        "row": row,
                        "qvec": qvec_maps[source][qid],
                        "systems": {},
                    }
                    for system in SYSTEMS:
                        top10 = rankings[source][system].get(qid, [])[:10]
                        passages = [parent_texts[pid] for pid in top10]
                        candidates = build_candidates(passages)
                        item["systems"][system] = {
                            "top10": top10,
                            "passages": passages,
                            "candidates": candidates,
                        }
                        for cand in candidates:
                            unique_sentences.add(cand.text)
                    items.append(item)

                texts = sorted(unique_sentences)
                t0 = time.perf_counter()
                vectors = e5.embed([f"passage: {t}" for t in texts])
                runtime["M2_sentence_embedding"] += time.perf_counter() - t0
                vec_map = {t: vectors[i] for i, t in enumerate(texts)}
                for item in items:
                    item["vec_map"] = vec_map
                    process_item(
                        item, dataset, kind, o200k, e5, parent_texts,
                        stats, per_query_handle, examples, runtime, counters,
                    )
                print(f"  {min(start + CHUNK_QUERIES, len(rows))}/{len(rows)}",
                      flush=True)

    rows = rows_from_stats(stats)
    tydi_rows = [r for r in rows if r["dataset_kind"] == "tydi"]
    miracl_out = [r for r in rows if r["dataset_kind"] == "miracl"]
    write_csv(RESULTS / "phase4_results_all.csv", rows)
    write_csv(RESULTS / "phase4_results_dense.csv",
              [r for r in tydi_rows if r["retriever"] == "Dense (e5-small)"])
    write_csv(RESULTS / "phase4_results_hybrid.csv",
              [r for r in tydi_rows if r["retriever"] == "Hybrid (RRF)"])
    write_csv(RESULTS / "phase4_results_miracl.csv", miracl_out)
    write_long_ci(rows)

    write_examples(examples)
    plot_path = write_plot(rows)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    memory = {
        "peak_rss_bytes": int(peak_bytes),
        "peak_rss_mib": peak_bytes / (1024 ** 2),
        "measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss; "
                       "macOS reports bytes, Linux KiB",
        "device": DEVICE,
    }
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": DEVICE,
        "torch_version": torch.__version__,
    }
    config = {
        "experiment": "Phase 4 context compression",
        "phase3_tag": "phase3-complete",
        "phase3_commit": json.loads(
            (RESULTS / "phase3_freeze_manifest.json").read_text(encoding="utf-8")
        )["tag_commit"],
        "phase3_commit_message": json.loads(
            (RESULTS / "phase3_freeze_manifest.json").read_text(encoding="utf-8")
        )["phase3_commit_message"],
        "working_tree_changes_preserved": json.loads(
            (RESULTS / "phase3_freeze_manifest.json").read_text(encoding="utf-8")
        )["working_tree_changes_preserved"],
        "retrievers": {
            "primary": "Dense (e5-small)",
            "secondary": "Hybrid (RRF), main MIRACL-train-tuned configuration",
        },
        "systems": SYSTEMS,
        "datasets": {
            "tydi_heldout": len(tydi_heldout_rows),
            "tydi_validation": len(tydi_val_rows),
            "miracl_dev": len(miracl_rows),
        },
        "tydi_split": {
            "seed": TYDI_SPLIT_SEED,
            "n_tuning": len(tuning_idx),
            "n_heldout": len(heldout_idx),
        },
        "methods": {
            "M0": {"k": M0_KS, "order_preserved": True},
            "M1": {"budgets": BUDGETS, "truncation": "o200k token prefix"},
            "M2": {"budgets": BUDGETS, "scorer": "multilingual-e5-small cosine"},
            "M3": {"budgets": BUDGETS, "scorer": "BM25",
                   "k1": BM25_K1, "b": BM25_B},
            "M4": {"scope": "TyDi only", "oracle": True},
            "M5": {"budgets": BUDGETS, "seed": M5_SEED,
                   "rng": "random.Random(f'{seed}|{qid}|{budget}')"},
        },
        "tokenizers": {
            "budget": "o200k_base",
            "tiktoken_version": o200k.version,
            "e5": "intfloat/multilingual-e5-small tokenizer",
            "transformers_version": metadata.version("transformers"),
            "o200k_special_tokens": "excluded",
            "e5_special_tokens": "included; no passage prefix; empty context=0",
        },
        "sentence_split": {
            "boundaries": ["Bangla danda (।)", "question mark (?)", "newline"],
            "decimal_points": "not boundaries",
            "quotes": "split after closing quote",
            "join": "sentences joined with a space within a passage; passages "
                    "joined with a blank line; original order restored",
        },
        "answer_matching": {
            "normalization": "Unicode NFC, Bangla digits -> ASCII digits, "
                             "lowercase, whitespace collapse",
            "matching": "exact normalized substring; no fuzzy matching",
            "multiple_answers": "OR over non-empty gold answers",
        },
        "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "ci": 0.95,
                      "unit": "question"},
        "runtime_seconds": round(time.time() - START, 1),
        "hardware": hardware,
        "memory": memory,
        "counters": counters,
        "runtime_by_stage_seconds": dict(runtime),
        "data_integrity": {
            "custom_qa_read": False,
            "custom_qa_used_for_filtering": False,
            "custom_qa_mixed_into_benchmark": False,
            "raw_data_overwritten": False,
            "phase3_results_overwritten": False,
        },
        "external_api_calls": False,
        "new_models": False,
    }
    (RESULTS / "phase4_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS / "phase4_results.json").write_text(json.dumps({
        "config": config,
        "results": rows,
        "m2_answer_loss_examples": examples,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(rows, config, dict(runtime), counters, memory,
                 RESULTS / "phase4_m2_answer_loss_examples.md", plot_path)

    print(f"[save] results, config, plot, examples")
    print(f"[done] {time.time() - START:.1f}s peak={memory['peak_rss_mib']:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
