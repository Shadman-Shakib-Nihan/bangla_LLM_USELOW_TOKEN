"""Batch 2 deterministic pipeline scaffolding.

This module intentionally avoids live Gemini calls and RAG experiment changes.
It provides inventory checks, corpus sampling, filtering/audit primitives, and
CSV/report writers with fixed seeds and stable ordering.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.batch2_text import (
    BATCH2_SEED,
    E5_MODEL,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    QUESTION_TYPES,
    answer_satisfies_type,
    build_rare_terms,
    canonical_json,
    compact_contains,
    compute_alt_gold_ids,
    contains_text,
    content_terms,
    infer_question_type,
    is_name_heavy,
    is_time_sensitive_undated,
    join_flags,
    max_shared_word_ngram,
    normalize_digits,
    normalize_space,
    question_rare_term_ok,
    question_terms,
    sha256_file,
    sha256_json,
    sha256_text,
    validate_question_type,
)
from src.retrieval import BM25Index, aggregate_parent_scores, build_parent_index, rank_parents

PART_A_FIELDS = [
    "batch_id", "part", "question_type", "question", "gold_answer", "gold_passage_id",
    "gold_passage_text", "alt_gold_ids", "answer_is_substring", "max_shared_word_4gram",
    "has_near_duplicate_answer", "self_retrieval_gold_rank_bm25",
    "self_retrieval_gold_rank_dense", "needs_human_check", "my_decision", "my_notes",
    "generation_model", "generation_seed", "filter_flags",
]
PART_B_FIELDS = [
    "batch_id", "part", "question_type", "question", "plausible_answerable",
    "source_passage_hint", "why_not_answered_from_corpus", "top_retrieved_passage_ids",
    "hidden_evidence_candidate", "needs_human_check", "my_decision", "my_notes",
    "generation_model", "generation_seed", "filter_flags",
]
PART_C_FIELDS = [
    "batch_id", "part", "question_type", "question", "original_gold_answer",
    "original_gold_passage_id", "removed_evidence_ids", "removal_reason",
    "top_retrieved_after_removal", "hidden_evidence_remaining", "needs_human_check",
    "my_decision", "my_notes", "generation_model", "generation_seed", "filter_flags",
]


@dataclass(frozen=True)
class PassageRecord:
    passage_id: str
    parent_id: str
    title: str
    text: str
    source: str
    is_short: bool
    article_key: str


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                out[key] = value
            writer.writerow(out)


def make_row(fieldnames: list[str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in fieldnames}
    row.update(values)
    row["needs_human_check"] = True
    row.setdefault("my_decision", "")
    row.setdefault("my_notes", "")
    row.setdefault("filter_flags", "")
    return row


def article_key(record: dict[str, Any]) -> str:
    pid = str(record.get("passage_id", ""))
    source = str(record.get("source", ""))
    title = normalize_space(record.get("title", ""))
    if source == "miracl" and "#" in pid:
        return f"miracl:{pid.split('#', 1)[0]}"
    if title:
        return f"{source}:{title.casefold()}"
    return f"{source}:{pid.split('#', 1)[0]}"


def load_passages(corpus_path: str | Path) -> list[PassageRecord]:
    records: list[PassageRecord] = []
    for row in read_jsonl(corpus_path):
        records.append(
            PassageRecord(
                passage_id=str(row["passage_id"]),
                parent_id=str(row.get("parent_id", row["passage_id"])),
                title=normalize_space(row.get("title", "")),
                text=normalize_space(row.get("text", "")),
                source=str(row.get("source", "")),
                is_short=bool(row.get("is_short", False)),
                article_key=article_key(row),
            )
        )
    return records


def load_units(units_path: str | Path) -> tuple[list[str], list[str], list[PassageRecord]]:
    texts: list[str] = []
    parents: list[str] = []
    records: list[PassageRecord] = []
    for row in read_jsonl(units_path):
        rec = PassageRecord(
            passage_id=str(row["passage_id"]),
            parent_id=str(row.get("parent_id", row["passage_id"])),
            title=normalize_space(row.get("title", "")),
            text=normalize_space(row.get("text", "")),
            source=str(row.get("source", "")),
            is_short=bool(row.get("is_short", False)),
            article_key=article_key(row),
        )
        texts.append(rec.text)
        parents.append(rec.parent_id)
        records.append(rec)
    return texts, parents, records


def load_exclusion_inventory(inventory_path: str | Path) -> dict[str, set[str]]:
    inv = read_json(inventory_path)
    excl = inv.get("exclusion_inventory") or inv.get("exclusions") or {}
    frozen = inv.get("frozen_reviewed_items") or {}
    tydi = inv.get("tydi_split") or {}
    return {
        "question_ids": set(excl.get("question_ids", [])),
        "normalized_questions": set(excl.get("normalized_question_texts") or excl.get("normalized_questions", [])),
        "answers": set(excl.get("answers", [])),
        "passage_ids": set(excl.get("passage_ids", [])),
        "frozen_ids": set(frozen.get("ids", [])),
        "tydi_tuning_ids": set(tydi.get("tuning_ids") or tydi.get("excluded_tuning_ids", [])),
    }


def verify_protected_checksums(project_root: str | Path, checksums_path: str | Path) -> list[str]:
    root = Path(project_root)
    data = read_json(checksums_path)
    mismatches: list[str] = []
    for item in data.get("protected_files", []):
        path = root / item["path"]
        if not path.exists():
            if item.get("exists", True):
                mismatches.append(f"missing:{item['path']}")
            continue
        actual = sha256_file(path)
        if actual != item["sha256"]:
            mismatches.append(f"sha256:{item['path']}")
    for path, expected in data.get("phase4_frozen_checksums_sha256", {}).items():
        p = root / path
        if not p.exists():
            mismatches.append(f"missing:{path}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            mismatches.append(f"sha256:{path}")
    return mismatches


class CorpusSearch:
    """Precomputed corpus view for fast alt-gold lookup; preserves compute_alt_gold_ids semantics."""

    def __init__(self, corpus: list[tuple[str, str]], rare_terms: set[str] | None = None) -> None:
        self.rare_terms = set(rare_terms or ())
        self.rows = []
        for pid, text in corpus:
            norm = normalize_space(text)
            self.rows.append(
                (
                    pid,
                    norm,
                    normalize_space(normalize_digits(text)),
                    text,
                )
            )

    def alt_gold_ids(self, answer: str | None, question: str | None, gold_passage_id: str, max_count: int = 10) -> tuple[list[str], int]:
        answer_text = normalize_space(answer)
        if not answer_text:
            return [], 0
        answer_norm = normalize_space(normalize_digits(answer))
        qterms = question_terms(question)
        matches: list[str] = []
        for pid, norm, norm_digits, original in self.rows:
            if pid == gold_passage_id:
                continue
            if answer_text not in norm and (not answer_norm or answer_norm not in norm_digits):
                continue
            pterms = content_terms(original)
            if self.rare_terms and not any(t in self.rare_terms and t in pterms for t in qterms):
                continue
            if not self.rare_terms and not (qterms & pterms):
                continue
            matches.append(pid)
        deduped = list(dict.fromkeys(matches))
        return deduped[:max_count], len(deduped)

def build_e5_length_counter(texts: list[str], tokenizer: Any, batch_size: int = 128) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = [E5_PASSAGE_PREFIX + t for t in texts[start : start + batch_size]]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False, return_length=True)
        raw = enc["length"]
        lengths.extend(int(x) for x in raw)
    return lengths


def sample_passages_for_part_a(
    passages: list[PassageRecord],
    e5_lengths: dict[str, int],
    exclusions: dict[str, set[str]],
    *,
    target: int,
    min_tokens: int = 40,
    max_tokens: int = 300,
    source: str = "miracl",
    seed: int = BATCH2_SEED,
) -> list[PassageRecord]:
    rng = random.Random(seed)
    by_article: dict[str, list[PassageRecord]] = defaultdict(list)
    for rec in passages:
        if rec.source != source or rec.is_short:
            continue
        if rec.passage_id in exclusions["passage_ids"] or rec.parent_id in exclusions["passage_ids"]:
            continue
        n = e5_lengths.get(rec.passage_id)
        if n is None or n < min_tokens or n > max_tokens:
            continue
        if not rec.text:
            continue
        by_article[rec.article_key].append(rec)

    chosen: list[PassageRecord] = []
    for recs in by_article.values():
        recs.sort(key=lambda r: (r.title, r.passage_id))
        chosen.append(recs[0])
    rng.shuffle(chosen)
    chosen.sort(key=lambda r: sha256_text(f"{seed}:{r.passage_id}"))
    return chosen[:target]


def sample_tydi_questions_for_part_c(
    questions: list[dict[str, Any]],
    exclusions: dict[str, set[str]],
    *,
    target: int,
    seed: int = BATCH2_SEED,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = []
    for row in questions:
        qid = str(row["question_id"])
        # Part C reuses TyDi heldout questions with evidence removed, so the
        # global question_ids set (which includes all TyDi ids already seen in
        # other datasets) must not filter them. Only tuning/frozen ids are
        # unavailable for Part C source, per the inventory design.
        if qid in exclusions["tydi_tuning_ids"]:
            continue
        if qid in exclusions["frozen_ids"]:
            continue
        if not row.get("question") or not row.get("gold_passage_ids"):
            continue
        candidates.append(row)
    rng.shuffle(candidates)
    candidates.sort(key=lambda r: sha256_text(f"{seed}:{r['question_id']}"))
    return candidates[:target]


def _resolve_pidx(parents: list[str] | Any):
    if hasattr(parents, "parent_ids") and hasattr(parents, "unit_to_parent"):
        return parents
    return build_parent_index(parents)


def bm25_search(index: BM25Index, parents: list[str] | Any, query: str, top_k: int = 20) -> list[tuple[str, float]]:
    from src.retrieval import tokenize

    unit_scores = index.score(tokenize(query), 1.5, 0.75)
    pidx = _resolve_pidx(parents)
    parent_scores = aggregate_parent_scores(unit_scores, pidx)
    ranked = rank_parents(parent_scores, pidx, top_k=top_k, only_positive=True)
    score_map = {pid: float(parent_scores[i]) for i, pid in enumerate(pidx.parent_ids)}
    return [(pid, score_map.get(pid, float("nan"))) for pid in ranked]


def dense_search(
    unit_embeddings: np.ndarray,
    query_vector: np.ndarray,
    parents: list[str] | Any,
    top_k: int = 20,
    candidate_units: int = 2000,
) -> list[tuple[str, float]]:
    scores = unit_embeddings @ query_vector.astype(np.float32)
    n_units = min(candidate_units, scores.shape[0])
    top_units = np.argpartition(-scores, n_units - 1)[:n_units]
    top_units = top_units[np.argsort(-scores[top_units])]
    pidx = _resolve_pidx(parents)
    parent_scores = np.full(pidx.n_parents, -np.inf, dtype=np.float32)
    np.maximum.at(parent_scores, pidx.unit_to_parent[top_units], scores[top_units])
    ranked = rank_parents(parent_scores, pidx, top_k=top_k, only_positive=True)
    score_map = {pid: float(parent_scores[i]) for i, pid in enumerate(pidx.parent_ids)}
    return [(pid, score_map.get(pid, float("-inf"))) for pid in ranked]


def gold_rank(ranked: list[tuple[str, float]], gold_id: str) -> int | None:
    for idx, (pid, _) in enumerate(ranked, start=1):
        if pid == gold_id:
            return idx
    return None


def validate_part_a_candidate(
    *,
    question: str,
    gold_answer: str,
    gold_passage_id: str,
    gold_passage_text: str,
    question_type: str,
    corpus: list[tuple[str, str]],
    rare_terms: set[str],
    existing_answers: list[str],
    alt_gold_lookup=None,
) -> tuple[bool, list[str], list[str], int]:
    flags: list[str] = []
    if not contains_text(gold_passage_text, gold_answer):
        flags.append("answer_not_substring")
    if not answer_satisfies_type(question_type, question, gold_answer):
        flags.append("answer_type_mismatch")
    ngram = max_shared_word_ngram(question, gold_passage_text, 4)
    if ngram >= 4:
        flags.append("shared_4gram_ge_4")
    if is_time_sensitive_undated(question):
        flags.append("time_sensitive_undated")
    if not question_rare_term_ok(question, gold_passage_text, rare_terms):
        flags.append("no_rare_question_term")
    if is_name_heavy(question):
        flags.append("name_heavy")
    norm_answer = normalize_space(normalize_digits(gold_answer)).casefold()
    near_dup = any(
        normalize_space(normalize_digits(old)).casefold() == norm_answer for old in existing_answers
    )
    if near_dup:
        flags.append("near_duplicate_answer")
    if alt_gold_lookup is None:
        alt_ids, _ = compute_alt_gold_ids(gold_answer, question, gold_passage_id, corpus, rare_terms)
    else:
        alt_ids, _ = alt_gold_lookup(gold_answer, question, gold_passage_id)
    hard_fail = {
        "answer_not_substring", "shared_4gram_ge_4", "time_sensitive_undated", "no_rare_question_term"
    }
    ok = not (hard_fail & set(flags))
    return ok, flags, alt_ids, ngram


def validate_part_b_no_hidden_evidence(
    question: str,
    top_hits: list[tuple[str, float]],
    passage_text: dict[str, str],
    *,
    likely_answer_terms: Iterable[str] = (),
) -> tuple[str, list[str]]:
    flags: list[str] = []
    candidates: list[str] = []
    terms = [normalize_space(t) for t in likely_answer_terms if normalize_space(t)]
    for pid, _ in top_hits:
        text = passage_text.get(pid, "")
        if terms and any(compact_contains(text, t) for t in terms):
            candidates.append(pid)
            continue
        # Conservative lexical heuristic: question key terms heavily present but exact answer unknown.
        q_overlap = len(question_terms(question) & content_terms(text))
        if q_overlap >= 4 and terms:
            flags.append("manual_evidence_review")
    status = "candidate_answerable" if candidates else "no_obvious_hidden_evidence"
    return status, candidates or flags


def make_part_a_row(**kwargs: Any) -> dict[str, Any]:
    return make_row(PART_A_FIELDS, part="A", **kwargs)


def make_part_b_row(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("plausible_answerable", "")
    return make_row(PART_B_FIELDS, part="B", **kwargs)


def make_part_c_row(**kwargs: Any) -> dict[str, Any]:
    return make_row(PART_C_FIELDS, part="C", **kwargs)


def write_part_csvs(
    output_dir: str | Path,
    *,
    part_a: list[dict[str, Any]] | None = None,
    part_b: list[dict[str, Any]] | None = None,
    part_c: list[dict[str, Any]] | None = None,
) -> None:
    out = Path(output_dir)
    if part_a is not None:
        write_csv(out / "part_a_answerable.csv", PART_A_FIELDS, part_a)
    if part_b is not None:
        write_csv(out / "part_b_unanswerable.csv", PART_B_FIELDS, part_b)
    if part_c is not None:
        write_csv(out / "part_c_removed_evidence.csv", PART_C_FIELDS, part_c)


def pilot_selection(
    answerable_candidates: list[PassageRecord],
    unanswerable_candidates: list[PassageRecord] | None = None,
    *,
    n_answerable: int = 5,
    n_unanswerable: int = 5,
    seed: int = BATCH2_SEED,
) -> dict[str, list[PassageRecord]]:
    rng = random.Random(seed)
    a = list(answerable_candidates)
    b = list(unanswerable_candidates or answerable_candidates)
    rng.shuffle(a)
    rng.shuffle(b)
    a.sort(key=lambda r: r.passage_id)
    b.sort(key=lambda r: r.passage_id)
    rng2 = random.Random(seed + 1)
    rng2.shuffle(a)
    rng2.shuffle(b)
    return {"answerable": a[:n_answerable], "unanswerable": b[:n_unanswerable]}
