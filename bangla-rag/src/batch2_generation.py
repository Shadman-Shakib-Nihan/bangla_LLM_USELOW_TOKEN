"""Mockable generation loops and exclusion tracking for Batch 2.

No live Gemini calls occur unless a real provider is passed explicitly.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.batch2_pipeline import (
    PART_A_FIELDS,
    PART_B_FIELDS,
    PART_C_FIELDS,
    PassageRecord,
    gold_rank,
    make_part_a_row,
    make_part_b_row,
    make_part_c_row,
    validate_part_a_candidate,
    validate_part_b_no_hidden_evidence,
)
from src.batch2_prompts import SYSTEM_INSTRUCTION, format_part_a, format_part_b, format_part_c
from src.batch2_text import (
    BATCH2_SEED,
    QUESTION_TYPES,
    contains_text,
    infer_question_type,
    join_flags,
    normalize_digits,
    normalize_space,
    validate_question_type,
)


class GenerationProvider(Protocol):
    model_id: str

    def generate_json(self, prompt: str, schema: dict[str, Any] | None = None, system: str | None = None) -> dict[str, Any]:
        ...


@dataclass
class MockProvider:
    responses: list[dict[str, Any]]
    model_id: str = "mock-batch2"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_json(self, prompt: str, schema: dict[str, Any] | None = None, system: str | None = None) -> dict[str, Any]:
        idx = len(self.calls)
        self.calls.append({"prompt": prompt, "schema": schema, "system": system})
        if idx >= len(self.responses):
            return {}
        return self.responses[idx]


@dataclass
class ExclusionTracker:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *, part: str, item_id: str, reason: str, stage: str, details: str = "") -> None:
        self.rows.append({
            "part": part,
            "item_id": item_id,
            "reason": reason,
            "stage": stage,
            "details": details,
        })

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_type(value: str | None, fallback_question: str = "") -> str:
    try:
        return validate_question_type(value)
    except Exception:
        return infer_question_type(fallback_question)


def generate_part_a(
    candidates: list[PassageRecord],
    *,
    provider: GenerationProvider,
    corpus: list[tuple[str, str]],
    rare_terms: set[str],
    batch_id: str = "batch2",
    seed: int = BATCH2_SEED,
    bm25_audit=None,
    dense_audit=None,
    max_items: int | None = None,
    alt_gold_lookup=None,
) -> tuple[list[dict[str, Any]], ExclusionTracker]:
    accepted: list[dict[str, Any]] = []
    exclusions = ExclusionTracker()
    seen_answers: list[str] = []
    limit = len(candidates) if max_items is None else min(max_items, len(candidates))
    for i, rec in enumerate(candidates[:limit]):
        prompt = format_part_a(rec.passage_id, rec.title, rec.text)
        raw = provider.generate_json(prompt, system=SYSTEM_INSTRUCTION)
        question = normalize_space(raw.get("question", ""))
        answer = normalize_space(raw.get("gold_answer", ""))
        qtype = _safe_type(raw.get("question_type"), question)
        item_id = f"{batch_id}_A_{i:04d}_{rec.passage_id}"
        if not question or not answer:
            exclusions.add(part="A", item_id=item_id, reason="empty_generation", stage="parse", details=rec.passage_id)
            continue
        ok, flags, alt_ids, ngram = validate_part_a_candidate(
            question=question,
            gold_answer=answer,
            gold_passage_id=rec.passage_id,
            gold_passage_text=rec.text,
            question_type=qtype,
            corpus=corpus,
            rare_terms=rare_terms,
            existing_answers=seen_answers,
            alt_gold_lookup=alt_gold_lookup,
        )
        bm25_rank = dense_rank = None
        if bm25_audit is not None:
            bm25_rank = gold_rank(bm25_audit(question), rec.passage_id)
        if dense_audit is not None:
            dense_rank = gold_rank(dense_audit(question), rec.passage_id)
        row = make_part_a_row(
            batch_id=batch_id,
            question=question,
            question_type=qtype,
            gold_answer=answer,
            gold_passage_id=rec.passage_id,
            gold_passage_text=rec.text,
            alt_gold_ids=";".join(alt_ids),
            answer_is_substring=contains_text(rec.text, answer),
            max_shared_word_4gram=ngram,
            has_near_duplicate_answer="near_duplicate_answer" in flags,
            self_retrieval_gold_rank_bm25=bm25_rank if bm25_rank is not None else "",
            self_retrieval_gold_rank_dense=dense_rank if dense_rank is not None else "",
            generation_model=provider.model_id,
            generation_seed=seed,
            filter_flags=join_flags(flags),
        )
        if not ok:
            exclusions.add(part="A", item_id=item_id, reason="validation_failed", stage="filter", details=join_flags(flags))
            continue
        seen_answers.append(answer)
        accepted.append(row)
    return accepted, exclusions


def generate_part_b(
    candidates: list[PassageRecord],
    *,
    provider: GenerationProvider,
    passage_text: dict[str, str],
    bm25_audit,
    dense_audit,
    batch_id: str = "batch2",
    seed: int = BATCH2_SEED,
    top_k: int = 20,
    max_items: int | None = None,
    queue_likely_answerable: bool = False,
    queue_manual_evidence_review: bool = False,
) -> tuple[list[dict[str, Any]], ExclusionTracker, list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    exclusions = ExclusionTracker()
    audits: list[dict[str, Any]] = []
    limit = len(candidates) if max_items is None else min(max_items, len(candidates))
    for i, rec in enumerate(candidates[:limit]):
        prompt = format_part_b(rec.passage_id, rec.title, rec.text)
        raw = provider.generate_json(prompt, system=SYSTEM_INSTRUCTION)
        question = normalize_space(raw.get("question", ""))
        qtype = _safe_type(raw.get("question_type"), question)
        item_id = f"{batch_id}_B_{i:04d}_{rec.passage_id}"
        if not question:
            exclusions.add(part="B", item_id=item_id, reason="empty_generation", stage="parse", details=rec.passage_id)
            continue
        bm25_hits = bm25_audit(question, top_k=top_k)
        dense_hits = dense_audit(question, top_k=top_k)
        near_miss = normalize_space(raw.get("near_miss_fact", ""))
        status_bm25, ev_bm25 = validate_part_b_no_hidden_evidence(
            question, bm25_hits, passage_text, likely_answer_terms=[near_miss] if near_miss else []
        )
        status_dense, ev_dense = validate_part_b_no_hidden_evidence(
            question, dense_hits, passage_text, likely_answer_terms=[near_miss] if near_miss else []
        )
        likely = status_bm25 == "candidate_answerable" or status_dense == "candidate_answerable"
        manual_review = (not likely) and ("manual_evidence_review" in ev_bm25 or "manual_evidence_review" in ev_dense)
        hidden = list(dict.fromkeys([x for x in ev_bm25 + ev_dense if x != "manual_evidence_review"]))
        if manual_review and "manual_evidence_review" not in hidden:
            hidden.append("manual_evidence_review")
        audit = {
            "item_id": item_id,
            "question": question,
            "bm25_top": [p for p, _ in bm25_hits],
            "dense_top": [p for p, _ in dense_hits],
            "hidden_evidence_candidates": hidden,
            "likely_answerable": likely,
            "manual_evidence_review": manual_review,
        }
        audits.append(audit)
        flags = []
        if likely:
            flags.append("likely_answerable")
        if manual_review:
            flags.append("manual_evidence_review")
        row = make_part_b_row(
            batch_id=batch_id,
            question=question,
            question_type=qtype,
            plausible_answerable="true" if likely else "false",
            source_passage_hint=rec.passage_id,
            why_not_answered_from_corpus=normalize_space(raw.get("why_unanswerable", "")),
            top_retrieved_passage_ids=";".join([p for p, _ in dense_hits[:10]]),
            hidden_evidence_candidate=";".join(hidden),
            generation_model=provider.model_id,
            generation_seed=seed,
            filter_flags=join_flags(flags),
        )
        if likely and not queue_likely_answerable:
            exclusions.add(part="B", item_id=item_id, reason="likely_answerable", stage="retrieval_audit", details=";".join(hidden))
            continue
        if manual_review and not queue_manual_evidence_review and False:
            # placeholder for optional behavior; approved run queues manual review.
            pass
        accepted.append(row)
    return accepted, exclusions, audits


def build_part_c_removed_evidence(
    question_rows: list[dict[str, Any]],
    *,
    passage_text: dict[str, str],
    provider: GenerationProvider | None = None,
    batch_id: str = "batch2",
    seed: int = BATCH2_SEED,
    bm25_audit_after_removal=None,
    dense_audit_after_removal=None,
    remnant_check_fn=None,
    max_items: int | None = None,
) -> tuple[list[dict[str, Any]], ExclusionTracker, list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    exclusions = ExclusionTracker()
    audits: list[dict[str, Any]] = []
    limit = len(question_rows) if max_items is None else min(max_items, len(question_rows))
    for i, src in enumerate(question_rows[:limit]):
        qid = str(src.get("question_id", f"src_{i}"))
        question = normalize_space(src.get("question", ""))
        answers = src.get("gold_answers") or []
        answer = normalize_space(answers[0] if answers else "")
        gold_ids = [str(x) for x in (src.get("gold_passage_ids") or [])]
        item_id = f"{batch_id}_C_{i:04d}_{qid}"
        if not question or not gold_ids:
            exclusions.add(part="C", item_id=item_id, reason="missing_source_fields", stage="source")
            continue
        qtype = infer_question_type(question)
        removal_reason = "remove_gold_evidence"
        if provider is not None:
            raw = provider.generate_json(format_part_c(question, answer, qtype), system=SYSTEM_INSTRUCTION)
            question = normalize_space(raw.get("question", question)) or question
            qtype = _safe_type(raw.get("question_type"), question)
            removal_reason = normalize_space(raw.get("removal_reason", removal_reason)) or removal_reason
        removed = gold_ids
        after_bm25 = bm25_audit_after_removal(question, removed, top_k=20) if bm25_audit_after_removal else []
        after_dense = dense_audit_after_removal(question, removed, top_k=20) if dense_audit_after_removal else []
        remnants: list[str] = []
        if remnant_check_fn is not None:
            remnants = remnant_check_fn(question, answer, [p for p, _ in after_dense or after_bm25], removed)
        audit = {
            "item_id": item_id,
            "source_question_id": qid,
            "removed_evidence_ids": removed,
            "bm25_top_after_removal": [p for p, _ in after_bm25],
            "dense_top_after_removal": [p for p, _ in after_dense],
            "hidden_evidence_remaining": remnants,
        }
        audits.append(audit)
        row = make_part_c_row(
            batch_id=batch_id,
            question=question,
            question_type=qtype,
            original_gold_answer=answer,
            original_gold_passage_id=gold_ids[0],
            removed_evidence_ids=";".join(removed),
            removal_reason=removal_reason,
            top_retrieved_after_removal=";".join([p for p, _ in (after_dense or after_bm25)[:10]]),
            hidden_evidence_remaining=";".join(remnants),
            generation_model=(provider.model_id if provider else "deterministic"),
            generation_seed=seed,
            filter_flags="remnant_evidence" if remnants else "",
        )
        if remnants:
            exclusions.add(part="C", item_id=item_id, reason="remnant_evidence", stage="remnant_check", details=";".join(remnants))
            continue
        accepted.append(row)
    return accepted, exclusions, audits
