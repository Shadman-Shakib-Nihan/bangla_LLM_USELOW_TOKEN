from __future__ import annotations

import json
from pathlib import Path

from src.batch2_generation import ExclusionTracker, MockProvider, build_part_c_removed_evidence, generate_part_a, generate_part_b
from src.batch2_pipeline import PassageRecord, load_exclusion_inventory, verify_protected_checksums
from src.batch2_reports import mandatory_review_rows, review_priority_rows, summarize_parts, write_qc_artifacts
from src.batch2_text import BATCH2_SEED


def _rec(pid: str, text: str, title: str = "") -> PassageRecord:
    return PassageRecord(passage_id=pid, parent_id=pid, title=title, text=text, source="miracl", is_short=False, article_key=f"miracl:{pid}")


def test_part_a_filters_answer_not_substring_and_accepts_valid() -> None:
    rec = _rec("p1", "ঢাকেশ্বরী মন্দির ঢাকা শহরে অবস্থিত।", "ঢাকেশ্বরী")
    provider = MockProvider([
        {"question": "ঢাকেশ্বরী মন্দির কোথায়?", "gold_answer": "চট্টগ্রামে", "question_type": "where"},
        {"question": "ঢাকেশ্বরী মন্দির কোন শহরে?", "gold_answer": "ঢাকা", "question_type": "where"},
    ])
    corpus = [(rec.passage_id, rec.text), ("p2", "ঢাকেশ্বরী মন্দির পুরান ঢাকায়।")]
    rows, ex = generate_part_a([rec, rec], provider=provider, corpus=corpus, rare_terms={"ঢাকেশ্বরী"}, max_items=2)
    assert len(rows) == 1
    assert rows[0]["gold_answer"] == "ঢাকা"
    assert rows[0]["needs_human_check"] is True
    assert rows[0]["my_decision"] == "" and rows[0]["my_notes"] == ""
    assert rows[0]["generation_seed"] == BATCH2_SEED
    assert any(r["reason"] == "validation_failed" for r in ex.rows)


def test_part_a_excludes_4gram_copy() -> None:
    text = "বাংলাদেশের রাজধানী ঢাকা একটি বড় শহর।"
    rec = _rec("p1", text)
    provider = MockProvider([{"question": "বাংলাদেশের রাজধানী ঢাকা একটি কী?", "gold_answer": "বড় শহর", "question_type": "what"}])
    rows, ex = generate_part_a([rec], provider=provider, corpus=[(rec.passage_id, rec.text)], rare_terms=set(text.split()))
    assert rows == []
    assert ex.rows and "shared_4gram_ge_4" in ex.rows[0]["details"]


def test_part_b_marks_hidden_evidence_candidate_and_excludes_by_default() -> None:
    rec = _rec("p1", "ঢাকা বাংলাদেশের রাজধানী।", "ঢাকা")
    passage_text = {"p1": rec.text, "p2": "ঢাকার মেয়র আতিকুল ইসলাম ছিলেন।"}

    def bm25_audit(q, top_k=20):
        return [("p2", 1.0)]

    def dense_audit(q, top_k=20):
        return [("p2", 0.9)]

    provider = MockProvider([{"question": "ঢাকার মেয়র কে ছিলেন?", "question_type": "who", "why_unanswerable": "not in hint", "near_miss_fact": "আতিকুল ইসলাম"}])
    rows, ex, audits = generate_part_b([rec], provider=provider, passage_text=passage_text, bm25_audit=bm25_audit, dense_audit=dense_audit)
    assert rows == []
    assert ex.rows[0]["reason"] == "likely_answerable"
    assert audits[0]["likely_answerable"] is True
    assert "p2" in audits[0]["hidden_evidence_candidates"]


def test_part_b_queues_likely_answerable_when_approved() -> None:
    rec = _rec("p1", "ঢাকা বাংলাদেশের রাজধানী।", "ঢাকা")
    passage_text = {"p1": rec.text, "p2": "ঢাকার মেয়র আতিকুল ইসলাম ছিলেন।"}

    def bm25_audit(q, top_k=20):
        return [("p2", 1.0)]

    provider = MockProvider([{"question": "ঢাকার মেয়র কে ছিলেন?", "question_type": "who", "why_unanswerable": "not in hint", "near_miss_fact": "আতিকুল ইসলাম"}])
    rows, ex, audits = generate_part_b([rec], provider=provider, passage_text=passage_text, bm25_audit=bm25_audit, dense_audit=bm25_audit, queue_likely_answerable=True)
    assert len(rows) == 1
    assert rows[0]["filter_flags"] == "likely_answerable"
    assert rows[0]["hidden_evidence_candidate"] == "p2"
    assert rows[0]["needs_human_check"] is True
    assert rows[0]["my_decision"] == ""
    assert ex.rows == []


def test_part_b_accepts_no_obvious_evidence_with_manual_review_defaults() -> None:
    rec = _rec("p1", "ঢাকা বাংলাদেশের রাজধানী।", "ঢাকা")
    passage_text = {"p1": rec.text, "p2": "চট্টগ্রাম বন্দরের শহর।"}

    def audit(q, top_k=20):
        return [("p2", 1.0)]

    provider = MockProvider([{"question": "ঢাকার প্রথম গোলাপি ভবনের নাম কী?", "question_type": "what", "why_unanswerable": "unknown"}])
    rows, ex, audits = generate_part_b([rec], provider=provider, passage_text=passage_text, bm25_audit=audit, dense_audit=audit)
    assert len(rows) == 1
    assert rows[0]["plausible_answerable"] == "false"
    assert rows[0]["needs_human_check"] is True
    assert rows[0]["my_decision"] == ""


def test_part_c_removed_evidence_and_remnant_exclusion() -> None:
    src = [{"question_id": "q1", "question": "রাজধানী কোথায়?", "gold_answers": ["ঢাকা"], "gold_passage_ids": ["p1"]}]
    passage_text = {"p1": "রাজধানী ঢাকা।", "p2": "রাজধানী ঢাকা ছিল।"}

    def after(query, removed, top_k=20):
        return [("p2", 1.0)]

    def remnant(query, answer, top_ids, removed):
        return ["p2"] if answer in passage_text.get(top_ids[0], "") else []

    rows, ex, audits = build_part_c_removed_evidence(src, passage_text=passage_text, bm25_audit_after_removal=after, dense_audit_after_removal=after, remnant_check_fn=remnant)
    assert rows == []
    assert ex.rows[0]["reason"] == "remnant_evidence"
    assert audits[0]["hidden_evidence_remaining"] == ["p2"]


def test_part_c_accepts_when_no_remnant() -> None:
    src = [{"question_id": "q1", "question": "রাজধানী কোথায়?", "gold_answers": ["ঢাকা"], "gold_passage_ids": ["p1"]}]
    rows, ex, audits = build_part_c_removed_evidence(src, passage_text={"p2": "অন্য পাঠ"}, dense_audit_after_removal=lambda q, r, top_k=20: [("p2", 1.0)], remnant_check_fn=lambda *a: [])
    assert len(rows) == 1
    assert rows[0]["part"] == "C"
    assert rows[0]["original_gold_passage_id"] == "p1"
    assert rows[0]["removed_evidence_ids"] == "p1"
    assert rows[0]["needs_human_check"] is True


def test_reports_and_mandatory_review(tmp_path: Path) -> None:
    part_a = [{"batch_id": "b", "question": "q", "question_type": "what", "filter_flags": "", "needs_human_check": True, "my_decision": "", "my_notes": "", "has_near_duplicate_answer": False, "self_retrieval_gold_rank_bm25": "1"}]
    part_b = [{"batch_id": "b", "question": "q2", "question_type": "what", "plausible_answerable": "false", "hidden_evidence_candidate": "", "filter_flags": "", "needs_human_check": True, "my_decision": "", "my_notes": ""}]
    summary = summarize_parts(part_a, part_b, [], [])
    assert summary["counts"]["total"] == 2
    assert summary["decisions_empty"] is True
    priorities = review_priority_rows(part_a, part_b, [])
    mandatory = mandatory_review_rows(priorities)
    assert any(r["part"] == "B" for r in mandatory)
    paths = write_qc_artifacts(tmp_path, part_a=part_a, part_b=part_b, part_c=[], exclusions=[])
    assert (tmp_path / "qc_summary.json").exists()
    assert (tmp_path / "mandatory_review.csv").exists()


def test_exclusion_tracker_jsonl(tmp_path: Path) -> None:
    tr = ExclusionTracker()
    tr.add(part="A", item_id="x", reason="test", stage="filter", details="d")
    out = tmp_path / "ex.jsonl"
    tr.to_jsonl(out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["reason"] == "test"


def test_real_protected_checksums_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    mismatches = verify_protected_checksums(root, root / "data/custom_batch2/protected_checksums.json")
    assert mismatches == []


def test_real_exclusion_inventory_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    d = load_exclusion_inventory(root / "data/custom_batch2/source_inventory.json")
    assert len(d["frozen_ids"]) == 56
    assert len(d["tydi_tuning_ids"]) == 1000
    assert len(d["passage_ids"]) > 0
