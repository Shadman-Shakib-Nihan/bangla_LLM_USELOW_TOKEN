from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.batch2_pipeline import (
    PART_A_FIELDS,
    PART_B_FIELDS,
    PART_C_FIELDS,
    bm25_search,
    dense_search,
    gold_rank,
    make_part_b_row,
    make_part_c_row,
    pilot_selection,
    sample_tydi_questions_for_part_c,
    validate_part_a_candidate,
    validate_part_b_no_hidden_evidence,
    write_part_csvs,
    PassageRecord,
)
from src.retrieval import BM25Index


def test_part_csv_schemas_and_human_fields(tmp_path: Path) -> None:
    a = [dict.fromkeys(PART_A_FIELDS, "")]
    a[0].update({"batch_id": "b", "part": "A", "question": "q", "question_type": "what", "needs_human_check": True, "my_decision": "", "my_notes": ""})
    b = [make_part_b_row(batch_id="b", question="q", question_type="what", generation_model="m", generation_seed=20261110)]
    c = [make_part_c_row(batch_id="b", question="q", question_type="what", generation_model="m", generation_seed=20261110)]
    write_part_csvs(tmp_path, part_a=a, part_b=b, part_c=c)
    for name, fields in [("part_a_answerable.csv", PART_A_FIELDS), ("part_b_unanswerable.csv", PART_B_FIELDS), ("part_c_removed_evidence.csv", PART_C_FIELDS)]:
        header = (tmp_path / name).read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == fields
        assert "my_decision" in header and "my_notes" in header


def test_pilot_selection_deterministic() -> None:
    recs = [PassageRecord(str(i), str(i), "", f"text {i}", "miracl", False, f"a{i}") for i in range(20)]
    a = pilot_selection(recs, recs, seed=20261110)
    b = pilot_selection(recs, recs, seed=20261110)
    assert [r.passage_id for r in a["answerable"]] == [r.passage_id for r in b["answerable"]]
    assert len(a["answerable"]) == 5 and len(a["unanswerable"]) == 5


def test_sample_tydi_excludes_tuning_and_frozen() -> None:
    questions = [
        {"question_id": "tune1", "question": "q", "gold_passage_ids": ["p"]},
        {"question_id": "freeze1", "question": "q", "gold_passage_ids": ["p"]},
        {"question_id": "ok1", "question": "q", "gold_passage_ids": ["p"]},
    ]
    exc = {"question_ids": set(), "frozen_ids": {"freeze1"}, "tydi_tuning_ids": {"tune1"}, "passage_ids": set()}
    got = sample_tydi_questions_for_part_c(questions, exc, target=10, seed=20261110)
    assert [r["question_id"] for r in got] == ["ok1"]


def test_sample_tydi_keeps_heldout_in_global_question_ids() -> None:
    # The global question_ids set includes all TyDi ids (seen in other datasets);
    # Part C must still be able to reuse heldout TyDi questions as removal source.
    questions = [
        {"question_id": "heldout1", "question": "q", "gold_passage_ids": ["p"]},
        {"question_id": "tune1", "question": "q", "gold_passage_ids": ["p"]},
    ]
    exc = {
        "question_ids": {"heldout1", "tune1"},
        "frozen_ids": set(),
        "tydi_tuning_ids": {"tune1"},
        "passage_ids": set(),
    }
    got = sample_tydi_questions_for_part_c(questions, exc, target=10, seed=20261110)
    assert [r["question_id"] for r in got] == ["heldout1"]


def test_bm25_and_dense_search_shapes() -> None:
    texts = ["ঢাকা বাংলাদেশের রাজধানী", "চট্টগ্রাম বন্দর শহর", "ঢাকা বড় শহর"]
    parents = ["p1", "p2", "p1"]
    idx = BM25Index(texts)
    bm = bm25_search(idx, parents, "ঢাকা রাজধানী", top_k=2)
    assert bm[0][0] == "p1"
    assert gold_rank(bm, "p1") == 1
    emb = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32)
    q = np.array([1, 0, 0], dtype=np.float32)
    de = dense_search(emb, q, parents, top_k=2)
    assert de[0][0] == "p1"


def test_validate_part_a_edge_cases() -> None:
    corpus = [("g1", "ঢাকেশ্বরী মন্দির ঢাকা শহরে অবস্থিত।"), ("g2", "ঢাকেশ্বরী মন্দিরটি ঢাকা অঞ্চলে।")]
    ok, flags, alt, ngram = validate_part_a_candidate(
        question="ঢাকেশ্বরী মন্দির কোথায়?", gold_answer="ঢাকা", gold_passage_id="g1",
        gold_passage_text=corpus[0][1], question_type="where", corpus=corpus,
        rare_terms={"ঢাকেশ্বরী"}, existing_answers=[],
    )
    assert ok
    assert "g2" in alt
    ok2, flags2, _, _ = validate_part_a_candidate(
        question="ঢাকেশ্বরী মন্দির ঢাকায় অবস্থিত কেন?", gold_answer="না", gold_passage_id="g1",
        gold_passage_text=corpus[0][1], question_type="where", corpus=corpus,
        rare_terms={"ঢাকেশ্বরী"}, existing_answers=[],
    )
    assert not ok2
    assert "answer_not_substring" in flags2


def test_part_b_hidden_evidence_heuristic() -> None:
    passage_text = {"p1": "ঢাকা রাজধানী।", "p2": "মেয়র আতিকুল ইসলাম ঢাকা।"}
    status, ev = validate_part_b_no_hidden_evidence("মেয়র কে?", [("p2", 1.0)], passage_text, likely_answer_terms=["আতিকুল"])
    assert status == "candidate_answerable"
    assert ev == ["p2"]


def test_real_inventory_exists_and_has_no_protected_mismatch() -> None:
    root = Path(__file__).resolve().parents[1]
    inv = json.loads((root / "data/custom_batch2/source_inventory.json").read_text(encoding="utf-8"))
    assert inv["seed"] == 20261110
    assert len(inv["frozen_reviewed_items"]["ids"]) == 56


def test_parent_index_reuse() -> None:
    import numpy as np
    import src.batch2_pipeline as bp
    orig = bp.build_parent_index
    counts = {'n': 0}
    def wrapped(parents):
        counts['n'] += 1
        return orig(parents)
    bp.build_parent_index = wrapped
    try:
        parents = ['a', 'b', 'a', 'b', 'c']
        pidx = orig(parents)
        emb = np.eye(5, dtype=np.float32)
        q = np.ones(5, dtype=np.float32)
        for _ in range(3):
            bp.dense_search(emb, q, parents, top_k=2)
            bp.dense_search(emb, q, pidx, top_k=2)
        assert counts['n'] == 3
    finally:
        bp.build_parent_index = orig
