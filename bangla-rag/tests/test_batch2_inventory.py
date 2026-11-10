from __future__ import annotations

import json
from pathlib import Path

from src.batch2_pipeline import make_part_a_row, verify_protected_checksums, write_csv, PART_A_FIELDS
from src.batch2_text import sha256_file


def test_verify_protected_checksums_reports_missing_and_mismatch(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    p = root / "data/protected.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    checks = {
        "protected_files": [
            {"path": "data/protected.csv", "sha256": sha256_file(p), "exists": True},
            {"path": "data/missing.csv", "sha256": "deadbeef", "exists": True},
        ],
        "phase4_frozen_checksums_sha256": {},
    }
    cpath = root / "checksums.json"
    cpath.write_text(json.dumps(checks), encoding="utf-8")
    assert verify_protected_checksums(root, cpath) == ["missing:data/missing.csv"]

    checks["protected_files"][1]["exists"] = False
    cpath.write_text(json.dumps(checks), encoding="utf-8")
    assert verify_protected_checksums(root, cpath) == []
    p.write_text("changed\n", encoding="utf-8")
    assert verify_protected_checksums(root, cpath) == ["sha256:data/protected.csv"]


def test_write_csv_schema_and_human_fields(tmp_path) -> None:
    row = make_part_a_row(
        batch_id="pilot", question="q", question_type="what", gold_answer="a",
        gold_passage_id="p1", gold_passage_text="text", generation_model="m",
        generation_seed=20261110, answer_is_substring=True, max_shared_word_4gram=0,
        has_near_duplicate_answer=False, alt_gold_ids="",
        self_retrieval_gold_rank_bm25=1, self_retrieval_gold_rank_dense=1,
        filter_flags="",
    )
    assert row["needs_human_check"] is True
    assert row["my_decision"] == ""
    assert row["my_notes"] == ""
    out = tmp_path / "part_a.csv"
    write_csv(out, PART_A_FIELDS, [row])
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("batch_id,part,question_type")
    assert "needs_human_check" in text
