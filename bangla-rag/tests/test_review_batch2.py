from __future__ import annotations

from pathlib import Path

import pytest

import review_batch2


def _write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


A_COLS = ["batch_id", "part", "question", "gold_answer", "gold_passage_id", "needs_human_check", "my_decision", "my_notes"]


def _setup(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / review_batch2.PARTS["A"]).parent.mkdir(parents=True, exist_ok=True)
    rows_a = [
        {"batch_id": "batch2full", "part": "A", "question": "প্রশ্ন ১", "gold_answer": "উত্তর ১", "gold_passage_id": "p1", "needs_human_check": "True", "my_decision": "", "my_notes": ""},
        {"batch_id": "batch2full", "part": "A", "question": "প্রশ্ন ২", "gold_answer": "উত্তর ২", "gold_passage_id": "p2", "needs_human_check": "True", "my_decision": "keep", "my_notes": ""},
    ]
    _write(tmp_path / review_batch2.PARTS["A"], A_COLS, rows_a)
    _write(tmp_path / review_batch2.PARTS["B"], A_COLS, [dict(rows_a[0], part="B")])
    _write(tmp_path / review_batch2.PARTS["C"], A_COLS, [dict(rows_a[0], part="C")])
    monkeypatch.setattr(review_batch2, "DATA", tmp_path)
    return tmp_path


def test_counts_and_status(tmp_path: Path, monkeypatch, capsys) -> None:
    _setup(tmp_path, monkeypatch)
    review_batch2.status()
    out = capsys.readouterr().out
    assert "Part A: total=2 reviewed=1 keep=1 fix=0 reject=0 pending=1" in out
    assert "Overall: total=4" in out


def test_apply_decision_requires_note() -> None:
    row = {"my_decision": "", "my_notes": ""}
    with pytest.raises(review_batch2.ReviewError):
        review_batch2.apply_decision(row, "reject", "")
    review_batch2.apply_decision(row, "reject", "duplicate")
    assert row["my_decision"] == "reject"
    assert row["my_notes"] == "duplicate"


def test_save_creates_backup_and_preserves_bangla(tmp_path: Path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch)
    path = tmp_path / review_batch2.PARTS["A"]
    columns, rows = review_batch2.load_csv(path)
    review_batch2.apply_decision(rows[0], "fix", "fix answer")
    dest = review_batch2.backup(path)
    review_batch2.save_csv(path, columns, rows)
    assert dest.exists()
    _, reloaded = review_batch2.load_csv(path)
    assert reloaded[0]["my_decision"] == "fix"
    assert reloaded[0]["my_notes"] == "fix answer"
    assert reloaded[0]["question"] == "প্রশ্ন ১"


def test_select_rows_pending_and_limit() -> None:
    rows = [
        {"my_decision": ""},
        {"my_decision": "keep"},
        {"my_decision": ""},
        {"my_decision": ""},
    ]
    assert review_batch2.select_rows(rows, limit=2, all_rows=False, pending_only=True, start=0) == [0, 2]
    assert review_batch2.select_rows(rows, limit=None, all_rows=True, pending_only=True, start=1) == [2, 3]


def test_export_only_kept(tmp_path: Path, monkeypatch, capsys) -> None:
    _setup(tmp_path, monkeypatch)
    review_batch2.export()
    out = capsys.readouterr().out
    assert "Part A: exported 1 kept rows" in out
    assert (tmp_path / "batch2_approved_answerable.csv").exists()
    assert (tmp_path / "batch2_approved.jsonl").exists()
