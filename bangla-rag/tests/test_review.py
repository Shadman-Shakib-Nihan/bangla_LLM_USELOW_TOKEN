import csv
from pathlib import Path

import pytest

import review


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_schema_unicode_and_missing_values(tmp_path):
    path = tmp_path / "bn.csv"
    write_csv(path, ["id", "question", "my_decision"], [{"id": "1", "question": "বাংলা প্রশ্ন", "my_decision": ""}])
    columns, rows = review.validate_columns(path, {"id", "question"})
    assert columns == ["id", "question", "my_decision"]
    assert rows[0]["question"] == "বাংলা প্রশ্ন"
    with pytest.raises(review.ReviewError, match="missing required"):
        review.validate_columns(path, {"gold_answer"})


@pytest.mark.parametrize("decision,note", [("keep", ""), ("fix", "Clarify the wording"), ("reject", "Incorrect answer")])
def test_decisions(decision, note):
    row = {"my_decision": "", "my_notes": ""}
    review.apply_decision(row, decision, note)
    assert row == {"my_decision": decision, "my_notes": note}


@pytest.mark.parametrize("decision", ["fix", "reject"])
def test_notes_required(decision):
    with pytest.raises(review.ReviewError, match="note is required"):
        review.apply_decision({"my_decision": "", "my_notes": ""}, decision)


def test_save_autosaves_and_backups(tmp_path):
    path = tmp_path / "review.csv"
    columns = ["id", "question", "my_decision", "my_notes"]
    rows = [{"id": "cd_1", "question": "বাংলা", "my_decision": "keep", "my_notes": ""}]
    write_csv(path, columns, [{**rows[0], "my_decision": ""}])
    review.save_csv(path, columns, rows)
    assert list(tmp_path.glob("review.csv.*.bak"))
    assert review.load_csv(path)[1] == rows


def test_existing_decisions_protection_from_review(tmp_path):
    path = tmp_path / "review.csv"
    columns = ["id", "question", "my_decision", "my_notes"]
    rows = [{"id": "cd_1", "question": "Q", "my_decision": "keep", "my_notes": ""}]
    answers = iter(["n"])
    review.review_loop(path, columns, rows, rows, "answerable", input_fn=lambda _: next(answers))
    assert not path.exists()
    assert rows[0]["my_decision"] == "keep"


def test_retrieval_match_requires_five_ranked_passages():
    question = {"id": "cd_1", "question": "বাংলা?"}
    passages = [{"id": "cd_1", "question": "বাংলা?", "rank": str(i)} for i in range(1, 6)]
    assert len(review.retrieval_map([question], passages)["cd_1"]) == 5
    with pytest.raises(review.ReviewError, match="expected 5"):
        review.retrieval_map([question], passages[:4])


def test_export_only_keeps_and_preserves_columns(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    target = tmp_path / "custom_questions.csv"
    columns = ["id", "question", "gold_answer", "passage_snippet", "my_decision", "my_notes", "language_pair"]
    rows = [
        dict(id="a", question="Q1", gold_answer="A1", passage_snippet="P1", my_decision="keep", my_notes="", language_pair="bn-bn"),
        dict(id="b", question="Q2", gold_answer="A2", passage_snippet="P2", my_decision="fix", my_notes="revise", language_pair="bn-bn"),
        dict(id="c", question="Q3", gold_answer="A3", passage_snippet="P3", my_decision="reject", my_notes="bad", language_pair="bn-bn"),
    ]
    write_csv(source, columns, rows)
    monkeypatch.setattr(review, "ANSWERABLE", source)
    monkeypatch.setattr(review, "EXPORT", target)
    review.export()
    out_columns, out_rows = review.load_csv(target)
    assert out_columns == [c for c in columns if c != "my_notes"]
    assert [row["id"] for row in out_rows] == ["a"]
    assert "P1" == out_rows[0]["passage_snippet"]


def test_protected_files_cannot_be_written(tmp_path):
    path = tmp_path / "custom_drafts.csv"
    with pytest.raises(review.ReviewError, match="protected"):
        review.save_csv(path, ["id"], [{"id": "1"}])
    assert not path.exists()
