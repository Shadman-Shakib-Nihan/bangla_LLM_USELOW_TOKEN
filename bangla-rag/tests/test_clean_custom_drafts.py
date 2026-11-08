import csv
import hashlib
import json

import pytest

from experiments.clean_custom_drafts import (
    answer_type,
    compact,
    deterministic_question_type,
    exclude_reasons,
    process,
    recompute_alt_gold,
    read_csv,
    verify_output_accounting,
)


FIELDS = [
    "id", "question", "question_type", "gold_answer", "passage_snippet",
    "gold_passage_id", "alt_gold_ids", "answerable", "language_pair",
    "answer_is_substring", "max_shared_word_4gram", "check_flags",
    "my_decision", "my_notes",
]


def row(**kw):
    base = dict(zip(FIELDS, [
        "q1", "ঢাকার রাজধানী কী?", "what", "উত্তর", "উত্তরের অংশ", "p1",
        "old;old", "true", "bn-bn", "true", "0", "", "keep", "note",
    ]))
    base.update(kw)
    return base


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def test_source_unchanged_and_output_created_separately(tmp_path):
    source = tmp_path / "custom_drafts.csv"
    write_csv(source, [row()])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"passage_id": "p1", "text": "প্রাসঙ্গিক তথ্য"}, ensure_ascii=False)+"\n", encoding="utf-8")
    clean, excluded, log = [tmp_path / n for n in ("v2.csv", "excluded.csv", "changelog.md")]
    process(source, corpus, clean, excluded, log)
    assert source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert clean.exists() and clean != source


@pytest.mark.parametrize("updates,reason", [
    ({"answer_is_substring": "false"}, "answer_is_substring_false"),
    ({"gold_answer": "১"}, "normalized_answer_under_2_characters"),
    ({"max_shared_word_4gram": "5"}, "shared_4gram_at_least_5"),
    ({"check_flags": "shared_4gram=1;name_heavy"}, "name_heavy"),
    ({"question": "উত্তর কোথায়?"}, "answer_appears_in_question"),
])
def test_each_exclusion_condition_independently(updates, reason):
    assert reason in exclude_reasons(row(**updates))


def test_multiple_exclusion_reasons_accumulate():
    reasons = exclude_reasons(row(answer_is_substring="false", max_shared_word_4gram="7", check_flags="name_heavy", gold_answer="১"))
    assert len(reasons) == 4
    assert set(reasons) == {"answer_is_substring_false", "normalized_answer_under_2_characters", "shared_4gram_at_least_5", "name_heavy"}


def test_bengali_english_digit_normalization_and_compact_match():
    assert compact("  মূল্য: ১২-৩৪! ") == compact("মূল্য 12 34")


def test_year_question_requires_four_digit_year_or_full_date():
    assert answer_type("কত সালে প্রতিষ্ঠিত?", "১৯৮০", True) == "valid_year"
    assert answer_type("কবে জন্মগ্রহণ করেন?", "২০ ডিসেম্বর, ১৯৮০", True) == "valid_full_date"
    assert answer_type("কোন তারিখে মুক্তি পায়?", "১৯ ডিসেম্বর, ২০১৪", True) == "valid_full_date"
    assert answer_type("কোন সালে?", "গত শতকে", True) == "year_mismatch"


def test_quantity_question_requires_recognizable_number():
    assert answer_type("কতজন ছিলেন?", "১২ জন", True) == "valid_number"
    assert answer_type("কতটি বই?", "তিনটি", True) == "valid_number"
    assert answer_type("কতটি বই?", "বেশ কয়েকটি", True) == "number_mismatch"
    assert answer_type("কী বাধ্যবাধকতা পালন করা হয়?", "শর্ত মানতে হয়", True) == "not_applicable"


def test_alt_passage_requires_answer_and_rare_question_term():
    docs = [("p1", "উত্তর টোকেন ঢাকার তথ্য"), ("p2", "উত্তর সাধারণ তথ্য"), ("p3", "বিরল শব্দ মাত্র")]
    results = recompute_alt_gold([row(question="টোকেনের তথ্য কী?", gold_answer="উত্তর", gold_passage_id="p0")], docs)
    assert results["q1"] == ["p1"]


def test_duplicate_alt_ids_are_removed():
    docs = [("p2", "উত্তর অনন্যটোকেন"), ("p2", "উত্তর অনন্যটোকেন")]
    result = recompute_alt_gold([row(question="অনন্যটোকেন কী?", gold_answer="উত্তর", gold_passage_id="p1")], docs)
    assert result["q1"] == ["p2"]


def test_alt_gold_list_cap_and_overflow(tmp_path):
    src = tmp_path / "in.csv"; write_csv(src, [row(question="দুর্লভশব্দ কী?", gold_answer="উত্তর")])
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("".join(json.dumps({"passage_id": f"p{i:04}", "text": "উত্তর দুর্লভশব্দ" if i < 13 else f"অন্য তথ্য {i}"}, ensure_ascii=False)+"\n" for i in range(3000)), encoding="utf-8")
    clean = tmp_path / "out.csv"
    process(src, corpus, clean, tmp_path / "ex.csv", tmp_path / "log.md")
    with clean.open(encoding="utf-8", newline="") as f: out = next(csv.DictReader(f))
    assert len(out["alt_gold_ids"].split(";")) == 10
    assert out["alt_gold_over_cap"] == "true"
    assert out["alt_gold_total_count"] == "13"


@pytest.mark.parametrize("question,expected", [
    ("কত সালে জন্মগ্রহণ করেন?", "when"),
    ("কতজন সদস্য ছিলেন?", "how_many"),
    ("কীভাবে কাজটি করেন?", "why_how"),
    ("কোন শহরে জন্মগ্রহণ করেন?", "where"),
    ("কাজটি কে করেন?", "who"),
    ("কোন বইটি প্রকাশিত হয়?", "what"),
    ("কী ধরনের বাধ্যবাধকতা পালন হয়?", "what"),
    ("কোন শিক্ষা প্রতিষ্ঠানে ভর্তি হয়েছিলেন?", "where"),
    ("কোন অস্ট্রেলীয় ক্রিকেটার জন্মগ্রহণ করেন?", "who"),
    ("কোন দেশটিতে সামরিক অভিযান হয়েছিল?", "where"),
    ("কোন ওয়েবসাইটের লেখক ছিলেন?", "what"),
])
def test_question_type_rules(question, expected):
    assert deterministic_question_type(question) == expected


def test_my_decision_and_notes_unchanged_for_retained_record(tmp_path):
    source = tmp_path / "in.csv"; original = row()
    write_csv(source, [original])
    corpus = tmp_path / "c.jsonl"; corpus.write_text('{"passage_id":"p2","text":"উত্তর অনন্যতথ্য"}\n', encoding="utf-8")
    clean = tmp_path / "out.csv"
    process(source, corpus, clean, tmp_path / "ex.csv", tmp_path / "log.md")
    with clean.open(encoding="utf-8", newline="") as f: out = next(csv.DictReader(f))
    assert out["my_decision"] == original["my_decision"]
    assert out["my_notes"] == original["my_notes"]


def test_csv_round_trip_preserves_bengali_and_separate_boolean_fields(tmp_path):
    source = tmp_path / "in.csv"; original = row(question="বাংলা প্রশ্ন—কী?", gold_answer="ঢাকা", answerable="false", answer_is_substring="true")
    write_csv(source, [original])
    corpus = tmp_path / "c.jsonl"; corpus.write_text('{"passage_id":"p2","text":"ঢাকা বিরলতথ্য"}\n', encoding="utf-8")
    clean = tmp_path / "out.csv"
    unanswerable = tmp_path / "unanswerable.csv"
    process(source, corpus, clean, tmp_path / "ex.csv", tmp_path / "log.md", unanswerable)
    with unanswerable.open(encoding="utf-8", newline="") as f: out = next(csv.DictReader(f))
    assert out["question"] == original["question"]
    assert out["answerable"] == "false" and out["answer_is_substring"] == "true"
    assert out["gold_answer"] == "ঢাকা"


def accounting_fixture(tmp_path):
    records = [
        row(id="keep", question="ঢাকা কী?", gold_answer="উত্তর", answer_is_substring="true", my_decision="accept", my_notes="retained note"),
        row(id="no_answer", question="এই তথ্যের উত্তর কী?", gold_answer="", answerable="false", answer_is_substring="", my_decision="hold", my_notes="unanswerable note"),
        row(id="multi", question="১ প্রশ্ন?", gold_answer="১", answer_is_substring="false", max_shared_word_4gram="5", check_flags="name_heavy", my_decision="reject", my_notes="all reasons"),
        row(id="answer_in_q", question="উত্তর কী?", gold_answer="উত্তর", my_decision="review", my_notes="question contains answer"),
    ]
    source = tmp_path / "input.csv"
    write_csv(source, records)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"passage_id":"passage1","text":"উত্তর ঢাকার তথ্য"}\n', encoding="utf-8")
    paths = {
        "clean": tmp_path / "custom_drafts_v2.csv",
        "unanswerable": tmp_path / "custom_drafts_unanswerable.csv",
        "excluded": tmp_path / "custom_drafts_excluded.csv",
        "log": tmp_path / "changelog.md",
        "review": tmp_path / "question_type_review.csv",
    }
    return records, source, corpus, paths


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_excluded_rows_preserve_values_and_every_applicable_reason(tmp_path):
    records, source, corpus, p = accounting_fixture(tmp_path)
    process(source, corpus, p["clean"], p["excluded"], p["log"], p["unanswerable"], p["review"])
    excluded = {r["id"]: r for r in read_rows(p["excluded"])}
    original = {r["id"]: r for r in records}
    assert set(excluded) == {"multi", "answer_in_q"}
    for rid, out in excluded.items():
        assert all(out.get(col, "") == original[rid].get(col, "") for col in FIELDS)
        assert out["exclusion_reasons"].split(";") == exclude_reasons(original[rid])
    assert set(excluded["multi"]["exclusion_reasons"].split(";")) == {
        "answer_is_substring_false", "normalized_answer_under_2_characters",
        "shared_4gram_at_least_5", "name_heavy", "answer_appears_in_question",
    }


def test_unanswerable_records_are_preserved_and_not_short_answer_exclusions(tmp_path):
    records, source, corpus, p = accounting_fixture(tmp_path)
    process(source, corpus, p["clean"], p["excluded"], p["log"], p["unanswerable"], p["review"])
    unanswerable = read_rows(p["unanswerable"])
    assert len(unanswerable) == 1
    assert unanswerable[0]["id"] == "no_answer"
    assert unanswerable[0]["answerable"] == "false"
    assert unanswerable[0]["gold_answer"] == ""
    assert all(unanswerable[0].get(col, "") == records[1].get(col, "") for col in FIELDS)
    assert "no_answer" not in {r["id"] for r in read_rows(p["excluded"])}


def test_all_source_ids_accounted_for_without_cross_output_duplicates(tmp_path):
    records, source, corpus, p = accounting_fixture(tmp_path)
    process(source, corpus, p["clean"], p["excluded"], p["log"], p["unanswerable"], p["review"])
    groups = [read_rows(p[k]) for k in ("clean", "unanswerable", "excluded")]
    ids = [[r["id"] for r in g] for g in groups]
    flattened = [rid for group in ids for rid in group]
    assert len(flattened) == len(records)
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {r["id"] for r in records}
    fields, source_rows = read_csv(source)
    result = verify_output_accounting(source_rows, fields, groups[0], groups[1], groups[2])
    assert result == {"source": 4, "answerable": 1, "unanswerable": 1, "excluded": 2, "accounted": 4, "unique_ids": 4, "verified": True}


def test_integrity_verifier_detects_lost_and_duplicate_records(tmp_path):
    records, source, corpus, p = accounting_fixture(tmp_path)
    process(source, corpus, p["clean"], p["excluded"], p["log"], p["unanswerable"], p["review"])
    fields, source_rows = read_csv(source)
    groups = [read_rows(p[k]) for k in ("clean", "unanswerable", "excluded")]
    with pytest.raises(ValueError, match="record accounting mismatch"):
        verify_output_accounting(source_rows, fields, groups[0], groups[1], groups[2][:-1])
    with pytest.raises(ValueError, match="duplicate IDs within clean"):
        verify_output_accounting(source_rows, fields, groups[0] + groups[0], groups[1], groups[2])


def test_my_decision_and_notes_preserved_across_all_outputs(tmp_path):
    records, source, corpus, p = accounting_fixture(tmp_path)
    process(source, corpus, p["clean"], p["excluded"], p["log"], p["unanswerable"], p["review"])
    original = {r["id"]: r for r in records}
    for key in ("clean", "unanswerable", "excluded"):
        for out in read_rows(p[key]):
            assert out["my_decision"] == original[out["id"]]["my_decision"]
            assert out["my_notes"] == original[out["id"]]["my_notes"]


def test_question_type_review_covers_every_record_and_marks_fallback_ambiguous(tmp_path):
    records, source, corpus, p = accounting_fixture(tmp_path)
    process(source, corpus, p["clean"], p["excluded"], p["log"], p["unanswerable"], p["review"])
    review = read_rows(p["review"])
    assert len(review) == len(records)
    assert {r["id"] for r in review} == {r["id"] for r in records}
    assert set(review[0]) == {"id", "question", "assigned_question_type", "classification_reason", "review_status"}
    assert all(r["classification_reason"] for r in review)
    assert any(r["review_status"] == "manual_review" for r in review)
