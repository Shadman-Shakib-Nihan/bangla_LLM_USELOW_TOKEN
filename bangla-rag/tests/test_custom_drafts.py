import csv
import json

import pytest

from experiments.build_custom_drafts import (
    ANSWER_SCHEMA,
    REVIEW_FIELDS,
    DraftResult,
    SampledPassage,
    SQLiteGenerationCache,
    article_key,
    build_answerable_row,
    build_unanswerable_row,
    classify_question_type,
    is_name_heavy,
    is_time_sensitive_undated,
    make_type_plan,
    max_shared_word_4gram,
    nfc_contains,
    parse_json_response,
    passage_snippet,
    select_sample,
    usage_cost_usd,
    write_csv,
)


def _record(pid: str, text: str, source: str = "miracl", title: str = "",
            is_short: bool = False, parent_id: str | None = None) -> dict:
    return {
        "passage_id": pid,
        "parent_id": parent_id or pid,
        "is_short": is_short,
        "title": title,
        "text": text,
        "source": source,
    }


def _draft(question: str, answer: str) -> DraftResult:
    return DraftResult(
        parsed={"question": question, "answer": answer},
        response_text=json.dumps({"question": question, "answer": answer}),
        prompt_tokens=100,
        output_tokens=20,
        total_tokens=120,
        finish_reason="STOP",
        cached=False,
    )


def test_article_key_groups_miracl_before_hash_and_tydi_by_title():
    assert article_key(_record("m1#0", "x")) == "miracl:m1"
    assert article_key(_record("m1#3", "x")) == "miracl:m1"
    assert article_key(_record("t1", "x", source="tydi", title="ঢাকা")) == "tydi:ঢাকা"


def test_select_sample_filters_short_lengths_and_one_per_article():
    records = [
        _record("a#0", "a", is_short=True),
        _record("a#1", "b"),
        _record("b#0", "c"),
        _record("c#0", "d"),
    ]
    lengths = [100, 100, 100, 500]
    sample = select_sample(records, lengths, seed=1, n_total=2)
    assert len(sample) == 2
    assert len({s.article_key for s in sample}) == 2
    assert all(40 <= s.e5_tokens <= 300 for s in sample)
    assert all(s.passage_id != "a#0" for s in sample)


def test_select_sample_is_deterministic():
    records = [_record(f"{i}#0", "x") for i in range(10)]
    lengths = [100] * 10
    a = [s.passage_id for s in select_sample(records, lengths, seed=7, n_total=5)]
    b = [s.passage_id for s in select_sample(records, lengths, seed=7, n_total=5)]
    assert a == b


def test_nfc_contains_handles_decomposed_bangla_text():
    composed = "\u09dc"
    decomposed = "\u09a1\u09bc"
    assert composed != decomposed
    assert nfc_contains(composed, f"রাজধানী {decomposed} শহর")
    assert not nfc_contains("", "কোনো পাঠ")


def test_max_shared_word_4gram_counts_overlap_not_just_presence():
    passage = "আমি বাংলা ভাষা ভালোবাসি এবং বাংলা সাহিত্য পড়ি"
    assert max_shared_word_4gram("বাংলা ভাষা ভালোবাসি এবং কেন", passage) == 1
    assert max_shared_word_4gram("কোন শহর", passage) == 0


def test_parse_json_response_accepts_fenced_json():
    parsed = parse_json_response('```json\n{"question": "প্রশ্ন", "answer": "উত্তর"}\n```')
    assert parsed == {"question": "প্রশ্ন", "answer": "উত্তর"}


def test_build_answerable_row_flags_clean_and_bad_answers():
    rec = SampledPassage(
        passage_id="p1",
        parent_id="p1",
        article_key="miracl:p1",
        title="ঢাকা",
        text="বাংলাদেশের রাজধানী ঢাকা শহরটি অনেক পুরনো।",
        source="miracl",
        e5_tokens=100,
    )
    corpus = [("p1", rec.text), ("p2", "অন্য পাঠে ঢাকা আছে।")]
    clean = build_answerable_row(
        1, rec, _draft("দেশের রাজধানী কোন শহর?", "ঢাকা"), corpus
    )
    assert clean["answer_is_substring"] == "true"
    assert clean["question_type"] == "what"
    assert "ঢাকা" in clean["passage_snippet"]
    assert clean["max_shared_word_4gram"] == "0"
    assert clean["check_flags"] == ""
    assert clean["alt_gold_ids"] == "p2"
    assert clean["answerable"] == "true"
    assert clean["my_decision"] == clean["my_notes"] == ""

    bad = build_answerable_row(
        2, rec, _draft("বাংলাদেশের রাজধানী ঢাকা শহরটি কোন জেলা?", "কলকাতা"), corpus
    )
    assert bad["answer_is_substring"] == "false"
    assert "answer_not_substring" in bad["check_flags"]
    assert int(bad["max_shared_word_4gram"]) > 0


def test_build_unanswerable_row_has_no_gold_passage():
    rec = SampledPassage(
        passage_id="p1",
        parent_id="p1",
        article_key="miracl:p1",
        title="ঢাকা",
        text="বাংলাদেশের রাজধানী ঢাকা শহরটি অনেক পুরনো।",
        source="miracl",
        e5_tokens=100,
    )
    result = DraftResult(
        parsed={"question": "ঢাকার মেয়র কে?", "premise_type": "missing_detail"},
        response_text="{}",
        prompt_tokens=100,
        output_tokens=20,
        total_tokens=120,
        finish_reason="STOP",
        cached=False,
    )
    row = build_unanswerable_row(151, rec, result)
    assert row["answerable"] == "false"
    assert row["gold_passage_id"] == "NONE"
    assert row["gold_answer"] == ""
    assert row["answer_is_substring"] == ""
    assert "manual_unanswerable_review" in row["check_flags"]


def test_sqlite_cache_round_trip(tmp_path):
    cache = SQLiteGenerationCache(tmp_path / "cache.sqlite")
    key = SQLiteGenerationCache.make_key(
        "gemini-3.1-flash-lite", "sys", "user", 0.0, 256, ANSWER_SCHEMA
    )
    assert cache.get(key) is None
    cache.put(
        key, "gemini-3.1-flash-lite", "sys", "user", '{"a": 1}',
        {"a": 1}, 10, 2, 12, "STOP",
    )
    hit = cache.get(key)
    assert hit is not None
    assert hit["response_json"] == {"a": 1}
    assert hit["prompt_tokens"] == 10
    assert cache.count() == 1


def test_usage_cost_matches_pricing():
    assert usage_cost_usd(1_000_000, 1_000_000, "gemini-3.1-flash-lite") == pytest.approx(1.75)


def test_write_csv_writes_exact_review_header(tmp_path):
    path = tmp_path / "review.csv"
    write_csv(path, REVIEW_FIELDS, [])
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        assert next(reader) == REVIEW_FIELDS
        assert list(reader) == []


def test_classify_question_type_prefers_date_heuristic():
    assert classify_question_type("what", "কত সালে জন্মগ্রহণ করেন?") == "when"
    assert classify_question_type("who", "কে কাজটি করেছেন?") == "who"
    assert classify_question_type("how", "কীভাবে কাজটি হলো?") == "why_how"
    assert classify_question_type("", "কোথায় জন্মগ্রহণ করেন?") == "where"


def test_name_heavy_detects_long_transliterated_name():
    long_name = "ব্লো-ব্জাং-য়ে-শেস-ব্স্তান-'দ্জিন-র্গ্যা-ম্ত্শোর বাবার নাম কী?"
    assert is_name_heavy(long_name)
    assert not is_name_heavy("আন্তনিন দ্ভরাক কোন সালে জন্মগ্রহণ করেন?")


def test_time_sensitive_requires_explicit_year():
    assert is_time_sensitive_undated("বর্তমানে এই দলের অধিনায়ক কে?")
    assert not is_time_sensitive_undated("২০২০ সালে এই দলের অধিনায়ক কে ছিলেন?")


def test_passage_snippet_centres_answer():
    text = "শুরু " + ("পূরণ " * 80) + "রাজধানী ঢাকা শহর" + (" শেষ" * 80)
    snippet = passage_snippet(text, "ঢাকা", width=100)
    assert "ঢাকা" in snippet
    assert snippet.startswith("...")


def test_type_plan_has_date_cap_and_all_types():
    plan = make_type_plan(
        {"who": 2, "when": 1, "where": 2, "what": 2, "how_many": 1, "why_how": 2},
        seed=3,
    )
    assert sorted(plan) == sorted(
        ["who", "who", "when", "where", "where", "what", "what", "how_many", "why_how", "why_how"]
    )
