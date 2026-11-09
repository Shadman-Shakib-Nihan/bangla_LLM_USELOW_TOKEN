import random

import pytest

from src.compression import (
    build_candidates,
    contains_answer,
    join_selected,
    m0_top_k,
    m1_hard_budget,
    m2_e5,
    m3_bm25,
    m4_oracle,
    m5_random,
    normalize_for_match,
    split_sentences,
)


def word_count(text: str) -> int:
    return len(text.split())


def word_truncate(text: str, budget: int) -> str:
    return " ".join(text.split()[:budget])


def test_split_on_bangla_danda():
    assert split_sentences("প্রথম বাক্য। দ্বিতীয় বাক্য।") == [
        "প্রথম বাক্য।",
        "দ্বিতীয় বাক্য।",
    ]


def test_split_on_question_mark():
    assert split_sentences("তুমি যাবে? আমি যাব।") == [
        "তুমি যাবে?",
        "আমি যাব।",
    ]


def test_split_on_newlines():
    assert split_sentences("প্রথম\nদ্বিতীয়\r\nতৃতীয়") == [
        "প্রথম",
        "দ্বিতীয়",
        "তৃতীয়",
    ]


def test_decimal_point_is_not_a_boundary():
    sentences = split_sentences("দাম ৩.১৪ টাকা। পরের বাক্য।")
    assert sentences == ["দাম ৩.১৪ টাকা।", "পরের বাক্য।"]
    assert "৩.১৪" in sentences[0]


def test_quoted_sentence_splits_after_closing_quote():
    text = 'তিনি বললেন, "আমি যাব।" তারপর চলে গেলেন।'
    assert split_sentences(text) == [
        'তিনি বললেন, "আমি যাব।"',
        "তারপর চলে গেলেন।",
    ]


def test_quoted_question_mark():
    text = 'প্রশ্ন "তুমি কি যাবে?" উত্তর পরে।'
    assert split_sentences(text) == [
        'প্রশ্ন "তুমি কি যাবে?"',
        "উত্তর পরে।",
    ]


def test_empty_and_no_punctuation():
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []
    assert split_sentences("কোনো যতিচিহ্ন নেই") == ["কোনো যতিচিহ্ন নেই"]


def test_normalize_nfc_and_digits():
    assert normalize_for_match("২০২৬ সালের 12") == "2026 সালের 12"
    composed = normalize_for_match("e\u0301")
    decomposed = normalize_for_match("e\u0301".encode().decode())
    assert composed == decomposed


def test_answer_matcher_exact_and_multiple_answers():
    context = "তিনি ১৯৭১ সালে ঢাকায় জন্মগ্রহণ করেন।"
    assert contains_answer(context, ["ঢাকা"])[0]
    assert contains_answer(context, ["১৯৭১"])[0]
    assert contains_answer(context, ["কলকাতা"])[0] is False
    assert contains_answer(context, ["কলকাতা", "ঢাকা"])[0]
    assert contains_answer(context, [])[0] is False


def test_answer_matcher_no_fuzzy_matching():
    assert contains_answer("আব্দুল করিম", ["আব্দুল করিম"])[0]
    assert contains_answer("আব্দুল করিম", ["আব্দুল করিম সাহেব"])[0] is False
    assert contains_answer("আব্দুল করিম", ["আব্দুল"])[0]


def test_m0_preserves_retrieval_order():
    passages = ["p1", "p2", "p3"]
    assert m0_top_k(passages, 2) == "p1\n\np2"
    assert m0_top_k(passages, 10) == "p1\n\np2\n\np3"


def test_m1_truncates_at_budget():
    passages = ["one two three", "four five"]
    context = m1_hard_budget(passages, 3, word_count, word_truncate)
    assert word_count(context) <= 3
    assert context == "one two three"


def test_m2_restores_original_order_and_respects_budget():
    passages = ["p1s1। p1s2।", "p2s1।"]
    candidates = build_candidates(passages)
    scores = [0.9, 0.1, 0.8]
    result = m2_e5(candidates, scores, 2, word_count, word_truncate)
    assert result.context == "p1s1।\n\np2s1।"
    assert word_count(result.context) <= 10
    assert [c.order for c in result.selected] == [0, 2]


def test_m3_uses_supplied_scores_and_budget():
    passages = ["a b। c d।", "e f।"]
    candidates = build_candidates(passages)
    result = m3_bm25(candidates, [0.1, 0.9, 0.5], 3, word_count, word_truncate)
    assert word_count(result.context) <= 3
    assert "c d।" in result.context


def test_m2_single_oversized_sentence_is_truncated():
    passages = ["one two three four five।"]
    candidates = build_candidates(passages)
    result = m2_e5(candidates, [1.0], 2, word_count, word_truncate)
    assert result.truncated is True
    assert result.context == "one two"


def test_m4_oracle_keeps_answer_sentences_only():
    passages = ["প্রথম বাক্য। উত্তর ঢাকায়।", "অন্য বাক্য।"]
    candidates = build_candidates(passages)
    result = m4_oracle(candidates, ["ঢাকা"])
    assert result.context == "উত্তর ঢাকায়।"
    assert len(result.selected) == 1


def test_m5_random_is_seeded_and_preserves_order():
    passages = ["s1। s2। s3।", "s4। s5।"]
    candidates = build_candidates(passages)
    r1 = m5_random(candidates, 6, word_count, word_truncate,
                   random.Random(7))
    r2 = m5_random(candidates, 6, word_count, word_truncate,
                   random.Random(7))
    assert r1.context == r2.context
    assert word_count(r1.context) <= 6
    assert r1.context == join_selected(r1.selected)


def test_join_selected_preserves_passage_boundaries():
    candidates = build_candidates(["a। b।", "c।"])
    selected = [candidates[2], candidates[0]]
    assert join_selected(selected) == "a।\n\nc।"


def test_empty_candidates_are_safe():
    assert m2_e5([], [], 10, word_count, word_truncate).context == ""
    assert m5_random([], 10, word_count, word_truncate,
                     random.Random(1)).context == ""
    assert m4_oracle([], ["x"]).context == ""
