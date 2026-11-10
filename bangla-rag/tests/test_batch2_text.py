from __future__ import annotations

from src.batch2_text import (
    answer_satisfies_type,
    compact_contains,
    compute_alt_gold_ids,
    contains_text,
    infer_question_type,
    is_name_heavy,
    is_time_sensitive_undated,
    max_shared_word_ngram,
    normalize_digits,
    question_rare_term_ok,
)


def test_contains_text_normalizes_bengali_digits() -> None:
    assert contains_text("তিনি ১৯৭১ সালে জন্ম নেন", "1971")


def test_normalize_digits_both_directions_for_comparison() -> None:
    assert normalize_digits("১২৩") == "123"


def test_answer_type_year_and_number() -> None:
    assert answer_satisfies_type("when", "কত সালে?", "১৯৭১")
    assert answer_satisfies_type("how_many", "কতজন?", "১২ জন")
    assert not answer_satisfies_type("how_many", "কতজন?", "অনেকে")


def test_max_shared_word_ngram_detects_copy() -> None:
    q = "বাংলাদেশের রাজধানী কোন শহর"
    p = "এই শহর বাংলাদেশের রাজধানী কোন শহর হিসেবে পরিচিত"
    assert max_shared_word_ngram(q, p, 4) >= 4


def test_time_sensitive_undated() -> None:
    assert is_time_sensitive_undated("বর্তমানে প্রধানমন্ত্রী কে?")
    assert not is_time_sensitive_undated("২০২৪ সালে প্রধানমন্ত্রী কে ছিলেন?")


def test_name_heavy_long_transliteration() -> None:
    assert is_name_heavy("ব্লো-ব্জাং-য়ে-শেস-ব্স্তান-র্গ্যা-ম্ত্শো কে ছিলেন?")


def test_question_rare_term_and_alt_gold() -> None:
    rare = {"ঢাকেশ্বরী"}
    passage = "ঢাকেশ্বরী মন্দির ঢাকায় অবস্থিত।"
    assert question_rare_term_ok("ঢাকেশ্বরী মন্দির কোথায়?", passage, rare)
    corpus = [
        ("g1", passage),
        ("g2", "ঢাকেশ্বরী মন্দির একটি প্রাচীন স্থাপনা।"),
        ("g3", "মন্দিরটি ঢাকায় অবস্থিত।"),
        ("g4", "ঢাকেশ্বরী মন্দিরটি ঢাকায় অবস্থিত এবং প্রাচীন।"),
    ]
    alt, total = compute_alt_gold_ids("ঢাকায়", "ঢাকেশ্বরী মন্দির কোথায়?", "g1", corpus, rare)
    assert alt == ["g4"]
    assert total == 1


def test_infer_question_type_rules() -> None:
    assert infer_question_type("কত সালে স্বাধীন হয়?") == "when"
    assert infer_question_type("কে ছিলেন?") == "who"
    assert infer_question_type("কোথায় অবস্থিত?") == "where"
    assert infer_question_type("কতজন সদস্য?") == "how_many"


def test_compact_contains() -> None:
    assert compact_contains("রবীন্দ্রনাথ ঠাকুর", "রবীন্দ্রনাথ")
