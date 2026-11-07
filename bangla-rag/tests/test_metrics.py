import math

import pytest

from src.metrics import (
    evaluate_rankings,
    mrr,
    ndcg_at_k,
    recall_at_k,
    recall_fraction_at_k,
)


def test_recall_at_k_hit_rate():
    ranking = ["a", "b", "c", "d", "e"]
    assert recall_at_k(ranking, {"c"}, 1) == 0.0
    assert recall_at_k(ranking, {"c"}, 3) == 1.0
    assert recall_at_k(ranking, {"c"}, 5) == 1.0


def test_recall_fraction_at_k():
    ranking = ["a", "b", "c", "d", "e"]
    assert recall_fraction_at_k(ranking, {"b", "d"}, 1) == 0.0
    assert recall_fraction_at_k(ranking, {"b", "d"}, 2) == 0.5
    assert recall_fraction_at_k(ranking, {"b", "d"}, 4) == 1.0
    assert recall_fraction_at_k(ranking, {"b", "d"}, 3) == 0.5


def test_mrr():
    ranking = ["a", "b", "c", "d"]
    assert mrr(ranking, {"a"}) == 1.0
    assert mrr(ranking, {"c"}) == pytest.approx(1 / 3)
    assert mrr(ranking, {"z"}) == 0.0


def test_ndcg_binary_single_gold():
    ranking = ["a", "b", "c", "d", "e"]
    # Relevant at rank 3: DCG = 1/log2(4) = 0.5; IDCG = 1.
    assert ndcg_at_k(ranking, {"c"}, 3) == pytest.approx(0.5)
    assert ndcg_at_k(ranking, {"c"}, 1) == 0.0


def test_ndcg_binary_multiple_gold():
    ranking = ["a", "b", "c", "d"]
    gold = {"b", "d"}
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (
        1 + 1 / math.log2(3)
    )
    assert ndcg_at_k(ranking, gold, 4) == pytest.approx(expected)


def test_evaluate_rankings():
    rankings = {"q1": ["a", "b", "c"], "q2": ["x", "y", "z"]}
    gold = {"q1": {"b"}, "q2": {"z"}}
    summary = evaluate_rankings(rankings, gold)
    assert summary["n"] == 2
    assert summary["recall@1"] == 0.0
    assert summary["recall@3"] == 1.0
    assert summary["mrr"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert summary["ndcg@10"] == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(4)) / 2
    )


def test_evaluate_rankings_dedupes():
    rankings = {"q1": ["a", "a", "b"]}
    gold = {"q1": {"b"}}
    summary = evaluate_rankings(rankings, gold)
    assert summary["mrr"] == pytest.approx(0.5)


def test_evaluate_rankings_requires_overlap():
    with pytest.raises(ValueError):
        evaluate_rankings({"q1": ["a"]}, {"q2": {"a"}})
