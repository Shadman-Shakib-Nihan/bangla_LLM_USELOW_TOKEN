import pytest

from src.phase4b import (
    bootstrap_mean_ci,
    metric_values,
    paired_bootstrap_diff,
    summarize_m4,
    summarize_values,
)


def test_bootstrap_mean_ci_is_deterministic():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    first = bootstrap_mean_ci(values, n_boot=500, seed=7)
    second = bootstrap_mean_ci(values, n_boot=500, seed=7)
    assert first == second
    assert first["n"] == 5
    assert first["mean"] == pytest.approx(0.6)
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_paired_bootstrap_requires_identical_ids():
    with pytest.raises(ValueError, match="identical question IDs"):
        paired_bootstrap_diff({"a": 1.0}, {"b": 1.0}, n_boot=100)


def test_paired_bootstrap_zero_difference_is_not_significant():
    values = {f"q{i}": float(i % 2) for i in range(40)}
    result = paired_bootstrap_diff(values, values, n_boot=1000, seed=42)
    assert result["mean_diff"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["includes_zero"] is True
    assert result["significant"] is False


def test_paired_bootstrap_consistent_gain_is_significant():
    a = {f"q{i}": 1.0 for i in range(40)}
    b = {f"q{i}": 0.0 for i in range(40)}
    result = paired_bootstrap_diff(a, b, n_boot=1000, seed=42)
    assert result["mean_diff"] == pytest.approx(100.0)
    assert result["ci_low"] > 0.0
    assert result["includes_zero"] is False
    assert result["significant"] is True


def test_paired_bootstrap_mean_difference_uses_percentage_points():
    a = {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 0.0}
    b = {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 0.0}
    result = paired_bootstrap_diff(a, b, n_boot=200, seed=1)
    assert result["mean_diff"] == pytest.approx(50.0)
    assert result["n_questions"] == 4


def test_metric_values_filters_budget_method_and_gold():
    records = [
        {"dataset": "d", "retriever": "r", "method": "M1", "budget_tokens": 128,
         "question_id": "a", "metric": 1.0, "gold_in_top10": True},
        {"dataset": "d", "retriever": "r", "method": "M1", "budget_tokens": 256,
         "question_id": "b", "metric": 0.0, "gold_in_top10": True},
        {"dataset": "d", "retriever": "r", "method": "M2", "budget_tokens": 128,
         "question_id": "c", "metric": 1.0, "gold_in_top10": False},
        {"dataset": "d", "retriever": "r", "method": "M0", "budget_tokens": None,
         "question_id": "d", "metric": 1.0, "gold_in_top10": True},
    ]
    assert metric_values(
        records, dataset="d", retriever="r", method="M1", metric="metric",
        budget=128,
    ) == {"a": 1.0}
    assert metric_values(
        records, dataset="d", retriever="r", method="M0", metric="metric",
    ) == {"d": 1.0}
    assert metric_values(
        records, dataset="d", retriever="r", method="M2", metric="metric",
        budget=128, require_gold_in_top10=True,
    ) == {}


def test_summarize_values_includes_median_and_extremes():
    result = summarize_values([1, 2, 3, 10])
    assert result == {
        "n": 4,
        "mean": 4.0,
        "median": 2.5,
        "min": 1.0,
        "max": 10.0,
    }


def test_summarize_m4_counts_zero_and_multiple_matches():
    rows = [
        {"o200k_tokens": 10, "e5_tokens": 9, "n_selected": 0},
        {"o200k_tokens": 20, "e5_tokens": 18, "n_selected": 1},
        {"o200k_tokens": 30, "e5_tokens": 25, "n_selected": 3},
        {"o200k_tokens": 40, "e5_tokens": 35, "n_selected": 2},
    ]
    result = summarize_m4(rows, n_boot=100, seed=42)
    assert result["n_questions"] == 4
    assert result["o200k"]["median"] == 25.0
    assert result["e5"]["median"] == 21.5
    assert result["mean_selected_sentences"] == 1.5
    assert result["zero_match_count"] == 1
    assert result["zero_match_share"] == 0.25
    assert result["multi_match_count"] == 2
    assert result["multi_match_share"] == 0.5
