import unicodedata

import numpy as np

from src.retrieval import (
    aggregate_parent_scores,
    build_parent_index,
    rank_parents,
    tokenize,
)


def test_tokenize_keeps_bangla_conjunct_and_vowel_signs():
    tokens = tokenize("বাংলা ভাষা।")
    assert "বাংলা" in tokens
    assert "ভাষা" in tokens
    assert "।" not in tokens


def test_tokenize_nfc_normalization():
    composed = unicodedata.normalize("NFC", "e\u0301")
    decomposed = unicodedata.normalize("NFD", "e\u0301")
    assert composed != decomposed
    assert tokenize(composed) == tokenize(decomposed) == ["é"]


def test_tokenize_keeps_digits():
    assert tokenize("২০২৬ সালে") == ["২০২৬", "সালে"]


def test_parent_aggregation_and_ranking():
    pidx = build_parent_index(["p1", "p1", "p2"])
    scores = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    parent_scores = aggregate_parent_scores(scores, pidx)
    assert pidx.parent_ids == ["p1", "p2"]
    assert parent_scores.tolist() == [3.0, 2.0]
    assert rank_parents(parent_scores, pidx, top_k=2) == ["p1", "p2"]


def test_rank_parents_only_positive():
    pidx = build_parent_index(["p1", "p2", "p3"])
    scores = np.array([0.0, 2.0, 1.0], dtype=np.float32)
    assert rank_parents(scores, pidx) == ["p2", "p3"]
    assert rank_parents(scores, pidx, only_positive=False) == ["p2", "p3", "p1"]
