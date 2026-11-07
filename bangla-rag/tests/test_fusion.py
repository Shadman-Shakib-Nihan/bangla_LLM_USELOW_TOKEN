import pytest

from src.fusion import rrf_fuse, weighted_rrf


def test_rrf_symmetric_tie_breaks_by_doc_id():
    fused = rrf_fuse([["a", "b"], ["b", "a"]])
    assert fused == ["a", "b"]


def test_rrf_weighting():
    assert rrf_fuse([["a", "b"], ["b", "a"]], weights=[1.0, 0.0]) == ["a", "b"]
    assert rrf_fuse([["a", "b"], ["b", "a"]], weights=[0.0, 1.0]) == ["b", "a"]


def test_rrf_middle_doc_loses():
    fused = rrf_fuse([["a", "b", "c"], ["c", "b", "a"]])
    assert fused == ["a", "c", "b"]


def test_rrf_deduplicates_within_a_ranking():
    fused = rrf_fuse([["a", "a", "b"]])
    assert fused == ["a", "b"]


def test_rrf_k_changes_scores_not_order_for_two_lists():
    fused_60 = rrf_fuse([["a", "b"], ["b", "a"]], k=60)
    fused_10 = rrf_fuse([["a", "b"], ["b", "a"]], k=10)
    assert fused_60 == fused_10


def test_rrf_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_validation():
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=-1)


def test_weighted_rrf():
    assert weighted_rrf(["a", "b"], ["b", "a"], 1.0) == ["a", "b"]
    assert weighted_rrf(["a", "b"], ["b", "a"], 0.0) == ["b", "a"]
    with pytest.raises(ValueError):
        weighted_rrf(["a"], ["a"], 1.5)
