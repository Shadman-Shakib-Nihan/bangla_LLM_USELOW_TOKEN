from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizers import compute_fertility, load_flores_pairs


class WhitespaceCounter:
    """Deterministic counter used to test the statistics, not a real tokenizer."""

    name = "whitespace"

    def count(self, text: str) -> int:
        return len(text.split())

    def count_many(self, texts):
        return [self.count(t) for t in texts]


def test_compute_fertility_stats():
    pairs = [("bn1 bn2", "en1"), ("bn1", "en1 en2 en3")]
    result = compute_fertility(pairs, WhitespaceCounter())

    assert result["n_sentences"] == 2
    assert result["bn_tokens"] == [2, 1]
    assert result["en_tokens"] == [1, 3]
    assert result["mean_bn_tokens"] == pytest.approx(1.5)
    assert result["mean_en_tokens"] == pytest.approx(2.0)
    # per-sentence ratios: 2/1 and 1/3 -> mean 7/6
    assert result["mean_ratio_per_sentence"] == pytest.approx(7 / 6)
    # corpus totals: 3 bn / 4 en
    assert result["corpus_ratio"] == pytest.approx(0.75)


def test_compute_fertility_skips_zero_denominator():
    pairs = [("bn1", ""), ("bn2", "en1")]
    result = compute_fertility(pairs, WhitespaceCounter())
    # The empty English sentence is excluded from the per-sentence mean.
    assert result["mean_ratio_per_sentence"] == pytest.approx(1.0)


def test_load_flores_pairs_alignment_and_limit(tmp_path: Path):
    root = tmp_path / "flores"
    dev = root / "dev"
    dev.mkdir(parents=True)
    (dev / "ben_Beng.dev").write_text("bn-১\nbn-২\nbn-৩\n", encoding="utf-8")
    (dev / "eng_Latn.dev").write_text("en one\nen two\nen three\n", encoding="utf-8")

    pairs = load_flores_pairs(split="dev", root=root)
    assert pairs == [("bn-১", "en one"), ("bn-২", "en two"), ("bn-৩", "en three")]

    limited = load_flores_pairs(split="dev", root=root, limit=2)
    assert limited == [("bn-১", "en one"), ("bn-২", "en two")]


def test_load_flores_pairs_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_flores_pairs(split="dev", root=tmp_path / "missing")
