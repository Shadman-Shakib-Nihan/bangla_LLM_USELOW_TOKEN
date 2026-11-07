from src.chunking import chunk_text, split_sentences


def word_count(text: str) -> int:
    return len(text.split())


def test_split_sentences_danda_and_newlines():
    text = "প্রথম বাক্য। দ্বিতীয় বাক্য\nতৃতীয় বাক্য"
    assert split_sentences(text) == ["প্রথম বাক্য।", "দ্বিতীয় বাক্য", "তৃতীয় বাক্য"]


def test_split_sentences_no_space_after_danda():
    assert split_sentences("ক।খ") == ["ক।", "খ"]


def test_split_sentences_drops_empty_parts():
    assert split_sentences("\n\nপ্রথম।\n\nদ্বিতীয়।\n") == ["প্রথম।", "দ্বিতীয়।"]


def test_chunk_text_uses_one_sentence_overlap():
    text = "s1। s2। s3। s4। s5। s6।"
    chunks, stats = chunk_text(text, word_count, max_tokens=3)
    assert chunks == ["s1। s2। s3।", "s3। s4। s5।", "s5। s6।"]
    assert stats.chunks == 3
    assert stats.oversized_sentences == 0
    assert all(word_count(c) <= 3 for c in chunks)


def test_chunk_text_splits_oversized_sentence_by_words():
    chunks, stats = chunk_text("a b c d e f।", word_count, max_tokens=2)
    assert chunks == ["a b", "c d", "e f।"]
    assert stats.oversized_sentences == 1
    assert all(word_count(c) <= 2 for c in chunks)


def test_chunk_text_short_text_is_single_chunk():
    chunks, stats = chunk_text("একটি ছোট বাক্য।", word_count, max_tokens=400)
    assert chunks == ["একটি ছোট বাক্য।"]
    assert stats.chunks == 1


def test_chunk_text_empty():
    chunks, stats = chunk_text("   ", word_count, max_tokens=400)
    assert chunks == []
    assert stats.chunks == 0
