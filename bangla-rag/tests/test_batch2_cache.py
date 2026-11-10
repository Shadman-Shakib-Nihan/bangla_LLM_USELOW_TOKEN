from __future__ import annotations

import json

from src.batch2_gemini import (
    ANSWER_SCHEMA,
    GeminiCallSpec,
    SQLiteGeminiCache,
    parse_json_text,
    parse_usage,
)


def test_cache_key_changes_with_model_prompt_and_settings(tmp_path) -> None:
    base = GeminiCallSpec(model="m1", prompt="p", temperature=0.0, thinking_budget=0)
    assert base.cache_key() == base.cache_key()
    assert base.cache_key() != GeminiCallSpec(model="m2", prompt="p", temperature=0.0, thinking_budget=0).cache_key()
    assert base.cache_key() != GeminiCallSpec(model="m1", prompt="q", temperature=0.0, thinking_budget=0).cache_key()
    assert base.cache_key() != GeminiCallSpec(model="m1", prompt="p", temperature=0.1, thinking_budget=0).cache_key()
    assert base.cache_key() != GeminiCallSpec(model="m1", prompt="p", temperature=0.0, thinking_budget=128).cache_key()


def test_cache_put_get(tmp_path) -> None:
    cache = SQLiteGeminiCache(tmp_path / "cache.sqlite")
    spec = GeminiCallSpec(model="m1", prompt="p", response_schema=ANSWER_SCHEMA, thinking_budget=0)
    raw = {
        "candidates": [{"content": {"parts": [{"text": '{"question":"q","gold_answer":"a","question_type":"what"}'}]}}],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": 3,
            "totalTokenCount": 21,
        },
    }
    from src.batch2_gemini import GeminiResult, GeminiUsage

    result = GeminiResult(
        text='{"question":"q","gold_answer":"a","question_type":"what"}',
        usage=parse_usage(raw),
        raw=raw,
        cached=False,
        model="m1",
        cache_key=spec.cache_key(),
    )
    cache.put(spec, result)
    hit = cache.get(spec.cache_key())
    assert hit is not None
    assert hit["model"] == "m1"
    assert hit["input_tokens"] == 11
    assert hit["thinking_tokens"] == 3
    assert cache.count() == 1


def test_parse_json_text_code_fence() -> None:
    assert parse_json_text('```json\n{"a":1}\n```') == {"a": 1}
