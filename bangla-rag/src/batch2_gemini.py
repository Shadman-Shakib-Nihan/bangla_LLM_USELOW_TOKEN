"""Gemini + SQLite cache primitives for Batch 2 (no live calls on import)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.batch2_text import BATCH2_SEED, sha256_json

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_CACHE = Path("data/cache/custom_batch2_gemini.sqlite")
DEFAULT_DOCS_URL = "https://ai.google.dev/gemini-api/docs/models"
DEFAULT_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"


class GeminiAPIError(RuntimeError):
    """Structured Gemini HTTP error carrying status, body, and retry hint."""

    def __init__(self, status: int, body: str) -> None:
        self.status = int(status)
        self.body = body or ""
        self.retry_after_seconds = _parse_retry_after(self.body)
        super().__init__(f"Gemini API HTTP {self.status}: {self.body[:200]}")

    @property
    def is_daily_quota_exhausted(self) -> bool:
        text = self.body.lower()
        return self.status == 429 and (
            "per day" in text
            or "perday" in text
            or "free_tier" in text
            or "free tier" in text
            or "resource_exhausted" in text
        ) and not self._looks_like_per_minute()

    def _looks_like_per_minute(self) -> bool:
        text = self.body.lower()
        return "per minute" in text or "perminute" in text

    @property
    def is_rate_limit(self) -> bool:
        return self.status == 429 and not self.is_daily_quota_exhausted

    @property
    def is_transient(self) -> bool:
        return self.status in {500, 502, 503, 504}


def _parse_retry_after(body: str) -> float | None:
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)h([0-9]+)m([0-9]+(?:\.[0-9]+)?)s", body)
    if match:
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", body)
    if match:
        return float(match.group(1))
    match = re.search(r"retry in ([0-9]+)m([0-9]+(?:\.[0-9]+)?)s", body)
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))
    match = re.search(r"retryDelay\"?\s*:\s*\"([0-9.]+)s\"", body)
    if match:
        return float(match.group(1))
    return None


def compute_backoff(attempt: int, retry_after: float | None = None, *, base: float = 2.0, factor: float = 2.0, cap: float = 60.0) -> float:
    """Bounded exponential backoff for transient/rate-limit errors.

    ``attempt`` is 1-based. If the server supplies a ``retry_after`` within the
    cap we honor it, otherwise we use ``base * factor**(attempt-1)`` capped.
    """
    delay = base * (factor ** max(0, attempt - 1))
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(cap, delay)


def classify_gemini_error(exc: BaseException) -> str:
    if isinstance(exc, GeminiAPIError):
        if exc.is_daily_quota_exhausted:
            return "daily_quota_exhausted"
        if exc.is_rate_limit:
            return "rate_limit"
        if exc.is_transient:
            return "transient"
        return "api_error"
    if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
        return "timeout"
    return "other"


ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "question": {"type": "STRING"},
        "gold_answer": {"type": "STRING"},
        "question_type": {"type": "STRING"},
        "reasoning_brief": {"type": "STRING"},
    },
    "required": ["question", "gold_answer", "question_type"],
}

UNANSWERABLE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "question": {"type": "STRING"},
        "question_type": {"type": "STRING"},
        "why_unanswerable": {"type": "STRING"},
        "near_miss_fact": {"type": "STRING"},
    },
    "required": ["question", "question_type", "why_unanswerable"],
}


@dataclass(frozen=True)
class GeminiCallSpec:
    model: str
    prompt: str
    temperature: float = 0.0
    seed: int = BATCH2_SEED
    max_output_tokens: int = 512
    response_schema: dict[str, Any] | None = None
    thinking_budget: int | None = None
    system_instruction: str | None = None

    def generation_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
            "seed": self.seed,
        }
        if self.thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        if self.response_schema is not None:
            cfg["responseMimeType"] = "application/json"
            cfg["responseSchema"] = self.response_schema
        return cfg

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": self.prompt}]}],
            "generationConfig": self.generation_config(),
        }
        if self.system_instruction:
            body["systemInstruction"] = {"parts": [{"text": self.system_instruction}]}
        return body

    def cache_key(self) -> str:
        return sha256_json(
            {
                "model": self.model,
                "prompt": self.prompt,
                "system_instruction": self.system_instruction or "",
                "generation_config": self.generation_config(),
            }
        )


@dataclass(frozen=True)
class GeminiUsage:
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class GeminiResult:
    text: str
    usage: GeminiUsage
    raw: dict[str, Any]
    cached: bool
    model: str
    cache_key: str


class SQLiteGeminiCache:
    def __init__(self, path: str | Path = DEFAULT_CACHE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                generation_config_json TEXT NOT NULL,
                system_instruction_hash TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_text TEXT NOT NULL,
                response_json TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                thinking_tokens INTEGER,
                total_tokens INTEGER,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM responses WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self.conn.execute("SELECT * FROM responses LIMIT 0").description]
        return dict(zip(cols, row, strict=False))

    def put(self, spec: GeminiCallSpec, result: GeminiResult) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO responses (
                cache_key, model, prompt_hash, generation_config_json,
                system_instruction_hash, request_json, response_text,
                response_json, input_tokens, output_tokens, thinking_tokens,
                total_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.cache_key(),
                spec.model,
                sha256_json(spec.prompt),
                json.dumps(spec.generation_config(), ensure_ascii=False, sort_keys=True),
                sha256_json(spec.system_instruction or ""),
                json.dumps(spec.payload(), ensure_ascii=False),
                result.text,
                json.dumps(result.raw, ensure_ascii=False),
                result.usage.input_tokens,
                result.usage.output_tokens,
                result.usage.thinking_tokens,
                result.usage.total_tokens,
                time.time(),
            ),
        )
        self.conn.commit()

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(row[0] if row else 0)


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        cache: SQLiteGeminiCache | None = None,
        timeout: int = 90,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.cache = cache or SQLiteGeminiCache()
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _request(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise GeminiAPIError(exc.code, body) from exc

    def list_models(self) -> dict[str, Any]:
        return self._request(f"{API_BASE}?key={self.api_key}")

    def generate(self, spec: GeminiCallSpec, use_cache: bool = True) -> GeminiResult:
        key = spec.cache_key()
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                raw = json.loads(hit["response_json"])
                return GeminiResult(
                    text=hit["response_text"],
                    usage=GeminiUsage(
                        hit["input_tokens"], hit["output_tokens"],
                        hit["thinking_tokens"], hit["total_tokens"],
                    ),
                    raw=raw,
                    cached=True,
                    model=spec.model,
                    cache_key=key,
                )
        raw = self._request(f"{API_BASE}/{spec.model}:generateContent", spec.payload())
        text = extract_text(raw)
        usage = parse_usage(raw)
        result = GeminiResult(
            text=text, usage=usage, raw=raw, cached=False, model=spec.model, cache_key=key
        )
        if use_cache:
            self.cache.put(spec, result)
        return result

    def one_token_verification(self, model: str) -> GeminiResult:
        spec = GeminiCallSpec(
            model=model,
            prompt='Return JSON {"ok":true}.',
            temperature=0.0,
            seed=BATCH2_SEED,
            max_output_tokens=8,
            response_schema={"type": "OBJECT", "properties": {"ok": {"type": "BOOLEAN"}}, "required": ["ok"]},
        )
        return self.generate(spec, use_cache=False)


def extract_text(raw: dict[str, Any]) -> str:
    candidates = raw.get("candidates") or []
    if not candidates:
        return ""
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    return "\n".join(texts).strip()


def parse_usage(raw: dict[str, Any]) -> GeminiUsage:
    meta = raw.get("usageMetadata") or {}
    def _int(key: str) -> int | None:
        value = meta.get(key)
        return None if value is None else int(value)
    return GeminiUsage(
        input_tokens=_int("promptTokenCount"),
        output_tokens=_int("candidatesTokenCount"),
        thinking_tokens=_int("thoughtsTokenCount") or _int("thinkingTokenCount"),
        total_tokens=_int("totalTokenCount"),
    )


def parse_json_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)
