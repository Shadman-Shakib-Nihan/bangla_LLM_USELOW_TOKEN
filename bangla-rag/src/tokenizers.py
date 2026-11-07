"""Tokenizer wrappers and tokenizer-fertility helpers for Bangla/English.

Phase 1 compares how three tokenizers segment parallel Bangla/English text:

* Gemini   - via the ``count_tokens`` API (requires GEMINI_API_KEY)
* o200k_base - OpenAI's GPT-4o tokenizer via ``tiktoken`` (local)
* Qwen2.5   - the Qwen2.5 tokenizer via HuggingFace ``tokenizers`` (local)

All counters implement a tiny protocol: ``name``, ``count(text) -> int`` and
``count_many(list[str]) -> list[int]``.  This keeps the fertility experiment and
the unit tests independent of any particular tokenizer implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Load bangla-rag/.env if present so GEMINI_API_KEY can live there (gitignored).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv is in requirements, but keep import soft
    pass

# Current stable model name taken from the Gemini docs (see README for the exact
# page).  Verified with a one-token call when GEMINI_API_KEY is available; the
# experiment scripts record the name they actually used.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_QWEN_REPO = "Qwen/Qwen2.5-0.5B-Instruct"


class Counter(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def count_many(self, texts: Sequence[str]) -> list[int]: ...


# --------------------------------------------------------------------------- #
# FLORES-200 loading
# --------------------------------------------------------------------------- #
def load_flores_pairs(
    split: str = "dev",
    root: Path = RAW_DIR / "flores200_dataset",
    limit: int | None = None,
    bn_lang: str = "ben_Beng",
    en_lang: str = "eng_Latn",
) -> list[tuple[str, str]]:
    """Load parallel Bangla/English sentence pairs from FLORES-200.

    ``split`` is one of ``dev`` or ``devtest``.  Lines are aligned by index
    between the two language files, as guaranteed by the dataset README.
    """
    bn_path = root / split / f"{bn_lang}.{split}"
    en_path = root / split / f"{en_lang}.{split}"
    if not bn_path.exists():
        raise FileNotFoundError(f"missing Bangla file: {bn_path}")
    if not en_path.exists():
        raise FileNotFoundError(f"missing English file: {en_path}")

    with bn_path.open(encoding="utf-8") as f_bn, en_path.open(encoding="utf-8") as f_en:
        pairs = [
            (bn.rstrip("\n"), en.rstrip("\n"))
            for bn, en in zip(f_bn, f_en)
        ]

    if len(pairs) == 0:
        raise ValueError(f"no sentence pairs found in {bn_path} / {en_path}")
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #
@dataclass
class TiktokenCounter:
    name: str = "o200k_base"
    encoding_name: str = "o200k_base"

    def __post_init__(self) -> None:
        import tiktoken

        # Cache the BPE vocabulary inside the project so reruns work offline.
        os.environ.setdefault(
            "TIKTOKEN_CACHE_DIR", str(CACHE_DIR / "tiktoken")
        )
        self._enc = tiktoken.get_encoding(self.encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [self.count(t) for t in texts]


@dataclass
class QwenCounter:
    name: str = "qwen2.5"
    repo_id: str = DEFAULT_QWEN_REPO

    def __post_init__(self) -> None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
        tokenizer_path = hf_hub_download(
            repo_id=self.repo_id, filename="tokenizer.json"
        )
        self._tok = Tokenizer.from_file(tokenizer_path)

    def count(self, text: str) -> int:
        # No special tokens: FLORES sentences are raw text, matching tiktoken.
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: Sequence[str]) -> list[int]:
        # Batched encoding is much faster than one call per sentence.
        encodings = self._tok.encode_batch(
            list(texts), add_special_tokens=False
        )
        return [len(enc.ids) for enc in encodings]


class GeminiCounter:
    """Token counter backed by the Gemini ``count_tokens`` endpoint.

    Token counts are cached to ``data/cache/gemini_token_counts.json`` keyed by
    model name + text hash so repeated runs do not hammer the API.  Counting
    tokens is free, but the cache keeps runs reproducible and fast.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        cache_path: Path | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it (do not hardcode) before "
                "running the Gemini tokenizer."
            )
        from google import genai

        # The SDK reads the key from the argument; it is never logged.
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"
        self._cache_path = cache_path or (CACHE_DIR / "gemini_token_counts.json")
        self._cache: dict[str, int] = self._load_cache()

    def _load_cache(self) -> dict[str, int]:
        if self._cache_path.exists():
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
        )

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.model}:{digest}"

    def count(self, text: str) -> int:
        key = self._key(text)
        if key not in self._cache:
            self._cache[key] = self._count_remote(text)
            self._save_cache()
        return self._cache[key]

    def _count_remote(self, text: str, max_attempts: int = 8) -> int:
        """Call count_tokens with retries on transient server/rate-limit errors."""
        import random
        import time

        from google.genai import errors as genai_errors

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = self._client.models.count_tokens(
                    model=self.model, contents=text
                )
                return int(response.total_tokens)
            except (genai_errors.ServerError, genai_errors.ClientError) as exc:
                code = getattr(exc, "code", None)
                if code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                    delay = min(60.0, 2.0 ** attempt) + random.random()
                    print(
                        f"[retry] Gemini {code}; sleeping {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [self.count(t) for t in texts]


def make_counter(kind: str) -> Counter:
    kind = kind.lower()
    if kind in {"gemini", "gemini-api"}:
        return GeminiCounter()
    if kind in {"o200k", "o200k_base", "tiktoken", "gpt-o200k"}:
        return TiktokenCounter()
    if kind in {"qwen", "qwen2.5", "qwen25"}:
        return QwenCounter()
    raise ValueError(f"unknown tokenizer kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Fertility statistics
# --------------------------------------------------------------------------- #
def compute_fertility(
    pairs: Sequence[tuple[str, str]], counter: Counter
) -> dict[str, object]:
    """Compute per-sentence token counts and Bangla/English fertility stats.

    Returns a dict with the raw per-sentence counts plus aggregate statistics.
    Ratios are computed both per sentence (then averaged) and on the corpus
    totals; reporting both avoids one long sentence dominating the mean.
    """
    bn_texts = [bn for bn, _ in pairs]
    en_texts = [en for _, en in pairs]
    bn_tokens = counter.count_many(bn_texts)
    en_tokens = counter.count_many(en_texts)

    per_sentence_ratio = [
        bn / en for bn, en in zip(bn_tokens, en_tokens) if en > 0
    ]
    total_bn = sum(bn_tokens)
    total_en = sum(en_tokens)

    return {
        "tokenizer": counter.name,
        "n_sentences": len(pairs),
        "bn_tokens": bn_tokens,
        "en_tokens": en_tokens,
        "mean_bn_tokens": total_bn / len(pairs),
        "mean_en_tokens": total_en / len(pairs),
        "mean_ratio_per_sentence": sum(per_sentence_ratio)
        / len(per_sentence_ratio),
        "corpus_ratio": total_bn / total_en,
        "total_bn_tokens": total_bn,
        "total_en_tokens": total_en,
    }


def summarize(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Strip per-sentence arrays, leaving the scalar summary fields."""
    keep = (
        "tokenizer",
        "n_sentences",
        "mean_bn_tokens",
        "mean_en_tokens",
        "mean_ratio_per_sentence",
        "corpus_ratio",
        "total_bn_tokens",
        "total_en_tokens",
    )
    return [{k: row[k] for k in keep} for row in rows]
