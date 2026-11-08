"""Draft custom answerable and unanswerable questions from the corpus.

The tool samples parent passages with a fixed seed, asks a cheap Gemini model
for Bangla question drafts, runs automatic checks, and writes review CSVs. It
never edits ``data/custom_questions.csv``.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/build_custom_drafts.py --mode estimate
    .venv/bin/python experiments/build_custom_drafts.py --mode run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import build_parent_index, rank_parents, tokenize  # noqa: E402

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
UNITS = PROJECT_ROOT / "data" / "processed" / "corpus_chunks.jsonl"
UNIT_EMB = PROJECT_ROOT / "data" / "embeddings" / "corpus_units.f16.npy"
OUT_REVIEW = PROJECT_ROOT / "data" / "custom_drafts.csv"
OUT_UNANSWERABLE_RETRIEVAL = (
    PROJECT_ROOT / "data" / "custom_drafts_unanswerable_retrieval.csv"
)
OUT_ESTIMATE = PROJECT_ROOT / "results" / "custom_drafts_cost_estimate.json"
OUT_PILOT = PROJECT_ROOT / "results" / "custom_drafts_pilot.jsonl"
OUT_MANIFEST = PROJECT_ROOT / "results" / "custom_drafts_manifest.json"
CACHE_DB = PROJECT_ROOT / "data" / "cache" / "custom_drafts.sqlite"

E5_MODEL = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
MAX_SEQ_LEN = 512

DEFAULT_MODEL = "gemini-3.1-flash-lite"
MODEL_PRICING = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}

SEED = 42
MIN_TOKENS = 40
MAX_TOKENS = 300
N_ANSWERABLE = 150
N_UNANSWERABLE = 30
N_TOTAL = N_ANSWERABLE + N_UNANSWERABLE
N_SAMPLE = 220
ANSWERABLE_POOL_N = 170
UNANSWERABLE_POOL_N = 50
ESTIMATE_N = 10

QUESTION_TYPES = ["who", "when", "where", "what", "how_many", "why_how"]
TYPE_ALIASES = {
    "who": "who",
    "when": "when",
    "where": "where",
    "what": "what",
    "how_many": "how_many",
    "how many": "how_many",
    "howmany": "how_many",
    "why": "why_how",
    "how": "why_how",
    "why_how": "why_how",
    "why/how": "why_how",
}
TYPE_INSTRUCTIONS = {
    "who": "Ask who did, held, or was involved (কে / কার / কাদের).",
    "when": "Ask for a year or date (কত সালে / কোন সালে / কবে).",
    "where": "Ask for a place (কোথায় / কোন স্থানে / কোন শহরে).",
    "what": "Ask what something is, was, or which thing.",
    "how_many": "Ask for a count (কতজন / কতটি / কত সংখ্যক).",
    "why_how": "Ask why or how (কেন / কীভাবে). Keep the answer a short verbatim phrase.",
}
ANSWERABLE_TYPE_PLAN = {
    "who": 25, "when": 41, "where": 20,
    "what": 35, "how_many": 13, "why_how": 16,
}
UNANSWERABLE_TYPE_PLAN = {
    "who": 5, "when": 7, "where": 5,
    "what": 7, "how_many": 2, "why_how": 4,
}
MAX_WHEN_ANSWERABLE = 47
MAX_WHEN_UNANSWERABLE = 7
MAX_NAME_HEAVY_ANSWERABLE = 15
MAX_NAME_HEAVY_UNANSWERABLE = 3
MAX_ATTEMPTS_PER_ITEM = 3

REVIEW_FIELDS = [
    "id",
    "question",
    "question_type",
    "gold_answer",
    "passage_snippet",
    "gold_passage_id",
    "alt_gold_ids",
    "answerable",
    "language_pair",
    "answer_is_substring",
    "max_shared_word_4gram",
    "check_flags",
    "my_decision",
    "my_notes",
]

UNANSWERABLE_RETRIEVAL_FIELDS = [
    "id",
    "question",
    "question_type",
    "premise_type",
    "rank",
    "passage_id",
    "dense_score",
    "title",
    "snippet",
]

DRAFT_SYSTEM = """You create evaluation questions for a Bangla retrieval benchmark.
Return JSON only with the keys "question", "answer", and "question_type".
Rules:
- The question must be self-contained, written in natural everyday Bangla (Bengali script), and not a yes/no question.
- Name the person, place, event, work, or organisation explicitly. Do not use vague references such as এই ব্যক্তি, এই স্থান, এই ঘটনা, উক্ত, or তিনি.
- The question must be understandable without seeing the passage.
- The answer must be one short contiguous span copied verbatim from the passage.
- The question must not copy any four-word sequence from the passage.
- Use a short natural descriptor instead of a long transliterated name.
- For birth, write জন্মগ্রহণ করেন. Never write পৃথিবীতে এসেছিলেন.
- Do not ask about current holders, current designs, current counts, or current rankings. If the passage gives a specific dated fact, state that date in the question.
- Prefer stable facts. Questions that ask for a year/date are allowed, but do not overuse them.
- The question_type must be exactly one of: who, when, where, what, how_many, why_how.
- Do not mention "passage", "text", or "according to".
- Do not answer the question inside the question.
"""

UNANSWERABLE_SYSTEM = """You create unanswerable evaluation questions for a Bangla retrieval benchmark.
Return JSON only with the keys "question", "premise_type", and "question_type".
Rules:
- The question must be self-contained, written in natural everyday Bangla (Bengali script), and not a yes/no question.
- Name the person, place, event, work, or organisation explicitly. Do not use vague references such as এই ব্যক্তি, এই স্থান, এই ঘটনা, উক্ত, or তিনি.
- The question must be understandable without seeing the passage.
- Use a short natural descriptor instead of a long transliterated name.
- For birth, write জন্মগ্রহণ করেন. Never write পৃথিবীতে এসেছিলেন.
- Do not ask about current holders, current designs, current counts, or current rankings. If the passage gives a specific dated fact, state that date in the question.
- The question must be about the same topic as the passage but must NOT be answerable from it.
- Use the requested unanswerable style: missing_detail, false_premise, or not_in_passage.
- The question must not copy any four-word sequence from the passage.
- The question_type must be exactly one of: who, when, where, what, how_many, why_how.
- Do not mention "passage", "text", or "according to".
- Do not provide an answer.
"""

ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "question": {"type": "STRING"},
        "answer": {"type": "STRING"},
        "question_type": {"type": "STRING"},
    },
    "required": ["question", "answer", "question_type"],
}

UNANSWERABLE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "question": {"type": "STRING"},
        "premise_type": {"type": "STRING"},
        "question_type": {"type": "STRING"},
    },
    "required": ["question", "premise_type", "question_type"],
}


@dataclass(frozen=True)
class SampledPassage:
    passage_id: str
    parent_id: str
    article_key: str
    title: str
    text: str
    source: str
    e5_tokens: int


@dataclass
class DraftResult:
    parsed: dict[str, Any]
    response_text: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    finish_reason: str
    cached: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def nfc_contains(answer: str, passage: str) -> bool:
    answer = nfc(answer).strip()
    return bool(answer) and answer in nfc(passage)


def article_key(rec: dict) -> str:
    source = str(rec.get("source", ""))
    if source == "miracl":
        return "miracl:" + str(rec["passage_id"]).split("#")[0]
    title = str(rec.get("title", "")).strip()
    if title:
        return f"{source}:{title}"
    return f"{source}:{rec['passage_id']}"


def max_shared_word_4gram(question: str, passage: str) -> int:
    q = tokenize(question)
    p = tokenize(passage)
    if len(q) < 4 or len(p) < 4:
        return 0
    passage_grams = {tuple(p[i:i + 4]) for i in range(len(p) - 3)}
    return sum(
        1 for i in range(len(q) - 3)
        if tuple(q[i:i + 4]) in passage_grams
    )


def infer_question_type(question: str) -> str:
    q = nfc(question)
    if re.search(r"কত\s*সালে|কোন\s*সালে|কবে|কত\s*তারিখ|কোন\s*তারিখ", q):
        return "when"
    if re.search(r"কোথায়|কোন\s*স্থানে|কোন\s*শহরে|কোন\s*দেশে", q):
        return "where"
    if re.search(r"কীভাবে|কিভাবে|কেন|কী\s*কারণে", q):
        return "why_how"
    if re.search(r"কতজন|কতটি|কয়টি|কত\s*সংখ্যক|কত\s*জন", q):
        return "how_many"
    if re.search(r"(^|\s)কে(\s|$)|কার|কাদের|কোন\s*ব্যক্তি", q):
        return "who"
    return "what"


def classify_question_type(raw_type: Any, question: str) -> str:
    key = str(raw_type or "").strip().lower().replace("-", "_")
    key = TYPE_ALIASES.get(key, key)
    heuristic = infer_question_type(question)
    if heuristic == "when":
        return "when"
    if key in QUESTION_TYPES:
        return key
    return heuristic


def has_explicit_year(question: str) -> bool:
    return bool(re.search(r"[0-9০-৯]{4}", question))


def is_time_sensitive_undated(question: str) -> bool:
    marker = re.search(
        r"বর্তমান|বর্তমানে|এখন|সর্বশেষ|সাম্প্রতিক|অদ্যাবধি|এ\s*যাবৎ|"
        r"সর্বোচ্চ|সর্বাধিক|নতুন\s*নিয়ম|বর্তমান\s*রেকর্ড",
        question,
    )
    return bool(marker) and not has_explicit_year(question)


def is_name_heavy(question: str) -> bool:
    runs = re.findall(r"[\u0980-\u09ffA-Za-z'’\-]+", nfc(question))
    lengths = [len(run) for run in runs]
    if not lengths:
        return False
    return max(lengths) >= 24 or sum(1 for n in lengths if n >= 16) >= 3


def has_forbidden_birth_phrase(question: str) -> bool:
    return "পৃথিবীতে এসেছিলেন" in question or "পৃথিবীতে আসেন" in question


def passage_snippet(text: str, answer: str = "", width: int = 500) -> str:
    text = str(text)
    answer = nfc(answer).strip()
    if answer:
        pos = text.find(answer)
        if pos == -1:
            pos = nfc(text).find(answer)
        if pos >= 0:
            start = max(0, pos - width // 3)
            end = min(len(text), start + width)
            if end - start < width:
                start = max(0, end - width)
            snippet = text[start:end]
            return (
                ("..." if start > 0 else "")
                + snippet
                + ("..." if end < len(text) else "")
            )
    snippet = text[:width]
    return snippet + ("..." if len(text) > width else "")


def make_type_plan(counts: dict[str, int], seed: int) -> list[str]:
    plan = [qtype for qtype, n in counts.items() for _ in range(n)]
    random.Random(seed).shuffle(plan)
    return plan


def empty_stats() -> dict:
    return {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "fresh_api_calls": 0,
        "cache_hits": 0,
        "rejected_attempts": 0,
    }


def add_usage(stats: dict, result: DraftResult) -> None:
    stats["prompt_tokens"] += result.prompt_tokens
    stats["output_tokens"] += result.output_tokens
    stats["total_tokens"] += result.total_tokens
    if result.cached:
        stats["cache_hits"] += 1
    else:
        stats["fresh_api_calls"] += 1


def load_corpus() -> list[dict]:
    with CORPUS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(E5_MODEL, local_files_only=True)


def e5_token_lengths(records: list[dict], tokenizer, batch_size: int = 512) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        enc = tokenizer(
            [PASSAGE_PREFIX + r["text"] for r in batch],
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )
        lengths.extend(int(x) for x in enc["length"])
        if start and start % (batch_size * 20) == 0:
            print(f"  tokenized {start:,}/{len(records):,}", flush=True)
    return lengths


def select_sample(
    records: list[dict],
    lengths: list[int],
    seed: int = SEED,
    n_total: int = N_TOTAL,
) -> list[SampledPassage]:
    groups: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    for rec, n_tok in zip(records, lengths):
        if rec.get("is_short"):
            continue
        if not (MIN_TOKENS <= n_tok <= MAX_TOKENS):
            continue
        groups[article_key(rec)].append((rec, n_tok))

    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    if len(keys) < n_total:
        raise RuntimeError(
            f"only {len(keys)} eligible articles; need {n_total}"
        )

    sample: list[SampledPassage] = []
    for key in keys[:n_total]:
        rec, n_tok = rng.choice(groups[key])
        sample.append(SampledPassage(
            passage_id=str(rec["passage_id"]),
            parent_id=str(rec.get("parent_id", rec["passage_id"])),
            article_key=key,
            title=str(rec.get("title", "")),
            text=str(rec["text"]),
            source=str(rec.get("source", "")),
            e5_tokens=n_tok,
        ))
    return sample


class SQLiteGenerationCache:
    def __init__(self, path: Path = CACHE_DB) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                system_instruction TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                response_text TEXT NOT NULL,
                response_json TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                finish_reason TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def make_key(
        model: str,
        system_instruction: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        response_schema: dict | None,
    ) -> str:
        payload = json.dumps({
            "model": model,
            "system_instruction": system_instruction,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_schema": response_schema,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict | None:
        cur = self.conn.execute(
            """
            SELECT response_text, response_json, prompt_tokens, output_tokens,
                   total_tokens, finish_reason, created_at
            FROM generations WHERE cache_key = ?
            """,
            (cache_key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "response_text": row[0],
            "response_json": json.loads(row[1]),
            "prompt_tokens": int(row[2]),
            "output_tokens": int(row[3]),
            "total_tokens": int(row[4]),
            "finish_reason": row[5] or "",
            "created_at": row[6],
        }

    def put(
        self,
        cache_key: str,
        model: str,
        system_instruction: str,
        user_prompt: str,
        response_text: str,
        parsed: dict,
        prompt_tokens: int,
        output_tokens: int,
        total_tokens: int,
        finish_reason: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO generations (
                cache_key, model, system_instruction, user_prompt,
                response_text, response_json, prompt_tokens, output_tokens,
                total_tokens, finish_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key, model, system_instruction, user_prompt,
                response_text, json.dumps(parsed, ensure_ascii=False),
                prompt_tokens, output_tokens, total_tokens, finish_reason,
                utc_now(),
            ),
        )
        self.conn.commit()

    def count(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM generations"
        ).fetchone()[0])


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def extract_usage(response) -> tuple[int, int, int, str]:
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    candidates = int(getattr(usage, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
    total = int(getattr(usage, "total_token_count", 0) or 0)
    if candidates == 0 and total > prompt_tokens:
        candidates = total - prompt_tokens
    output_tokens = candidates + thoughts
    finish = ""
    try:
        finish = str(response.candidates[0].finish_reason)
    except Exception:
        pass
    return prompt_tokens, output_tokens, total, finish


class GeminiDrafter:
    def __init__(self, model: str, cache: SQLiteGenerationCache) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.cache = cache

    def verify_one_token(self) -> dict:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents="Reply with exactly: OK",
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1,
            ),
        )
        prompt_tokens, output_tokens, total_tokens, finish = extract_usage(response)
        return {
            "model": self.model,
            "verified": True,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish,
            "checked_at": utc_now(),
        }

    def _generate(self, system_instruction: str, user_prompt: str,
                  response_schema: dict, max_output_tokens: int) -> Any:
        from google.genai import errors as genai_errors
        from google.genai import types

        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                        max_output_tokens=max_output_tokens,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                    ),
                )
            except (genai_errors.ServerError, genai_errors.ClientError) as exc:
                code = getattr(exc, "code", None)
                if code in {429, 500, 502, 503, 504} and attempt < 5:
                    delay = min(60.0, 2.0 ** attempt) + random.random()
                    print(f"[retry] Gemini {code}; sleeping {delay:.1f}s")
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def draft(self, system_instruction: str, user_prompt: str,
              response_schema: dict, max_output_tokens: int = 256) -> DraftResult:
        cache_key = SQLiteGenerationCache.make_key(
            self.model, system_instruction, user_prompt, 0.0,
            max_output_tokens, response_schema,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return DraftResult(
                parsed=cached["response_json"],
                response_text=cached["response_text"],
                prompt_tokens=cached["prompt_tokens"],
                output_tokens=cached["output_tokens"],
                total_tokens=cached["total_tokens"],
                finish_reason=cached["finish_reason"],
                cached=True,
            )

        response = self._generate(
            system_instruction, user_prompt, response_schema, max_output_tokens
        )
        response_text = getattr(response, "text", "") or ""
        parsed = parse_json_response(response_text)
        prompt_tokens, output_tokens, total_tokens, finish = extract_usage(response)
        self.cache.put(
            cache_key, self.model, system_instruction, user_prompt,
            response_text, parsed, prompt_tokens, output_tokens,
            total_tokens, finish,
        )
        return DraftResult(
            parsed=parsed,
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            finish_reason=finish,
            cached=False,
        )


def usage_cost_usd(prompt_tokens: int, output_tokens: int, model: str) -> float:
    price = MODEL_PRICING[model]
    return (
        prompt_tokens * price["input"] + output_tokens * price["output"]
    ) / 1_000_000


def answerable_prompt(
    rec: SampledPassage,
    requested_type: str | None = None,
    extra_instruction: str = "",
) -> str:
    lines = []
    if requested_type:
        lines.append(f"Requested question_type: {requested_type}")
        lines.append(TYPE_INSTRUCTIONS[requested_type])
    if extra_instruction:
        lines.append(extra_instruction)
    lines.append(f"Passage title: {rec.title}")
    lines.append("")
    lines.append("Passage:")
    lines.append(rec.text)
    return "\n".join(lines)


def unanswerable_prompt(
    rec: SampledPassage,
    premise_type: str,
    requested_type: str | None = None,
    extra_instruction: str = "",
) -> str:
    lines = [f"Requested unanswerable style: {premise_type}"]
    if requested_type:
        lines.append(f"Requested question_type: {requested_type}")
        lines.append(TYPE_INSTRUCTIONS[requested_type])
    if extra_instruction:
        lines.append(extra_instruction)
    lines.append(f"Passage title: {rec.title}")
    lines.append("")
    lines.append("Passage:")
    lines.append(rec.text)
    return "\n".join(lines)


def build_answerable_row(
    idx: int,
    rec: SampledPassage,
    draft: DraftResult,
    normalized_corpus: list[tuple[str, str]],
    question_type: str | None = None,
) -> dict:
    question = str(draft.parsed.get("question", "")).strip()
    answer = str(draft.parsed.get("answer", "")).strip()
    qtype = question_type or classify_question_type(
        draft.parsed.get("question_type"), question
    )
    answer_ok = nfc_contains(answer, rec.text)
    shared = max_shared_word_4gram(question, rec.text)
    alt_ids = [
        pid for pid, text in normalized_corpus
        if pid != rec.passage_id and nfc(answer).strip() in text
    ] if answer else []

    flags = []
    if not answer_ok:
        flags.append("answer_not_substring")
    if shared > 0:
        flags.append(f"shared_4gram={shared}")
    if is_name_heavy(question):
        flags.append("name_heavy")
    if is_time_sensitive_undated(question):
        flags.append("time_sensitive_undated")

    return {
        "id": f"cd_{idx:04d}",
        "question": question,
        "question_type": qtype,
        "gold_answer": answer,
        "passage_snippet": passage_snippet(rec.text, answer),
        "gold_passage_id": rec.passage_id,
        "alt_gold_ids": ";".join(alt_ids),
        "answerable": "true",
        "language_pair": "bn-bn",
        "answer_is_substring": "true" if answer_ok else "false",
        "max_shared_word_4gram": str(shared),
        "check_flags": ";".join(flags),
        "my_decision": "",
        "my_notes": "",
    }


def build_unanswerable_row(
    idx: int,
    rec: SampledPassage,
    draft: DraftResult,
    question_type: str | None = None,
) -> dict:
    question = str(draft.parsed.get("question", "")).strip()
    qtype = question_type or classify_question_type(
        draft.parsed.get("question_type"), question
    )
    shared = max_shared_word_4gram(question, rec.text)
    flags = ["manual_unanswerable_review"]
    if shared > 0:
        flags.append(f"shared_4gram={shared}")
    if is_name_heavy(question):
        flags.append("name_heavy")
    if is_time_sensitive_undated(question):
        flags.append("time_sensitive_undated")
    return {
        "id": f"cd_{idx:04d}",
        "question": question,
        "question_type": qtype,
        "gold_answer": "",
        "passage_snippet": passage_snippet(rec.text),
        "gold_passage_id": "NONE",
        "alt_gold_ids": "",
        "answerable": "false",
        "language_pair": "bn-bn",
        "answer_is_substring": "",
        "max_shared_word_4gram": str(shared),
        "check_flags": ";".join(flags),
        "my_decision": "",
        "my_notes": "",
    }


def candidate_issues(
    question: str,
    qtype: str,
    requested_type: str,
    type_counts: dict[str, int],
    name_heavy_count: int,
    max_when: int,
    max_name_heavy: int,
) -> list[str]:
    issues = []
    if qtype == "when" and type_counts["when"] >= max_when:
        issues.append("date_cap")
    if is_name_heavy(question) and name_heavy_count >= max_name_heavy:
        issues.append("name_cap")
    if has_forbidden_birth_phrase(question):
        issues.append("birth_phrase")
    if is_time_sensitive_undated(question):
        issues.append("time_sensitive")
    if qtype != requested_type:
        issues.append("type_mismatch")
    return issues


def correction_text(issues: list[str], requested_type: str) -> str:
    parts = []
    if "type_mismatch" in issues:
        parts.append(
            f"Use exactly question_type={requested_type}. "
            f"{TYPE_INSTRUCTIONS[requested_type]}"
        )
    if "date_cap" in issues:
        parts.append("Do not ask for a year or date; use the requested question type.")
    if "name_cap" in issues:
        parts.append(
            "Do not build the question mainly around a long transliterated name; "
            "use a short natural descriptor."
        )
    if "birth_phrase" in issues:
        parts.append("For birth, write জন্মগ্রহণ করেন, never পৃথিবীতে এসেছিলেন.")
    if "time_sensitive" in issues:
        parts.append(
            "Do not ask about a current or changeable fact. Ask about a stable fact "
            "or state the specific date given in the passage."
        )
    return " ".join(parts)


def generate_answerable_rows(
    drafter: GeminiDrafter,
    pool: list[SampledPassage],
    plan: list[str],
    target: int,
    normalized_corpus: list[tuple[str, str]],
    max_when: int,
    max_name_heavy: int,
    id_start: int = 1,
) -> tuple[list[dict], dict, dict[str, int], int]:
    rows: list[dict] = []
    type_counts: dict[str, int] = defaultdict(int)
    name_heavy_count = 0
    stats = empty_stats()

    for rec in pool:
        if len(rows) >= target:
            break
        requested_type = plan[len(rows)]
        accepted: tuple[DraftResult, str] | None = None
        issues: list[str] = []
        for attempt in range(MAX_ATTEMPTS_PER_ITEM):
            extra = "" if attempt == 0 else correction_text(issues, requested_type)
            result = drafter.draft(
                DRAFT_SYSTEM,
                answerable_prompt(rec, requested_type, extra),
                ANSWER_SCHEMA,
                max_output_tokens=256,
            )
            add_usage(stats, result)
            question = str(result.parsed.get("question", "")).strip()
            qtype = classify_question_type(
                result.parsed.get("question_type"), question
            )
            issues = candidate_issues(
                question, qtype, requested_type, type_counts,
                name_heavy_count, max_when, max_name_heavy,
            )
            hard = [issue for issue in issues if issue != "type_mismatch"]
            if not issues or (not hard and attempt == MAX_ATTEMPTS_PER_ITEM - 1):
                accepted = (result, qtype)
                break
            stats["rejected_attempts"] += 1

        if accepted is None:
            continue

        result, qtype = accepted
        question = str(result.parsed.get("question", "")).strip()
        row = build_answerable_row(
            id_start + len(rows), rec, result, normalized_corpus, qtype
        )
        rows.append(row)
        type_counts[qtype] += 1
        if is_name_heavy(question):
            name_heavy_count += 1
        if len(rows) % 10 == 0 or len(rows) == target:
            print(f"  answerable {len(rows)}/{target}", flush=True)

    if len(rows) < target:
        raise RuntimeError(
            f"only drafted {len(rows)} answerable rows from pool {len(pool)}"
        )
    return rows, stats, dict(type_counts), name_heavy_count


def generate_unanswerable_rows(
    drafter: GeminiDrafter,
    pool: list[SampledPassage],
    plan: list[str],
    target: int,
    max_when: int,
    max_name_heavy: int,
    id_start: int,
) -> tuple[list[dict], dict, dict[str, int], int, list[dict]]:
    rows: list[dict] = []
    meta: list[dict] = []
    type_counts: dict[str, int] = defaultdict(int)
    name_heavy_count = 0
    stats = empty_stats()
    styles = ["missing_detail", "false_premise", "not_in_passage"]

    for rec in pool:
        if len(rows) >= target:
            break
        requested_type = plan[len(rows)]
        premise_type = styles[len(rows) % len(styles)]
        accepted: tuple[DraftResult, str] | None = None
        issues: list[str] = []
        for attempt in range(MAX_ATTEMPTS_PER_ITEM):
            extra = "" if attempt == 0 else correction_text(issues, requested_type)
            result = drafter.draft(
                UNANSWERABLE_SYSTEM,
                unanswerable_prompt(
                    rec, premise_type, requested_type, extra
                ),
                UNANSWERABLE_SCHEMA,
                max_output_tokens=256,
            )
            add_usage(stats, result)
            question = str(result.parsed.get("question", "")).strip()
            qtype = classify_question_type(
                result.parsed.get("question_type"), question
            )
            issues = candidate_issues(
                question, qtype, requested_type, type_counts,
                name_heavy_count, max_when, max_name_heavy,
            )
            hard = [issue for issue in issues if issue != "type_mismatch"]
            if not issues or (not hard and attempt == MAX_ATTEMPTS_PER_ITEM - 1):
                accepted = (result, qtype)
                break
            stats["rejected_attempts"] += 1

        if accepted is None:
            continue

        result, qtype = accepted
        question = str(result.parsed.get("question", "")).strip()
        row = build_unanswerable_row(
            id_start + len(rows), rec, result, qtype
        )
        rows.append(row)
        meta.append({
            "rec": rec,
            "result": result,
            "qtype": qtype,
            "premise_type": premise_type,
        })
        type_counts[qtype] += 1
        if is_name_heavy(question):
            name_heavy_count += 1
        if len(rows) % 10 == 0 or len(rows) == target:
            print(f"  unanswerable {len(rows)}/{target}", flush=True)

    if len(rows) < target:
        raise RuntimeError(
            f"only drafted {len(rows)} unanswerable rows from pool {len(pool)}"
        )
    return rows, stats, dict(type_counts), name_heavy_count, meta


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_corpus(records: list[dict]) -> list[tuple[str, str]]:
    return [(str(rec["passage_id"]), nfc(str(rec["text"]))) for rec in records]


def projected_cost(avg_prompt: float, avg_output: float, model: str,
                   n_questions: int) -> float:
    return usage_cost_usd(
        int(round(avg_prompt * n_questions)),
        int(round(avg_output * n_questions)),
        model,
    )


def run_estimate(
    sample: list[SampledPassage],
    records: list[dict],
    model: str,
) -> dict:
    n_pilot = min(ESTIMATE_N, len(sample))
    cache = SQLiteGenerationCache()
    drafter = GeminiDrafter(model, cache)
    verification = drafter.verify_one_token()
    verify_cost = usage_cost_usd(
        verification["prompt_tokens"], verification["output_tokens"], model
    )

    norm_corpus = normalized_corpus(records)
    plan = make_type_plan(ANSWERABLE_TYPE_PLAN, SEED + 1)[:n_pilot]
    max_when = max(1, int(n_pilot * 0.30))
    max_name_heavy = max(1, int(n_pilot * 0.10))
    pool = sample[:min(len(sample), n_pilot + 10)]
    rows, stats, type_counts, name_heavy = generate_answerable_rows(
        drafter, pool, plan, n_pilot, norm_corpus, max_when, max_name_heavy
    )

    with OUT_PILOT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(OUT_REVIEW, REVIEW_FIELDS, [])
    write_csv(OUT_UNANSWERABLE_RETRIEVAL, UNANSWERABLE_RETRIEVAL_FIELDS, [])

    prompt_tokens = stats["prompt_tokens"]
    output_tokens = stats["output_tokens"]
    total_tokens = stats["total_tokens"]
    pilot_cost = usage_cost_usd(prompt_tokens, output_tokens, model)
    avg_prompt = prompt_tokens / n_pilot
    avg_output = output_tokens / n_pilot
    full_cost = projected_cost(
        avg_prompt, avg_output, model, N_ANSWERABLE + N_UNANSWERABLE
    )
    per_question = full_cost / (N_ANSWERABLE + N_UNANSWERABLE)

    estimate = {
        "generated_at": utc_now(),
        "model": model,
        "model_verification": verification,
        "pricing_usd_per_1m_tokens": MODEL_PRICING[model],
        "pricing_note": "Paid Standard tier, text input/output, November 2026.",
        "pilot": {
            "n": n_pilot,
            "fresh_api_calls": stats["fresh_api_calls"],
            "cache_hits": stats["cache_hits"],
            "rejected_attempts": stats["rejected_attempts"],
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(pilot_cost, 8),
            "verification_cost_usd": round(verify_cost, 8),
            "type_counts": type_counts,
            "name_heavy": name_heavy,
        },
        "projection": {
            "n_answerable": N_ANSWERABLE,
            "n_unanswerable": N_UNANSWERABLE,
            "n_total": N_ANSWERABLE + N_UNANSWERABLE,
            "assumption": "same average prompt/output tokens as the pilot",
            "avg_prompt_tokens_per_question": round(avg_prompt, 3),
            "avg_output_tokens_per_question": round(avg_output, 3),
            "cost_per_question_usd": round(per_question, 8),
            "total_cost_usd": round(full_cost, 6),
            "date_question_share": round(
                type_counts.get("when", 0) / n_pilot, 4
            ),
        },
        "cache": {
            "path": str(CACHE_DB.relative_to(PROJECT_ROOT)),
            "rows": cache.count(),
        },
        "sample": {
            "seed": SEED,
            "selected_candidate_pool": len(sample),
            "final_answerable": N_ANSWERABLE,
            "final_unanswerable": N_UNANSWERABLE,
        },
        "pilot_rows": rows,
    }
    OUT_ESTIMATE.parent.mkdir(parents=True, exist_ok=True)
    OUT_ESTIMATE.write_text(
        json.dumps(estimate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nCOST ESTIMATE")
    print(f"  model: {model} (one-token test: OK)")
    print(
        f"  pilot: {n_pilot} passages, {stats['fresh_api_calls']} API calls, "
        f"{stats['cache_hits']} cache hits, {stats['rejected_attempts']} rejected attempts"
    )
    print(f"  types: {type_counts}; name_heavy={name_heavy}")
    print(
        f"  tokens: prompt={prompt_tokens:,} output={output_tokens:,} "
        f"total={total_tokens:,}"
    )
    print(f"  pilot cost: ${pilot_cost:.6f} (plus ${verify_cost:.6f} verification)")
    print(f"  avg tokens/question: prompt={avg_prompt:.1f} output={avg_output:.1f}")
    print(
        f"  projected {N_ANSWERABLE + N_UNANSWERABLE} questions: "
        f"${full_cost:.4f} (${per_question:.6f}/question)"
    )
    print(f"  wrote {OUT_ESTIMATE.relative_to(PROJECT_ROOT)}")
    print(f"  wrote header-only {OUT_REVIEW.relative_to(PROJECT_ROOT)}")
    print("\nPILOT ROWS (first 5)")
    for row in rows[:5]:
        print(json.dumps({
            "id": row["id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "answer_is_substring": row["answer_is_substring"],
            "max_shared_word_4gram": row["max_shared_word_4gram"],
            "alt_gold_count": len(row["alt_gold_ids"].split(";")) if row["alt_gold_ids"] else 0,
            "check_flags": row["check_flags"],
        }, ensure_ascii=False))
    return estimate

def load_e5_for_queries():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(E5_MODEL, dtype=torch.float16)
    model = model.to(device).eval()
    return tokenizer, model, device


def embed_queries(
    questions: list[str], tokenizer, model, device, batch_size: int = 16
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    out: list[np.ndarray] = []
    for start in range(0, len(questions), batch_size):
        batch = [QUERY_PREFIX + q for q in questions[start:start + batch_size]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        out.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def load_units_for_dense() -> list[str]:
    parent_ids: list[str] = []
    with UNITS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            parent_ids.append(str(rec["parent_id"]))
    return parent_ids


def dense_top5(
    questions: list[str],
    parent_ids: list[str],
    parent_text: dict[str, str],
    parent_titles: dict[str, str],
) -> list[list[dict]]:
    unit_emb = np.load(UNIT_EMB).astype(np.float32)
    if unit_emb.shape[0] != len(parent_ids):
        raise RuntimeError(
            f"embedding rows {unit_emb.shape[0]} != units {len(parent_ids)}"
        )
    pidx = build_parent_index(parent_ids)

    tokenizer, model, device = load_e5_for_queries()
    qvec = embed_queries(questions, tokenizer, model, device)
    results: list[list[dict]] = []
    for qv in qvec:
        unit_scores = unit_emb @ qv
        n_units = min(1000, unit_scores.size)
        top_units = np.argpartition(-unit_scores, n_units - 1)[:n_units]
        top_units = top_units[np.argsort(-unit_scores[top_units])]
        parent_scores = np.full(pidx.n_parents, -np.inf, dtype=np.float32)
        np.maximum.at(
            parent_scores,
            pidx.unit_to_parent[top_units],
            unit_scores[top_units],
        )
        top = rank_parents(parent_scores, pidx, top_k=5, only_positive=False)
        score_by_id = {
            pid: float(parent_scores[i])
            for i, pid in enumerate(pidx.parent_ids)
        }
        rows = []
        for rank, pid in enumerate(top, start=1):
            text = parent_text.get(pid, "")
            snippet = text[:500] + ("..." if len(text) > 500 else "")
            rows.append({
                "rank": rank,
                "passage_id": pid,
                "dense_score": score_by_id.get(pid, float("-inf")),
                "title": parent_titles.get(pid, ""),
                "snippet": snippet,
            })
        results.append(rows)
    return results


def run_full(
    sample: list[SampledPassage],
    records: list[dict],
    model: str,
    seed: int,
) -> dict:
    cache = SQLiteGenerationCache()
    drafter = GeminiDrafter(model, cache)
    verification = drafter.verify_one_token()
    norm_corpus = normalized_corpus(records)

    answerable_pool = sample[:ANSWERABLE_POOL_N]
    unanswerable_pool = sample[
        ANSWERABLE_POOL_N:ANSWERABLE_POOL_N + UNANSWERABLE_POOL_N
    ]
    answerable_plan = make_type_plan(ANSWERABLE_TYPE_PLAN, seed + 1)
    unanswerable_plan = make_type_plan(UNANSWERABLE_TYPE_PLAN, seed + 2)

    print(f"[draft] {N_ANSWERABLE} answerable from pool {len(answerable_pool)}")
    answerable_rows, stats_a, types_a, name_a = generate_answerable_rows(
        drafter, answerable_pool, answerable_plan, N_ANSWERABLE,
        norm_corpus, MAX_WHEN_ANSWERABLE, MAX_NAME_HEAVY_ANSWERABLE,
    )

    print(f"[draft] {N_UNANSWERABLE} unanswerable from pool {len(unanswerable_pool)}")
    unanswerable_rows, stats_u, types_u, name_u, unanswerable_meta = (
        generate_unanswerable_rows(
            drafter, unanswerable_pool, unanswerable_plan, N_UNANSWERABLE,
            MAX_WHEN_UNANSWERABLE, MAX_NAME_HEAVY_UNANSWERABLE,
            N_ANSWERABLE + 1,
        )
    )

    rows = answerable_rows + unanswerable_rows
    type_counts = defaultdict(int)
    for counts in (types_a, types_u):
        for qtype, n in counts.items():
            type_counts[qtype] += n
    missing_types = [t for t in QUESTION_TYPES if type_counts[t] == 0]
    date_share = type_counts["when"] / len(rows)
    name_heavy_total = name_a + name_u
    if missing_types:
        raise RuntimeError(f"missing question types: {missing_types}")
    if date_share > 0.30 + 1e-9:
        raise RuntimeError(f"date question share {date_share:.3f} exceeds 0.30")
    if name_heavy_total > MAX_NAME_HEAVY_ANSWERABLE + MAX_NAME_HEAVY_UNANSWERABLE:
        raise RuntimeError(f"name-heavy count {name_heavy_total} exceeds 10% cap")

    print(f"[retrieve] dense top-5 for {N_UNANSWERABLE} unanswerable questions")
    parent_ids = load_units_for_dense()
    parent_text = {str(r["passage_id"]): str(r["text"]) for r in records}
    parent_titles = {str(r["passage_id"]): str(r.get("title", "")) for r in records}
    top5 = dense_top5(
        [row["question"] for row in unanswerable_rows],
        parent_ids,
        parent_text,
        parent_titles,
    )

    retrieval_rows = []
    for row, meta, retrieved in zip(unanswerable_rows, unanswerable_meta, top5):
        premise_type = str(
            meta["result"].parsed.get("premise_type", meta["premise_type"])
        )
        for item in retrieved:
            retrieval_rows.append({
                "id": row["id"],
                "question": row["question"],
                "question_type": row["question_type"],
                "premise_type": premise_type,
                "rank": item["rank"],
                "passage_id": item["passage_id"],
                "dense_score": f"{item['dense_score']:.6f}",
                "title": item["title"],
                "snippet": item["snippet"],
            })

    write_csv(OUT_REVIEW, REVIEW_FIELDS, rows)
    write_csv(
        OUT_UNANSWERABLE_RETRIEVAL,
        UNANSWERABLE_RETRIEVAL_FIELDS,
        retrieval_rows,
    )

    stats = {
        "fresh_api_calls": stats_a["fresh_api_calls"] + stats_u["fresh_api_calls"],
        "cache_hits": stats_a["cache_hits"] + stats_u["cache_hits"],
        "rejected_attempts": (
            stats_a["rejected_attempts"] + stats_u["rejected_attempts"]
        ),
        "prompt_tokens": stats_a["prompt_tokens"] + stats_u["prompt_tokens"],
        "output_tokens": stats_a["output_tokens"] + stats_u["output_tokens"],
        "total_tokens": stats_a["total_tokens"] + stats_u["total_tokens"],
    }

    manifest = {
        "generated_at": utc_now(),
        "model": model,
        "verification": verification,
        "pricing_usd_per_1m_tokens": MODEL_PRICING[model],
        "seed": seed,
        "sampling": {
            "method": "uniform over eligible articles, one passage per article",
            "min_e5_tokens": MIN_TOKENS,
            "max_e5_tokens": MAX_TOKENS,
            "candidate_pool": len(sample),
            "answerable_pool": len(answerable_pool),
            "unanswerable_pool": len(unanswerable_pool),
            "selected_answerable": len(answerable_rows),
            "selected_unanswerable": len(unanswerable_rows),
        },
        "constraints": {
            "question_type_counts": dict(type_counts),
            "date_question_share": round(date_share, 4),
            "date_question_cap": 0.30,
            "name_heavy_questions": name_heavy_total,
            "name_heavy_cap": 0.10,
            "all_required_types_present": not missing_types,
        },
        "cache": {
            "path": str(CACHE_DB.relative_to(PROJECT_ROOT)),
            "rows": cache.count(),
        },
        "usage": {
            **stats,
            "estimated_cost_usd": round(
                usage_cost_usd(
                    stats["prompt_tokens"], stats["output_tokens"], model
                ),
                6,
            ),
        },
        "outputs": {
            "review_csv": str(OUT_REVIEW.relative_to(PROJECT_ROOT)),
            "unanswerable_retrieval_csv": str(
                OUT_UNANSWERABLE_RETRIEVAL.relative_to(PROJECT_ROOT)
            ),
        },
    }
    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[ok] wrote {OUT_REVIEW.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")
    print(
        f"[ok] wrote {OUT_UNANSWERABLE_RETRIEVAL.relative_to(PROJECT_ROOT)} "
        f"({len(retrieval_rows)} rows)"
    )
    print(f"[ok] types={dict(type_counts)} date_share={date_share:.3f} "
          f"name_heavy={name_heavy_total}")
    print("\nFIRST 5 REVIEW ROWS")
    for row in rows[:5]:
        print(json.dumps({
            "id": row["id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "passage_snippet": row["passage_snippet"][:220],
            "check_flags": row["check_flags"],
        }, ensure_ascii=False))
    return manifest

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["estimate", "run"], default="estimate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.model not in MODEL_PRICING:
        raise SystemExit(
            f"unknown model {args.model!r}; known: {sorted(MODEL_PRICING)}"
        )

    print(f"[load] corpus {CORPUS.relative_to(PROJECT_ROOT)}")
    records = load_corpus()
    print(f"[tokenize] e5 lengths for {len(records):,} parents")
    tokenizer = load_tokenizer()
    lengths = e5_token_lengths(records, tokenizer)
    sample = select_sample(records, lengths, seed=args.seed, n_total=N_SAMPLE)
    print(
        f"[sample] seed={args.seed} candidate_pool={len(sample)} "
        f"(target answerable={N_ANSWERABLE}, unanswerable={N_UNANSWERABLE})"
    )

    if args.mode == "estimate":
        run_estimate(sample, records, args.model)
    else:
        run_full(sample, records, args.model, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
