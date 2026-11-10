"""Deterministic text helpers for Batch 2 draft generation and audits."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BATCH2_SEED = 20261110
QUESTION_TYPES = ("who", "when", "where", "what", "how_many", "why_how")
PARTS = ("A", "B", "C")
E5_MODEL = "intfloat/multilingual-e5-small"
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_WS_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")
_MONTHS = (
    "জানুয়ারি|জানুয়ারি|ফেব্রুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|"
    "আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|january|february|march|"
    "april|may|june|july|august|september|october|november|december"
)
_FULL_DATE_NUMERIC = re.compile(
    r"\d{4}[^\d]{0,10}\d{1,2}[^\d]{0,5}\d{1,2}|\d{1,2}[^\d]{1,5}\d{1,2}[^\d]{1,5}\d{4}"
)
_FULL_DATE_NAMED = re.compile(
    rf"\d{{1,2}}[^\w]?(?:{_MONTHS})|(?:{_MONTHS})[^\w]?\d{{1,2}}",
    re.I,
)
_BN_NUMBERS = {
    "শূন্য", "এক", "একটি", "একজন", "দুই", "দুটি", "দুজন", "তিন", "তিনটি", "তিনজন",
    "চার", "চারটি", "পাঁচ", "পাঁচটি", "ছয়", "ছয়", "ছয়টি", "ছয়টি", "সাত", "সাতটি",
    "আট", "আটটি", "নয়", "নয়", "নয়টি", "নয়টি", "দশ", "এগারো", "বারো", "তেরো",
    "চৌদ্দ", "পনেরো", "ষোল", "সতেরো", "আঠারো", "উনিশ", "বিশ", "শত", "হাজার",
    "লক্ষ", "লাখ", "কোটি",
}
_SUFFIXES = sorted(
    {
        "গুলোর", "গুলিতে", "গুলোতে", "গুলোরও", "সমূহের", "সমূহকে", "গুলিকে", "গুলোকে",
        "দেরকে", "দেরের", "টিরও", "টির", "টিতে", "টিকে", "দের", "গুলি", "গুলো", "সমূহ",
        "েদের", "ের", "কে", "টি", "টা", "তে", "য়", "য়", "র", "ে", "ও",
    },
    key=len,
    reverse=True,
)
_STOPWORDS = set(
    """
    আমি আমরা তুমি তোমরা আপনি আপনার তিনি তারা তাহারা যে যা যিনি যাদের যাহা এ এই সেই তা
    তার তাঁর তাদের এরা ও এবং কিন্তু অথবা তবে যদি কারণ জন্য থেকে পর্যন্ত মধ্যে উপর নিচে
    কাছে পাশে সঙ্গে দ্বারা প্রতি কোন কোনটি কী কি কেন কিভাবে কীভাবে কোথায় কোথায় কবে কত
    কতজন কতটি কার কাদের হিসেবে অনুযায়ী অনুযায়ী সালে সালের তারিখ ছিল ছিলেন হয় হয় করা
    করেন করে করেছে হয়েছিল হয়েছিল হতে হওয়া হওয়া হওয়ার একটি এক জন টি টির এইসব উক্ত
    সম্পর্কিত বিষয় বিষয় তথ্য নাম পদ ব্যক্তি সংখ্যা অংশ প্রধান অন্যতম বিভিন্ন নিজের
    মাধ্যমে নিজস্ব প্রথম দ্বিতীয় দ্বিতীয় তৃতীয় তৃতীয় সব সকল কোনো কোনোও এখন বর্তমানে
    বর্তমান
    """.split()
)
_QUESTION_WORDS = set(
    "কোন কোনটি কী কি কেন কিভাবে কীভাবে কোথায় কোথায় কবে কত কতজন কতটি কার কাদের".split()
)


def nfc(text: str | None) -> str:
    return unicodedata.normalize("NFC", "" if text is None else str(text))


def normalize_digits(text: str | None) -> str:
    return nfc(text).translate(_DIGITS)


def normalize_space(text: str | None) -> str:
    return _WS_RE.sub(" ", nfc(text)).strip()


def word_tokens(text: str | None) -> list[str]:
    value = normalize_digits(text).casefold()
    tokens: list[str] = []
    buf: list[str] = []
    for ch in value:
        if unicodedata.category(ch)[0] in "LMN":
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def compact(text: str | None) -> str:
    return "".join(ch for ch in normalize_digits(text).casefold() if unicodedata.category(ch)[0] in "LMN")


def contains_text(haystack: str | None, needle: str | None) -> bool:
    needle_text = normalize_space(needle)
    if not needle_text:
        return False
    if needle_text in normalize_space(haystack):
        return True
    needle_digits = normalize_space(normalize_digits(needle))
    return bool(needle_digits) and needle_digits in normalize_space(normalize_digits(haystack))


def compact_contains(haystack: str | None, needle: str | None) -> bool:
    needle_text = compact(needle)
    if not needle_text:
        return False
    return needle_text in compact(haystack)


def stem_bn(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def content_terms(text: str | None) -> set[str]:
    terms: set[str] = set()
    for token in word_tokens(text):
        stem = stem_bn(token)
        if len(stem) < 2 or token in _STOPWORDS or stem in _STOPWORDS:
            continue
        if token in _QUESTION_WORDS or stem in _QUESTION_WORDS:
            continue
        if stem.isdigit():
            continue
        terms.add(stem)
    return terms


def question_terms(text: str | None) -> set[str]:
    return content_terms(text)


def max_shared_word_ngram(question: str | None, passage: str | None, n: int = 4) -> int:
    q = word_tokens(question)
    p = word_tokens(passage)
    if len(q) < n or len(p) < n:
        return 0
    qgrams = {tuple(q[i : i + n]) for i in range(len(q) - n + 1)}
    for i in range(len(p) - n + 1):
        if tuple(p[i : i + n]) in qgrams:
            return n
    return 0


def shared_word_ngram_hits(
    question: str | None, passage: str | None, n: int = 4
) -> set[tuple[str, ...]]:
    q = word_tokens(question)
    p = word_tokens(passage)
    if len(q) < n or len(p) < n:
        return set()
    qgrams = {tuple(q[i : i + n]) for i in range(len(q) - n + 1)}
    return {
        tuple(p[i : i + n]) for i in range(len(p) - n + 1) if tuple(p[i : i + n]) in qgrams
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_question_type(value: str | None) -> str:
    qtype = normalize_space(value).lower()
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"unsupported question_type: {value!r}")
    return qtype


def infer_question_type(question: str | None) -> str:
    q = " ".join(word_tokens(question))
    if re.search(
        r"(?:^| )(?:কত|কোন) সালে(?: |$)|(?:^| )কবে(?: |$)|(?:^| )(?:কত|কোন) (?:তারিখ|তারিখে|তারিখের)(?: |$)|(?:^| )(?:কোন|কত) সময়কাল(?:ে| থেকে)?(?: |$)",
        q,
    ):
        return "when"
    if re.search(
        r"(?:^| )(?:কতজন|কতজনের|কতটি|কতগুলো|কতগুলি|কয়টি|কয়টি|কতটা|কত|কয়জন|কয়জন)(?: |$)|(?:^| )কত ?সংখ্যক(?: |$)",
        q,
    ):
        return "how_many"
    if re.search(r"(?:^| )(?:কীভাবে|কিভাবে|কেন)(?: |$)|(?:^| )(?:কী|কোন) কারণে(?: |$)", q):
        return "why_how"
    if re.search(
        r"(?:^| )(?:কোথায়|কোথায়)(?: |$)|(?:^| )কোন(?: [^ ]+){0,3} (?:শহরে|দেশে|দেশের|জেলায়|জেলায়|উপজেলায়|উপজেলায়|অঞ্চলে|স্থানে|গ্রামে|বিশ্ববিদ্যালয়ে|বিশ্ববিদ্যালয়|প্রতিষ্ঠানে|এলাকায়|এলাকায়)(?: |$)",
        q,
    ):
        return "where"
    if re.search(
        r"(?:^| )(?:কে|কার|কারা|কাদের)(?: |$)|(?:^| )কোন(?: [^ ]+){0,2} (?:ব্যক্তি|ব্যক্তির|নির্মাতা|অভিনেতা|লেখক|পরিচালক|রাষ্ট্রপতি|প্রধানমন্ত্রী|অধিনায়ক|অধিনায়ক)(?: |$)",
        q,
    ):
        return "who"
    return "what"


def is_time_sensitive_undated(question: str | None) -> bool:
    q = " ".join(word_tokens(question))
    if _YEAR_RE.search(normalize_digits(question)):
        return False
    patterns = (
        r"বর্তমান", r"বর্তমানে", r"এখন", r"আজ", r"বর্তমানের", r"চলমান",
        r"সাম্প্রতিক", r"বর্তমান সভাপতি", r"বর্তমান প্রধানমন্ত্রী",
    )
    return any(re.search(p, q) for p in patterns)


def normalize_person_name(text: str | None) -> str:
    return compact(text)


def answer_satisfies_type(question_type: str, question: str | None, answer: str | None) -> bool:
    qtype = validate_question_type(question_type)
    ans = normalize_digits(answer)
    if not normalize_space(ans):
        return False
    if qtype == "when":
        year_q = bool(
            re.search(r"(?:^| )(?:কত|কোন) সালে(?: |$)|(?:^| )কবে(?: |$)", " ".join(word_tokens(question)))
        )
        if year_q and _YEAR_RE.search(ans):
            return True
        return bool(_FULL_DATE_NUMERIC.search(ans) or _FULL_DATE_NAMED.search(ans) or _YEAR_RE.search(ans))
    if qtype == "how_many":
        return bool(re.search(r"\d", ans) or any(tok in _BN_NUMBERS for tok in word_tokens(ans)))
    return True


def is_name_heavy(question: str | None) -> bool:
    q = normalize_space(question)
    if q.count("-") >= 3:
        return True
    tokens = [t for t in word_tokens(q) if t not in _QUESTION_WORDS and t not in _STOPWORDS]
    long_tokens = [t for t in tokens if len(t) >= 7 and not t.isdigit()]
    return len(long_tokens) >= 4


def question_rare_term_ok(question: str | None, passage: str | None, rare_terms: set[str]) -> bool:
    terms = question_terms(question)
    if not terms:
        return False
    passage_terms = content_terms(passage)
    overlap = {t for t in terms if t in rare_terms and t in passage_terms}
    return bool(overlap)


def compute_alt_gold_ids(
    answer: str | None,
    question: str | None,
    gold_passage_id: str,
    corpus: Iterable[tuple[str, str]],
    rare_terms: set[str],
    max_count: int = 10,
) -> tuple[list[str], int]:
    answer_text = normalize_space(answer)
    if not answer_text:
        return [], 0
    qterms = question_terms(question)
    matches: list[str] = []
    for pid, text in corpus:
        if pid == gold_passage_id:
            continue
        if not contains_text(text, answer_text):
            continue
        pterms = content_terms(text)
        if rare_terms and not any(t in rare_terms and t in pterms for t in qterms):
            continue
        if not rare_terms and not (qterms & pterms):
            continue
        matches.append(pid)
    deduped = list(dict.fromkeys(matches))
    return deduped[:max_count], len(deduped)


def build_rare_terms(corpus_texts: Iterable[str], df_ratio: float = 0.005) -> set[str]:
    texts = list(corpus_texts)
    if not texts:
        return set()
    df: Counter[str] = Counter()
    for text in texts:
        df.update(content_terms(text))
    threshold = max(1, int(len(texts) * df_ratio))
    return {term for term, count in df.items() if 0 < count <= threshold}


def join_flags(flags: Iterable[str]) -> str:
    seen: list[str] = []
    for flag in flags:
        value = normalize_space(flag)
        if value and value not in seen:
            seen.append(value)
    return ";".join(seen)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
