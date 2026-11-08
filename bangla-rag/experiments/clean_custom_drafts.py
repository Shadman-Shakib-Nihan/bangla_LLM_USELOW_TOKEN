"""Clean custom Bengali QA drafts without changing the source CSV."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/custom_drafts.csv"
CORPUS = ROOT / "data/processed/corpus.jsonl"
OUT_CLEAN = ROOT / "data/custom_drafts_v2.csv"
OUT_EXCLUDED = ROOT / "data/custom_drafts_excluded.csv"
OUT_UNANSWERABLE = ROOT / "data/custom_drafts_unanswerable.csv"
OUT_QTYPE_REVIEW = ROOT / "data/question_type_review.csv"
OUT_CHANGELOG = ROOT / "results/custom_drafts_changelog.md"

REQUIRED = [
    "id", "question", "question_type", "gold_answer", "passage_snippet",
    "gold_passage_id", "alt_gold_ids", "answerable", "language_pair",
    "answer_is_substring", "max_shared_word_4gram", "check_flags",
    "my_decision", "my_notes",
]
ADDED = ["answer_type_status", "alt_gold_over_cap", "alt_gold_total_count"]
EXCLUSION_FIELD = "exclusion_reasons"
QTYPE_REVIEW_FIELDS = ["id", "question", "assigned_question_type", "classification_reason", "review_status"]
DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
RARE_DF_RATIO = 0.005

STOPWORDS = set("""
আমি আমরা তুমি তোমরা আপনি আপনার তিনি তারা তাহারা যে যা যিনি যাদের যাহা
এ এই সেই তা তার তাঁর তাদের এরা ও এবং কিন্তু অথবা তবে যদি কারণ জন্য থেকে
পর্যন্ত মধ্যে উপর নিচে কাছে পাশে সঙ্গে দ্বারা প্রতি কোন কোনটি কী কি কেন
কিভাবে কীভাবে কোথায় কোথায় কবে কত কতজন কতটি কার কাদের হিসেবে অনুযায়ী
অনুযায়ী সালে সালের তারিখ সালে সালে থেকে মধ্যে ছিল ছিলেন হয় হয় করা করেন
করে করেছে হয়েছিল হয়েছিল হতে হওয়া হওয়া হওয়ার হওয়ার জন্যেই একটি এক জন
টি টির টিের এইসব উক্ত যথাক্রমে সম্পর্কিত বিষয় বিষয় তথ্য নাম পদ ব্যক্তি
সংখ্যা অংশ প্রধান অন্যতম বিভিন্ন নিজের মাধ্যমে নিজস্ব প্রথম দ্বিতীয়
দ্বিতীয় তৃতীয় তৃতীয় সব সকল কোনো কোনোও এখন বর্তমানে বর্তমান
""".split())
QUESTION_WORDS = set("কোন কোনটি কী কি কেন কিভাবে কীভাবে কোথায় কোথায় কবে কত কতজন কতটি কার কাদের".split())
BENGALI_NUMBERS = {
    "শূন্য", "এক", "একটি", "একজন", "দুই", "দুটি", "দুজন", "তিন", "তিনটি", "তিনজন",
    "চার", "চারটি", "পাঁচ", "পাঁচটি", "ছয়", "ছয়", "ছয়টি", "ছয়টি", "সাত", "সাতটি",
    "আট", "আটটি", "নয়", "নয়", "নয়টি", "নয়টি", "দশ", "দশটি", "এগারো", "বারো",
    "এগারো", "তেরো", "চৌদ্দ", "পনেরো", "ষোল", "সতেরো", "আঠারো", "উনিশ", "একুশ", "বাইশ",
    "বিশ", "পঁচিশ", "ত্রিশ", "চল্লিশ", "পঞ্চাশ", "শত", "শতাধিক", "হাজার", "লক্ষ", "লাখ", "কোটি",
}
SUFFIXES = sorted({
    "গুলোর", "গুলিতে", "গুলোতে", "গুলোরও", "সমূহের", "সমূহকে", "গুলিকে", "গুলোকে",
    "দেরকে", "দেরের", "টিরও", "টির", "টিতে", "টিকে", "দের", "গুলি", "গুলো", "সমূহ",
    "েদের", "ের", "দের", "কে", "টি", "টা", "তে", "য়", "য়", "র", "ে", "ও",
}, key=len, reverse=True)


def normalize_digits(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).translate(DIGITS)


def compact(text: str) -> str:
    """Casefold, map digits, and ignore punctuation/whitespace for matching."""
    value = normalize_digits(text).casefold()
    return "".join(ch for ch in value if unicodedata.category(ch)[0] in "LMN")


def word_tokens(text: str) -> list[str]:
    value = normalize_digits(text).casefold()
    result, buf = [], []
    for ch in value:
        if unicodedata.category(ch)[0] in "LMN":
            buf.append(ch)
        elif buf:
            result.append("".join(buf)); buf = []
    if buf:
        result.append("".join(buf))
    return result


def stem_bn(token: str) -> str:
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def question_terms(question: str) -> set[str]:
    terms = set()
    for token in word_tokens(question):
        stem = stem_bn(token)
        if len(stem) < 2 or stem in STOPWORDS or token in STOPWORDS or token in QUESTION_WORDS:
            continue
        if stem.isdigit():
            continue
        terms.add(stem)
    return terms


def answer_type(question: str, answer: str, answerable: bool) -> str:
    if not answerable:
        return "not_applicable"
    q = " ".join(word_tokens(question))
    a = normalize_digits(answer)
    year_q = bool(re.search(r"(?:^| )(?:কত|কোন) সালে(?: |$)|(?:^| )কবে(?: |$)|(?:^| )(?:কত|কোন) (?:তারিখ|তারিখে|তারিখের)(?: |$)", q))
    if year_q:
        if re.search(r"(?<!\d)\d{4}(?!\d)", a):
            month_names = (
                "জানুয়ারি|জানুয়ারি|ফেব্রুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|"
                "আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|january|february|march|"
                "april|may|june|july|august|september|october|november|december"
            )
            numeric_date = re.search(r"\d{4}[^\d]{0,10}\d{1,2}[^\d]{0,5}\d{1,2}|\d{1,2}[^\d]{1,5}\d{1,2}[^\d]{1,5}\d{4}", a)
            named_month_date = re.search(rf"\d{{1,2}}[^\w]?(?:{month_names})|(?:{month_names})[^\w]?\d{{1,2}}", a, re.I)
            if numeric_date or named_month_date:
                return "valid_full_date"
            return "valid_year"
        return "year_mismatch"
    quantity_q = bool(re.search(r"(?:^| )(?:কতজন|কতজনের|কতজনকে|কতটি|কতগুলো|কতগুলি|কয়টি|কয়টি|কতটা|কততম|কত|কয়জন|কয়জন)(?: |$)|(?:^| )কত ?সংখ্যক(?: |$)", q))
    if quantity_q:
        if re.search(r"\d", a) or any(w in BENGALI_NUMBERS for w in word_tokens(a)):
            return "valid_number"
        return "number_mismatch"
    return "not_applicable"


def deterministic_question_type(question: str) -> str:
    q = " ".join(word_tokens(question))
    if re.search(r"(?:^| )(?:কত|কোন) সালে(?: |$)|(?:^| )কবে(?: |$)|(?:^| )(?:কত|কোন) (?:তারিখ|তারিখে|তারিখের)(?: |$)|(?:^| )(?:কোন|কত) সময়কাল(?:ে| থেকে)?(?: |$)", q):
        return "when"
    if re.search(r"(?:^| )(?:কতজন|কতজনের|কতজনকে|কতটি|কতগুলো|কতগুলি|কয়টি|কয়টি|কতটা|কততম|কত|কয়জন|কয়জন)(?: |$)|(?:^| )কত ?সংখ্যক(?: |$)", q):
        return "how_many"
    if re.search(r"(?:^| )(?:কীভাবে|কিভাবে|কেন)(?: |$)|(?:^| )(?:কী|কোন) কারণে(?: |$)", q):
        return "why_how"
    if re.search(r"(?:^| )কোথায়(?: |$)|(?:^| )কোথায়(?: |$)|(?:^| )কোন(?: [^ ]+){0,3} (?:শহরে|শহরগুলোতে|দেশে|দেশের|দেশটিতে|জেলায়|জেলায়|উপজেলায়|উপজেলায়|অঞ্চলে|স্থানে|গ্রামে|নগরে|রাজ্যে|প্রদেশে|এলাকায়|এলাকাতে|প্রতিষ্ঠানে|প্রতিষ্ঠান থেকে|শিক্ষাপ্রতিষ্ঠান থেকে|বিশ্ববিদ্যালয়ে|বিশ্ববিদ্যালয়|অবস্থানে|দিকে|উচ্চতায়|উচ্চতায়|পথে)(?: |$)", q):
        return "where"
    if "ওয়েবসাইট" in q or "ওয়েবসাইট" in q:
        return "what"
    if re.search(r"(?:^| )(?:কে|কার|কারা|কাদের)(?: |$)|(?:^| )কোন(?: [^ ]+){0,2} (?:ব্যক্তি|ব্যক্তির|ব্যক্তিত্ব|নির্মাতা|নির্মাতার|অভিনেতা|অভিনেতার|ক্রিকেটার|ফুটবলার|আইনজীবী|আইনজীবি|লেখক|লেখকের|পরিচালক|পরিচালকের|পিতা|বাবা|রাষ্ট্রপতি|প্রধানমন্ত্রী|অধিনায়ক|অধিনায়ক|খেলোয়াড়|খেলোয়াড়)(?: |$)", q):
        return "who"
    return "what"


def classify_question_type_with_reason(question: str) -> tuple[str, str, str]:
    """Return deterministic label, reason, and an explicit uncertainty status."""
    q = " ".join(word_tokens(question))
    qtype = deterministic_question_type(question)
    cues = []
    if re.search(r"(?:^| )(?:কত|কোন) সালে(?: |$)|(?:^| )কবে(?: |$)|(?:^| )(?:কত|কোন) (?:তারিখ|তারিখে|তারিখের)(?: |$)|(?:^| )(?:কোন|কত) সময়কাল(?:ে| থেকে)?(?: |$)", q): cues.append("when")
    if re.search(r"(?:^| )(?:কতজন|কতজনের|কতজনকে|কতটি|কতগুলো|কতগুলি|কয়টি|কয়টি|কতটা|কততম|কত|কয়জন|কয়জন)(?: |$)|(?:^| )কত ?সংখ্যক(?: |$)", q): cues.append("how_many")
    if re.search(r"(?:^| )(?:কীভাবে|কিভাবে|কেন)(?: |$)|(?:^| )(?:কী|কোন) কারণে(?: |$)", q): cues.append("why_how")
    explicit_where = bool(re.search(r"(?:^| )(?:কোথায়|কোথায়)(?: |$)", q))
    if explicit_where or qtype == "where": cues.append("where")
    explicit_who = bool(re.search(r"(?:^| )(?:কে|কার|কারা|কাদের)(?: |$)", q))
    if explicit_who or qtype == "who": cues.append("who")
    if len(set(cues)) > 1:
        return qtype, "Conflicting interrogative cue families: " + ", ".join(sorted(set(cues))), "manual_review"
    if qtype == "when": return qtype, "Explicit year/date interrogative pattern", "rule_assigned"
    if qtype == "how_many": return qtype, "Explicit quantity interrogative pattern", "rule_assigned"
    if qtype == "why_how": return qtype, "Explicit why/how interrogative pattern", "rule_assigned"
    if qtype == "where":
        if explicit_where: return qtype, "Explicit where interrogative", "rule_assigned"
        return qtype, "Place inferred from a কোন + locative suffix pattern", "manual_review"
    if qtype == "who":
        if explicit_who: return qtype, "Explicit কে/কার/কাদের interrogative", "rule_assigned"
        return qtype, "Person inferred from a কোন + person-role pattern", "manual_review"
    if re.search(r"(?:^| )(?:কী|কি|কোন|কোনটি)(?: |$)", q):
        return qtype, "What/which construction; object interpretation may be ambiguous", "manual_review"
    return qtype, "No supported interrogative cue; assigned what by fallback", "manual_review"


def flag_tokens(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(";") if item.strip()}


def _cell(value: object) -> str:
    """Compare CSV values with empty/missing cells treated equivalently."""
    return "" if value is None else str(value)


def verify_output_accounting(source_rows: list[dict[str, str]], fields: list[str],
                             clean_rows: list[dict[str, str]],
                             unanswerable_rows: list[dict[str, str]],
                             excluded_rows: list[dict[str, str]]) -> dict:
    """Raise on loss, duplication, changed source fields, or incomplete reasons."""
    source_by_id = {r["id"]: r for r in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source contains duplicate IDs; stable-ID accounting is ambiguous")
    outputs = [clean_rows, unanswerable_rows, excluded_rows]
    ids_by_output = [[r["id"] for r in group] for group in outputs]
    for name, ids in zip(("clean", "unanswerable", "excluded"), ids_by_output):
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs within {name} output")
    all_out_ids = [rid for group in ids_by_output for rid in group]
    if len(all_out_ids) != len(set(all_out_ids)):
        raise ValueError("duplicate IDs across output datasets")
    if set(all_out_ids) != set(source_by_id):
        missing = sorted(set(source_by_id) - set(all_out_ids))
        extra = sorted(set(all_out_ids) - set(source_by_id))
        raise ValueError(f"record accounting mismatch; missing={missing}, extra={extra}")

    for row in clean_rows:
        original = source_by_id[row["id"]]
        if original["answerable"].lower() != "true":
            raise ValueError(f"non-answerable record placed in clean answerable output: {row['id']}")
        for field in fields:
            if field in {"question_type", "alt_gold_ids"}:
                continue
            if _cell(row.get(field)) != _cell(original.get(field)):
                raise ValueError(f"retained source value changed: {row['id']}:{field}")

    for row in unanswerable_rows:
        original = source_by_id[row["id"]]
        if original["answerable"].lower() != "false":
            raise ValueError(f"answerable record placed in unanswerable output: {row['id']}")
        for field in fields:
            if _cell(row.get(field)) != _cell(original.get(field)):
                raise ValueError(f"unanswerable source value changed: {row['id']}:{field}")

    for row in excluded_rows:
        original = source_by_id[row["id"]]
        if original["answerable"].lower() != "true":
            raise ValueError(f"non-answerable record placed in ordinary exclusions: {row['id']}")
        for field in fields:
            if _cell(row.get(field)) != _cell(original.get(field)):
                raise ValueError(f"excluded source value changed: {row['id']}:{field}")
        expected = exclude_reasons(original)
        actual = [r for r in _cell(row.get(EXCLUSION_FIELD)).split(";") if r]
        if actual != expected:
            raise ValueError(f"exclusion reason mismatch for {row['id']}: expected={expected}, actual={actual}")

    return {
        "source": len(source_rows), "answerable": len(clean_rows),
        "unanswerable": len(unanswerable_rows), "excluded": len(excluded_rows),
        "accounted": len(all_out_ids), "unique_ids": len(set(all_out_ids)),
        "verified": True,
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv.field_size_limit(50_000_000)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    missing = [name for name in REQUIRED if name not in fields]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return fields, rows


def inspect(rows: list[dict[str, str]], fields: list[str]) -> dict:
    details = {}
    for col in fields:
        values = [r.get(col, "") for r in rows]
        nonempty = [v for v in values if v not in ("", None)]
        if nonempty and set(nonempty) <= {"true", "false"}:
            dtype = "boolean-like string"
        elif nonempty and all(re.fullmatch(r"-?\d+", v) for v in nonempty):
            dtype = "integer-like string"
        else:
            dtype = "string"
        details[col] = {
            "dtype": dtype,
            "missing": sum(v in ("", None) for v in values),
            "samples": [v[:100] for v in nonempty[:3]],
        }
    return details


def exclude_reasons(row: dict[str, str]) -> list[str]:
    reasons = []
    if row.get("answer_is_substring", "").strip().lower() == "false":
        reasons.append("answer_is_substring_false")
    normalized_answer = compact(row.get("gold_answer", ""))
    if len(normalized_answer) < 2:
        reasons.append("normalized_answer_under_2_characters")
    try:
        shared = int(row.get("max_shared_word_4gram", ""))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"cannot evaluate shared 4-gram exclusion for {row.get('id')}: {row.get('max_shared_word_4gram')!r}") from exc
    if shared >= 5:
        reasons.append("shared_4gram_at_least_5")
    if "name_heavy" in flag_tokens(row.get("check_flags", "")):
        reasons.append("name_heavy")
    if normalized_answer and normalized_answer in compact(row.get("question", "")):
        reasons.append("answer_appears_in_question")
    return reasons


def load_corpus(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"passage corpus missing: {path}")
    docs = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            rec = json.loads(line)
            pid, text = rec.get("passage_id"), rec.get("text")
            if pid is None or text is None:
                raise ValueError(f"corpus record {n} missing passage_id or text")
            docs.append((str(pid), str(text)))
    return docs


def recompute_alt_gold(rows: list[dict[str, str]], docs: list[tuple[str, str]]) -> dict[str, list[str]]:
    n_docs = len(docs)
    df: Counter[str] = Counter()
    for _, text in docs:
        terms = {stem_bn(t) for t in word_tokens(text) if len(stem_bn(t)) >= 2}
        df.update(terms)
    rare_limit = max(5, math.floor(n_docs * RARE_DF_RATIO))
    row_terms = {
        row["id"]: {term for term in question_terms(row["question"]) if df.get(term, 0) <= rare_limit}
        for row in rows
    }
    term_rows: dict[str, set[str]] = defaultdict(set)
    for rid, terms in row_terms.items():
        for term in terms:
            term_rows[term].add(rid)
    answers = {row["id"]: compact(row.get("gold_answer", "")) for row in rows}
    gold_ids = {row["id"]: row.get("gold_passage_id", "") for row in rows}
    hits: dict[str, set[str]] = {row["id"]: set() for row in rows}
    for pid, text in docs:
        text_norm = compact(text)
        terms = {stem_bn(t) for t in word_tokens(text) if len(stem_bn(t)) >= 2}
        candidates: set[str] = set()
        for term in terms:
            candidates.update(term_rows.get(term, ()))
        for rid in candidates:
            answer = answers[rid]
            if not answer or pid == gold_ids[rid]:
                continue
            if answer in text_norm:
                hits[rid].add(pid)
    return {rid: sorted(ids) for rid, ids in hits.items()}


def process(source: Path = SOURCE, corpus_path: Path = CORPUS,
            clean_path: Path = OUT_CLEAN, excluded_path: Path = OUT_EXCLUDED,
            changelog_path: Path = OUT_CHANGELOG,
            unanswerable_path: Path | None = None,
            qtype_review_path: Path | None = None) -> dict:
    unanswerable_path = unanswerable_path or clean_path.parent / OUT_UNANSWERABLE.name
    qtype_review_path = qtype_review_path or clean_path.parent / OUT_QTYPE_REVIEW.name
    fields, rows = read_csv(source)
    missing_required = [f for f in REQUIRED if f not in fields]
    if missing_required:
        raise ValueError(f"missing required columns: {missing_required}")
    data_inspection = inspect(rows, fields)
    duplicate_ids = [k for k, v in Counter(r["id"] for r in rows).items() if v > 1]
    flags_observed = Counter(flag for row in rows for flag in flag_tokens(row.get("check_flags", "")))
    # A name_heavy marker is evaluated only as the exact semicolon-delimited token.
    known_name_flag = "name_heavy" in flags_observed

    excluded, retained, unanswerable = [], [], []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        answerable_value = row.get("answerable", "").strip().lower()
        if answerable_value == "false":
            unanswerable.append(dict(row))
            continue
        if answerable_value != "true":
            raise ValueError(f"cannot route record with unclear answerable value: {row.get('id')}={answerable_value!r}")
        reasons = exclude_reasons(row)
        if reasons:
            out = dict(row)
            out[EXCLUSION_FIELD] = ";".join(reasons)
            excluded.append(out)
            reason_counts.update(reasons)
        else:
            retained.append(dict(row))

    corpus = load_corpus(corpus_path)
    alt = recompute_alt_gold(retained, corpus)
    full_date_count = answer_mismatch_count = 0
    alt_changed = overflow_count = 0
    qtype_changed = 0
    for row in retained:
        old_alt = row.get("alt_gold_ids", "")
        all_ids = alt[row["id"]]
        if old_alt != ";".join(all_ids):
            alt_changed += 1
        row["alt_gold_ids"] = ";".join(all_ids[:10])
        row["alt_gold_over_cap"] = "true" if len(all_ids) > 10 else "false"
        row["alt_gold_total_count"] = str(len(all_ids))
        row["answer_type_status"] = answer_type(
            row["question"], row.get("gold_answer", ""),
            row.get("answerable", "").lower() == "true",
        )
        if row["answer_type_status"] == "valid_full_date":
            full_date_count += 1
        if row["answer_type_status"] in {"year_mismatch", "number_mismatch"}:
            answer_mismatch_count += 1
        new_type = deterministic_question_type(row["question"])
        if new_type != row["question_type"]:
            qtype_changed += 1
        row["question_type"] = new_type

    clean_fields = fields + [f for f in ADDED if f not in fields]
    excluded_fields = fields + [EXCLUSION_FIELD]
    for path, fieldnames, output_rows in [
        (clean_path, clean_fields, retained),
        (excluded_path, excluded_fields, excluded),
        (unanswerable_path, fields, unanswerable),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader(); writer.writerows(output_rows)

    qtype_review_rows = []
    for original in rows:
        assigned, reason, review_status = classify_question_type_with_reason(original["question"])
        qtype_review_rows.append({
            "id": original["id"], "question": original["question"],
            "assigned_question_type": assigned,
            "classification_reason": reason,
            "review_status": review_status,
        })
    qtype_review_path.parent.mkdir(parents=True, exist_ok=True)
    with qtype_review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QTYPE_REVIEW_FIELDS)
        writer.writeheader(); writer.writerows(qtype_review_rows)

    _, clean_serialized = read_csv(clean_path)
    _, excluded_serialized = read_csv(excluded_path)
    _, unanswerable_serialized = read_csv(unanswerable_path)
    accounting = verify_output_accounting(
        rows, fields, clean_serialized, unanswerable_serialized, excluded_serialized
    )
    if len(qtype_review_rows) != len(rows) or len({r["id"] for r in qtype_review_rows}) != len(rows):
        raise ValueError("question_type_review.csv does not have one unique row per source record")

    mismatched_types = [r for r in retained if r["answer_type_status"] in {"year_mismatch", "number_mismatch"}]
    review_status_counts = Counter(r["review_status"] for r in qtype_review_rows)
    review_type_counts = Counter(r["assigned_question_type"] for r in qtype_review_rows)
    # Keep per-row counts from the complete qualifying IDs, not just the capped output.
    over_cap_ids = [rid for rid, ids in alt.items() if len(ids) > 10]
    schema_lines = ["| Column | Inferred CSV value type | Missing | Sample values |", "|---|---|---:|---|"]
    for col, detail in data_inspection.items():
        samples = "<br>".join(s.replace("|", "\\|") for s in detail["samples"])
        schema_lines.append(f"| `{col}` | {detail['dtype']} | {detail['missing']} | {samples} |")
    changelog = f"""# Custom Bengali QA drafts v2 processing log

## Input inspection

- Input: `{source.name}`; UTF-8 with optional BOM accepted.
- Original rows: **{len(rows)}**; original columns: **{len(fields)}**.
- Duplicate IDs: **{len(duplicate_ids)}**{(': ' + ', '.join(duplicate_ids)) if duplicate_ids else ''}.
- Mappings: `gold_answer` is the answer; `passage_snippet` is the passage snippet; `max_shared_word_4gram` is the shared 4-gram count. `answerable` and `answer_is_substring` were read and preserved separately.
- All source columns are retained. Only `question_type` and `alt_gold_ids` are recomputed for retained answerable rows; `my_decision` and `my_notes` are unchanged.
- `check_flags` is parsed as semicolon-delimited exact tokens. The observed `name_heavy` flag was {'present' if known_name_flag else 'not present'} ({flags_observed.get('name_heavy', 0)} row(s)).

### Column types, missing values, and sample values

{chr(10).join(schema_lines)}

## Exclusions

- Answerable retained in `{clean_path.name}`: **{len(retained)}**.
- Unanswerable preserved in `{unanswerable_path.name}`: **{len(unanswerable)}**.
- Genuinely excluded answerable records in `{excluded_path.name}`: **{len(excluded)}**.
- Counts by applicable reason (a row may count under more than one reason):
{chr(10).join(f'  - `{reason}`: {count}' for reason, count in sorted(reason_counts.items())) or '  - None'}
- Excluded rows and all applicable reasons are in `{excluded_path.name}`. Original values remain present; `exclusion_reasons` was added.
- Unanswerable rows were routed before short-answer validation; their 14 original columns and values were copied unchanged.

## Answer-type validation

- Year/date questions with valid four-digit years: {sum(r['answer_type_status'] == 'valid_year' for r in retained)}.
- Full-date answers: **{full_date_count}** (`valid_full_date`).
- Quantity questions with recognizable numeric answers: {sum(r['answer_type_status'] == 'valid_number' for r in retained)}.
- Type mismatches: **{answer_mismatch_count}** ({sum(r['answer_type_status'] == 'year_mismatch' for r in retained)} year/date; {sum(r['answer_type_status'] == 'number_mismatch' for r in retained)} quantity). They remain in the output for review.
- Non-date/non-quantity and unanswerable rows use `not_applicable`.
- Bengali and English digits are normalized to ASCII for validation. Common Bengali number words are accepted from a fixed documented list in the processing script.

## Alternative gold passages

- Corpus: `{corpus_path.name}`, {len(corpus)} passage records; available and read successfully.
- Qualifying passage requires both normalized gold-answer presence and at least one rare question content term. A rare term has document frequency at most {max(5, math.floor(len(corpus) * RARE_DF_RATIO))} ({RARE_DF_RATIO:.1%} of corpus size, with a minimum threshold of 5). Question words, stopwords, numeric-only tokens, and short terms are excluded. Common Bengali suffixes are stripped for term matching.
- Existing `alt_gold_ids` changed for **{alt_changed}** retained rows. Results are deduplicated, sorted, and capped at 10 IDs.
- Rows with more than 10 qualifying passages: **{len(over_cap_ids)}**; `alt_gold_total_count` retains the full number and `alt_gold_over_cap` marks overflow.
- Limitation: suffix handling and the fixed Bengali stopword/number-word lists are heuristic; rare/entity matching may miss inflected terms or may include uncommon non-entity content words.

## Question type

- Recomputed deterministically for retained answerable examples using question-form patterns: when/date, quantity, why/how, place, who, then what as the fallback.
- Labels changed among retained rows: **{qtype_changed}**. `{OUT_QTYPE_REVIEW.name}` records every source question, the assigned label, reason, and review status.
- Review statuses across all source questions: {dict(review_status_counts)}; assigned type counts: {dict(review_type_counts)}.
- The rules are not fully reliable. Ambiguous/uncertain cases and fallback assignments are marked `manual_review` for human checking.

## Record accounting

- Serialized outputs: {len(retained)} answerable + {len(unanswerable)} unanswerable + {len(excluded)} genuinely excluded = {accounting['accounted']} records.
- Stable-ID accounting verified: {accounting['unique_ids']} unique output IDs matched {accounting['source']} unique source IDs, with no missing, extra, or duplicate records.
- Original values in unanswerable and excluded rows were compared to source. Retained values were compared except explicitly recomputed `question_type` and `alt_gold_ids`; `my_decision` and `my_notes` were checked unchanged.

## Preservation and outputs

- `my_decision` and `my_notes` were copied unchanged for every retained record.
- `answerable` and `answer_is_substring` remain separate original columns and were not conflated.
- Source file was read only. Outputs: `{clean_path.name}`, `{unanswerable_path.name}`, `{excluded_path.name}`, and `{OUT_QTYPE_REVIEW.name}`.
- No answers, passage IDs, or evidence annotations were invented.
"""
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    changelog_path.write_text(changelog, encoding="utf-8")
    return {
        "rows": len(rows), "retained": len(retained), "unanswerable": len(unanswerable), "excluded": len(excluded),
        "reason_counts": dict(reason_counts), "type_mismatches": answer_mismatch_count,
        "full_dates": full_date_count, "alt_overflow_rows": len(over_cap_ids),
        "alt_changed": alt_changed, "question_type_changed": qtype_changed,
        "name_heavy_flags": flags_observed.get("name_heavy", 0),
        "duplicate_ids": duplicate_ids, "inspection": data_inspection,
        "accounting": accounting, "qtype_review_status_counts": dict(review_status_counts),
        "clean_path": str(clean_path), "unanswerable_path": str(unanswerable_path),
        "qtype_review_path": str(qtype_review_path), "excluded_path": str(excluded_path),
        "changelog_path": str(changelog_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--clean", type=Path, default=OUT_CLEAN)
    parser.add_argument("--excluded", type=Path, default=OUT_EXCLUDED)
    parser.add_argument("--changelog", type=Path, default=OUT_CHANGELOG)
    parser.add_argument("--unanswerable", type=Path, default=OUT_UNANSWERABLE)
    parser.add_argument("--question-type-review", type=Path, default=OUT_QTYPE_REVIEW)
    args = parser.parse_args()
    print(json.dumps(process(args.source, args.corpus, args.clean, args.excluded, args.changelog, args.unanswerable, args.question_type_review), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
