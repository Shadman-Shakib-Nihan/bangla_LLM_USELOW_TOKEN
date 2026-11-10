"""Approved full Batch 2 generation with checkpoint/resume and review package.

Safety: protected checksums verified before/after; seed 20261110; temperature 0;
thinkingBudget 0; max 3 attempts/target; Part A passage replacement and no reuse;
Part B likely-answerable/manual-review items queued (not auto-excluded); Part C
deterministic with no Gemini calls; live-call budget capped at $0.10 (stop+ask if
projected to exceed). Accepted rows keep needs_human_check=true and empty decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.batch2_gemini import (  # noqa: E402
    ANSWER_SCHEMA,
    BATCH2_SEED,
    UNANSWERABLE_SCHEMA,
    GeminiCallSpec,
    GeminiClient,
    classify_gemini_error,
    compute_backoff,
    parse_json_text,
)
from src.batch2_generation import (  # noqa: E402
    ExclusionTracker,
    build_part_c_removed_evidence,
    generate_part_a,
    generate_part_b,
)
from src.batch2_pipeline import (  # noqa: E402
    PART_A_FIELDS,
    PART_B_FIELDS,
    PART_C_FIELDS,
    CorpusSearch,
    bm25_search,
    dense_search,
    load_exclusion_inventory,
    load_passages,
    read_jsonl,
    sample_tydi_questions_for_part_c,
    verify_protected_checksums,
    write_csv,
    write_part_csvs,
)
from src.batch2_reports import (  # noqa: E402
    mandatory_review_rows,
    random_review_sample,
    review_priority_rows,
    write_qc_artifacts,
)
from src.batch2_text import (  # noqa: E402
    E5_MODEL,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    build_rare_terms,
    canonical_json,
    compact_contains,
    normalize_space,
    sha256_text,
)
from src.retrieval import BM25Index, build_parent_index  # noqa: E402

MODEL = "gemini-3.1-flash-lite"
THINKING_BUDGET = 0
TEMPERATURE = 0.0
SEED = BATCH2_SEED
MAX_ATTEMPTS = 3
INPUT_PER_M = 0.25
OUTPUT_PER_M = 1.50
BUDGET_CAP_USD = 0.10
TARGET_A, TARGET_B, TARGET_C = 200, 80, 150
OUT = PROJECT_ROOT / "data/custom_batch2/batch2_full_approved"


class QuotaExhausted(RuntimeError):
    """Raised to stop the run cleanly (checkpoint preserved) on daily quota exhaustion."""

    def __init__(self, info: dict) -> None:
        self.info = info
        super().__init__(f"QUOTA_EXHAUSTED during part {info.get('part')}: {info.get('error')}")


def _retry_after(exc: BaseException) -> float | None:
    return getattr(exc, "retry_after_seconds", None)


def _backoff_seconds(attempt: int, exc: BaseException) -> float:
    return compute_backoff(attempt, _retry_after(exc), base=2.0, factor=2.0, cap=60.0)


class BudgetGuard:
    def __init__(self, cap: float = BUDGET_CAP_USD) -> None:
        self.cap = cap
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.calls = 0
        self.cached_calls = 0
        self.failed_calls = 0
        self.events: list[dict] = []

    @property
    def cost(self) -> float:
        return self.input_tokens / 1e6 * INPUT_PER_M + self.output_tokens / 1e6 * OUTPUT_PER_M

    def note(self, usage: dict, cached: bool, label: str) -> None:
        if cached:
            self.cached_calls += 1
            return
        self.calls += 1
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        self.thinking_tokens += int(usage.get("thinking_tokens") or 0)
        self.events.append(
            {
                "label": label,
                "input": usage.get("input_tokens"),
                "output": usage.get("output_tokens"),
                "thinking": usage.get("thinking_tokens"),
                "cumulative_cost_est": round(self.cost, 6),
            }
        )

    def check_before(self) -> None:
        projected = (self.input_tokens + 1200) / 1e6 * INPUT_PER_M + (self.output_tokens + 200) / 1e6 * OUTPUT_PER_M
        if projected > self.cap:
            raise RuntimeError(
                f"BUDGET_CAP_RISK: projected ${projected:.6f} > cap ${self.cap:.2f}. "
                "Stopping before the next live call; ask for approval to raise the cap."
            )

    def summary(self) -> dict:
        return {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "thinking_budget": THINKING_BUDGET,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "failed_calls": self.failed_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cost_usd_est": round(self.cost, 6),
            "cap_usd": self.cap,
            "pricing_basis": {"input_per_m_usd": INPUT_PER_M, "output_per_m_usd": OUTPUT_PER_M},
        }


class _HardTimeout:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._old = None

    def _handle(self, signum, frame):
        raise TimeoutError(f"live Gemini call exceeded hard timeout of {self.seconds}s")

    def __enter__(self):
        self._old = signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        if self._old is not None:
            signal.signal(signal.SIGALRM, self._old)
        return False


class GeminiProvider:
    model_id = MODEL

    def __init__(self, client: GeminiClient, schema: dict, budget: BudgetGuard, label: str, state_dir: Path) -> None:
        self.client = client
        self.schema = schema
        self.budget = budget
        self.label = label
        self.state_dir = state_dir

    def generate_json(self, prompt: str, schema: dict | None = None, system: str | None = None):
        spec = GeminiCallSpec(
            model=MODEL,
            prompt=prompt,
            temperature=TEMPERATURE,
            seed=SEED,
            max_output_tokens=512,
            response_schema=schema or self.schema,
            thinking_budget=THINKING_BUDGET,
            system_instruction=system,
        )
        if not self.client.cache.get(spec.cache_key()):
            self.budget.check_before()
        with _HardTimeout(45):
            result = self.client.generate(spec, use_cache=True)
        self.budget.note(result.usage.__dict__, result.cached, self.label)
        (self.state_dir / "budget.json").write_text(canonical_json(self.budget.summary()), encoding="utf-8")
        try:
            return parse_json_text(result.text)
        except Exception:
            return {"_parse_error": True, "_raw": result.text[:300]}


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [{k: r.get(k, "") for k in fields} for r in csv.DictReader(fh)]


def _csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = json.dumps(r, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def load_eligible(passages, exclusions, cache_path: Path):
    if cache_path.exists():
        rows = read_jsonl(cache_path)
        by_id = {p.passage_id: p for p in passages}
        return [by_id[r["passage_id"]] for r in rows if r["passage_id"] in by_id]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL)
    base = [
        p
        for p in passages
        if p.source == "miracl"
        and not p.is_short
        and p.title
        and p.text
        and p.passage_id not in exclusions["passage_ids"]
        and p.parent_id not in exclusions["passage_ids"]
    ]
    eligible, rows = [], []
    batch_size = 256
    for i in range(0, len(base), batch_size):
        batch = base[i : i + batch_size]
        lengths = tokenizer(
            [E5_PASSAGE_PREFIX + p.text for p in batch],
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )["length"]
        for p, n in zip(batch, lengths):
            if 40 <= int(n) <= 300:
                eligible.append(p)
                rows.append({"passage_id": p.passage_id, "e5_length": int(n)})
        if i and i % (batch_size * 20) == 0:
            print(f"e5 scan {i}/{len(base)} eligible={len(eligible)}", flush=True)
    rows.sort(key=lambda r: sha256_text(f"{SEED}:{r['passage_id']}"))
    eligible.sort(key=lambda r: sha256_text(f"{SEED}:{r.passage_id}"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return eligible


def run_part_a(eligible, provider, corpus_pairs, rare_terms, bm25_audit, dense_audit, alt_lookup):
    csv_path = OUT / "batch2_answerable_review.csv"
    ex_path = OUT / "excluded_part_a_candidates.jsonl"
    part_a = read_csv(csv_path, PART_A_FIELDS)
    exclusion = ExclusionTracker()
    exclusion.rows.extend(load_rows(ex_path))
    attempts: list[dict] = load_rows(OUT / "audits" / "attempt_log.jsonl")
    used = {r["gold_passage_id"] for r in part_a}
    answers = {r["gold_answer"] for r in part_a}
    for rec in eligible:
        if len(part_a) >= TARGET_A:
            break
        if rec.passage_id in used:
            continue
        accepted = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if len(part_a) >= TARGET_A:
                break
            try:
                rows, ex = generate_part_a(
                    [rec],
                    provider=provider,
                    corpus=corpus_pairs,
                    rare_terms=rare_terms,
                    batch_id="batch2full",
                    seed=SEED,
                    bm25_audit=bm25_audit,
                    dense_audit=dense_audit,
                    max_items=1,
                    alt_gold_lookup=alt_lookup,
                )
            except Exception as exc:
                kind = classify_gemini_error(exc)
                attempts.append(
                    {
                        "part": "A",
                        "passage_id": rec.passage_id,
                        "attempt": attempt,
                        "error": repr(exc),
                        "error_kind": kind,
                    }
                )
                write_jsonl(OUT / "audits" / "attempt_log.jsonl", dedupe_rows(attempts))
                if kind == "daily_quota_exhausted":
                    raise QuotaExhausted(
                        {
                            "part": "A",
                            "passage_id": rec.passage_id,
                            "attempt": attempt,
                            "error": repr(exc),
                            "retry_after_seconds": _retry_after(exc),
                        }
                    ) from exc
                time.sleep(_backoff_seconds(attempt, exc))
                continue
            attempts.append(
                {
                    "part": "A",
                    "passage_id": rec.passage_id,
                    "attempt": attempt,
                    "accepted": len(rows),
                    "excluded": len(ex.rows),
                    "reasons": [r["reason"] for r in ex.rows],
                    "flags": [r.get("details", "") for r in ex.rows],
                }
            )
            if rows:
                row = dict(rows[0])
                if row["gold_answer"] in answers:
                    exclusion.add(
                        part="A",
                        item_id=f"A_{rec.passage_id}_a{attempt}",
                        reason="near_duplicate_answer_global",
                        stage="attempt",
                        details=row["gold_answer"][:80],
                    )
                else:
                    row["needs_human_check"] = True
                    row["my_decision"] = ""
                    row["my_notes"] = ""
                    part_a.append(row)
                    answers.add(row["gold_answer"])
                    used.add(rec.passage_id)
                    write_csv(csv_path, PART_A_FIELDS, part_a)
                    (OUT / "checkpoint.json").write_text(
                        canonical_json(
                            {"part": "A", "accepted": len(part_a), "last_passage": rec.passage_id, "ts": time.time()}
                        ),
                        encoding="utf-8",
                    )
                    write_jsonl(OUT / "audits" / "attempt_log.jsonl", dedupe_rows(attempts))
                    if len(part_a) % 10 == 0:
                        print(f"A accepted={len(part_a)} cost=${provider.budget.cost:.5f}", flush=True)
                    accepted = True
                    break
            for er in ex.rows:
                exclusion.add(
                    part="A",
                    item_id=f"A_{rec.passage_id}_a{attempt}",
                    reason=er["reason"],
                    stage=er["stage"],
                    details=er.get("details", ""),
                )
            # deterministic prompt + temperature 0: a validation failure will repeat
            # identically, so do not waste attempts; replace the passage instead.
            if ex.rows and all(er["stage"] != "parse" for er in ex.rows):
                break
        if not accepted:
            exclusion.add(
                part="A",
                item_id=f"A_{rec.passage_id}",
                reason="max_attempts_exhausted_replace_passage",
                stage="attempt",
                details=rec.passage_id,
            )
        write_jsonl(ex_path, dedupe_rows(exclusion.rows))
        write_jsonl(OUT / "audits" / "attempt_log.jsonl", dedupe_rows(attempts))
    if len(part_a) < TARGET_A:
        raise RuntimeError(f"PART_A_SHORTFALL accepted={len(part_a)} target={TARGET_A}; no fabrication/no silent reduction")
    return part_a, exclusion


def run_part_b(eligible, provider, passage_text, bm25_audit, dense_audit, used_passages):
    csv_path = OUT / "batch2_unanswerable_review.csv"
    ex_path = OUT / "excluded_part_b_candidates.jsonl"
    audit_path = OUT / "audits" / "part_b_audit.json"
    part_b = read_csv(csv_path, PART_B_FIELDS)
    exclusion = ExclusionTracker()
    exclusion.rows.extend(load_rows(ex_path))
    attempts: list[dict] = load_rows(OUT / "audits" / "attempt_log.jsonl")
    audits = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else []
    used_b = {r.get("source_passage_hint", "") for r in part_b}
    for rec in eligible:
        if len(part_b) >= TARGET_B:
            break
        if rec.passage_id in used_passages or rec.passage_id in used_b:
            continue
        accepted = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if len(part_b) >= TARGET_B:
                break
            try:
                rows, ex, aud = generate_part_b(
                    [rec],
                    provider=provider,
                    passage_text=passage_text,
                    bm25_audit=bm25_audit,
                    dense_audit=dense_audit,
                    batch_id="batch2full",
                    seed=SEED,
                    top_k=20,
                    max_items=1,
                    queue_likely_answerable=True,
                    queue_manual_evidence_review=True,
                )
            except Exception as exc:
                kind = classify_gemini_error(exc)
                attempts.append(
                    {
                        "part": "B",
                        "passage_id": rec.passage_id,
                        "attempt": attempt,
                        "error": repr(exc),
                        "error_kind": kind,
                    }
                )
                write_jsonl(OUT / "audits" / "attempt_log.jsonl", dedupe_rows(attempts))
                if kind == "daily_quota_exhausted":
                    raise QuotaExhausted(
                        {
                            "part": "B",
                            "passage_id": rec.passage_id,
                            "attempt": attempt,
                            "error": repr(exc),
                            "retry_after_seconds": _retry_after(exc),
                        }
                    ) from exc
                time.sleep(_backoff_seconds(attempt, exc))
                continue
            attempts.append(
                {
                    "part": "B",
                    "passage_id": rec.passage_id,
                    "attempt": attempt,
                    "accepted": len(rows),
                    "excluded": len(ex.rows),
                    "likely": [a.get("likely_answerable") for a in aud],
                    "manual": [a.get("manual_evidence_review") for a in aud],
                }
            )
            audits.extend(aud)
            if rows:
                row = dict(rows[0])
                row["needs_human_check"] = True
                row["my_decision"] = ""
                row["my_notes"] = ""
                part_b.append(row)
                used_b.add(rec.passage_id)
                write_csv(csv_path, PART_B_FIELDS, part_b)
                audit_path.write_text(canonical_json(audits), encoding="utf-8")
                (OUT / "checkpoint.json").write_text(
                    canonical_json(
                        {"part": "B", "accepted": len(part_b), "last_passage": rec.passage_id, "ts": time.time()}
                    ),
                    encoding="utf-8",
                )
                write_jsonl(OUT / "audits" / "attempt_log.jsonl", dedupe_rows(attempts))
                if len(part_b) % 10 == 0:
                    print(f"B accepted={len(part_b)} cost=${provider.budget.cost:.5f}", flush=True)
                accepted = True
                break
            for er in ex.rows:
                exclusion.add(
                    part="B",
                    item_id=f"B_{rec.passage_id}_a{attempt}",
                    reason=er["reason"],
                    stage=er["stage"],
                    details=er.get("details", ""),
                )
            if ex.rows and all(er["stage"] != "parse" for er in ex.rows):
                break
        if not accepted:
            exclusion.add(
                part="B",
                item_id=f"B_{rec.passage_id}",
                reason="max_attempts_exhausted",
                stage="attempt",
                details=rec.passage_id,
            )
        write_jsonl(ex_path, dedupe_rows(exclusion.rows))
        write_jsonl(OUT / "audits" / "attempt_log.jsonl", dedupe_rows(attempts))
        audit_path.write_text(canonical_json(audits), encoding="utf-8")
    if len(part_b) < TARGET_B:
        raise RuntimeError(f"PART_B_SHORTFALL accepted={len(part_b)} target={TARGET_B}; no fabrication/no silent reduction")
    return part_b, exclusion, audits


def run_part_c(exclusions, passage_text, bm25_audit, dense_audit):
    csv_path = OUT / "batch2_removed_evidence_review.csv"
    ex_path = OUT / "excluded_part_c_candidates.jsonl"
    audit_path = OUT / "audits" / "part_c_audit.json"
    part_c = read_csv(csv_path, PART_C_FIELDS)
    exclusion = ExclusionTracker()
    exclusion.rows.extend(load_rows(ex_path))
    attempts: list[dict] = load_rows(OUT / "audits" / "attempt_log.jsonl")
    audits = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else []
    done = {(r["original_gold_answer"], r["question"]) for r in part_c}
    tydi = read_jsonl(PROJECT_ROOT / "data/processed/tydi_train_questions.jsonl")
    # Part C has a high remnant-evidence rejection rate (~86%), so draw from the
    # full heldout TyDi pool (1390) rather than a small fixed slice.
    pool = sample_tydi_questions_for_part_c(tydi, exclusions, target=len(tydi), seed=SEED)

    def after_dense(question, removed, top_k=20):
        removed_set = set(str(x) for x in removed)
        return [(p, s) for p, s in dense_audit(question, top_k=top_k) if p not in removed_set]

    def after_bm25(question, removed, top_k=20):
        removed_set = set(str(x) for x in removed)
        return [(p, s) for p, s in bm25_audit(question, top_k=top_k) if p not in removed_set]

    def remnant_check(question, answer, top_ids, removed):
        removed_set = set(str(x) for x in removed)
        answer_text = normalize_space(answer)
        if not answer_text:
            return []
        return [
            pid
            for pid in top_ids[:10]
            if pid not in removed_set and compact_contains(passage_text.get(pid, ""), answer_text)
        ]

    for src in pool:
        if len(part_c) >= TARGET_C:
            break
        key = (normalize_space((src.get("gold_answers") or [""])[0]), normalize_space(src.get("question", "")))
        if key in done:
            continue
        rows, ex, aud = build_part_c_removed_evidence(
            [src],
            passage_text=passage_text,
            batch_id="batch2full",
            seed=SEED,
            bm25_audit_after_removal=after_bm25,
            dense_audit_after_removal=after_dense,
            remnant_check_fn=remnant_check,
            max_items=1,
        )
        attempts.append(
            {
                "part": "C",
                "source_question_id": src.get("question_id"),
                "accepted": len(rows),
                "excluded": len(ex.rows),
                "remnant": aud[0].get("hidden_evidence_remaining") if aud else [],
            }
        )
        audits.extend(aud)
        if rows:
            row = dict(rows[0])
            row["needs_human_check"] = True
            row["my_decision"] = ""
            row["my_notes"] = ""
            part_c.append(row)
            done.add(key)
            write_csv(csv_path, PART_C_FIELDS, part_c)
            (OUT / "checkpoint.json").write_text(
                canonical_json({"part": "C", "accepted": len(part_c), "ts": time.time()}), encoding="utf-8"
            )
            audit_path.write_text(canonical_json(audits), encoding="utf-8")
            write_jsonl(OUT / "audits" / "attempt_log.jsonl", dedupe_rows(attempts))
            if len(part_c) % 25 == 0:
                print(f"C accepted={len(part_c)}", flush=True)
        for er in ex.rows:
            exclusion.add(
                part="C",
                item_id=str(src.get("question_id", "")),
                reason=er["reason"],
                stage=er["stage"],
                details=er.get("details", ""),
            )
        write_jsonl(ex_path, dedupe_rows(exclusion.rows))
        audit_path.write_text(canonical_json(audits), encoding="utf-8")
        write_jsonl(OUT / "audits" / "attempt_log.jsonl", dedupe_rows(attempts))
    if len(part_c) < TARGET_C:
        raise RuntimeError(f"PART_C_SHORTFALL accepted={len(part_c)} target={TARGET_C}; no fabrication/no silent reduction")
    return part_c, exclusion, audits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "audits").mkdir(parents=True, exist_ok=True)
    mismatches = verify_protected_checksums(PROJECT_ROOT, PROJECT_ROOT / "data/custom_batch2/protected_checksums.json")
    if mismatches:
        raise SystemExit(f"PROTECTED_CHECKSUM_MISMATCH: {mismatches}")

    budget = BudgetGuard()
    client = GeminiClient(timeout=30)
    verification = GeminiCallSpec(
        model=MODEL,
        prompt='Return JSON {"ok":true}.',
        temperature=TEMPERATURE,
        seed=SEED,
        max_output_tokens=64,
        response_schema={"type": "OBJECT", "properties": {"ok": {"type": "BOOLEAN"}}, "required": ["ok"]},
        thinking_budget=THINKING_BUDGET,
    )
    try:
        verify = client.generate(verification, use_cache=True)
    except Exception as exc:
        if classify_gemini_error(exc) == "daily_quota_exhausted":
            info = {
                "status": "QUOTA_EXHAUSTED",
                "part": "verification",
                "error": repr(exc),
                "retry_after_seconds": _retry_after(exc),
                "accepted_at_stop": {
                    "A": _csv_count(out / "batch2_answerable_review.csv"),
                    "B": _csv_count(out / "batch2_unanswerable_review.csv"),
                    "C": _csv_count(out / "batch2_removed_evidence_review.csv"),
                },
            }
            (out / "quota_exhausted.json").write_text(canonical_json(info), encoding="utf-8")
            print(json.dumps(info, ensure_ascii=False), flush=True)
            return 0
        raise
    budget.note(verify.usage.__dict__, verify.cached, "verification")

    exclusions = load_exclusion_inventory(PROJECT_ROOT / "data/custom_batch2/source_inventory.json")
    passages = load_passages(PROJECT_ROOT / "data/processed/corpus.jsonl")
    unit_parents = [
        row.get("parent_id", row["passage_id"])
        for row in (json.loads(line) for line in (PROJECT_ROOT / "data/processed/corpus_chunks.jsonl").open(encoding="utf-8"))
    ]
    passage_text = {p.passage_id: p.text for p in passages}
    corpus_pairs = [(p.passage_id, p.text) for p in passages if p.source == "miracl"]
    rare_terms = build_rare_terms([p.text for p in passages if p.source == "miracl"][:8000])

    eligible = load_eligible(passages, exclusions, out / "audits" / "e5_eligible_candidates.jsonl")
    print(f"eligible={len(eligible)}", flush=True)
    if len(eligible) < TARGET_A + TARGET_B + 200:
        print("WARNING: eligible pool is tight; attempts may exhaust.", flush=True)

    bm25 = BM25Index.load(PROJECT_ROOT / "data/cache/bm25_index.pkl")
    from sentence_transformers import SentenceTransformer

    emb = np.load(PROJECT_ROOT / "data/embeddings/corpus_units.f16.npy", mmap_mode="r")
    encoder = SentenceTransformer(E5_MODEL, device="cpu")
    parent_index = build_parent_index(unit_parents)
    corpus_search = CorpusSearch(corpus_pairs, rare_terms)

    def bm25_audit(question, top_k=20):
        return bm25_search(bm25, parent_index, question, top_k=top_k)

    def dense_audit(question, top_k=20):
        qv = encoder.encode([E5_QUERY_PREFIX + question], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        return dense_search(emb, qv, parent_index, top_k=top_k)

    provider_a = GeminiProvider(client, ANSWER_SCHEMA, budget, "part_a", out / "audits")
    provider_b = GeminiProvider(client, UNANSWERABLE_SCHEMA, budget, "part_b", out / "audits")

    try:
        part_a, ex_a = run_part_a(eligible, provider_a, corpus_pairs, rare_terms, bm25_audit, dense_audit, corpus_search.alt_gold_ids)
        used = {r["gold_passage_id"] for r in part_a}
        part_b, ex_b, audits_b = run_part_b(eligible, provider_b, passage_text, bm25_audit, dense_audit, used)
        part_c, ex_c, audits_c = run_part_c(exclusions, passage_text, bm25_audit, dense_audit)
    except QuotaExhausted as exc:
        info = dict(exc.info)
        info["status"] = "QUOTA_EXHAUSTED"
        info["accepted_at_stop"] = {
            "A": _csv_count(out / "batch2_answerable_review.csv"),
            "B": _csv_count(out / "batch2_unanswerable_review.csv"),
            "C": _csv_count(out / "batch2_removed_evidence_review.csv"),
        }
        info["budget"] = budget.summary()
        (out / "quota_exhausted.json").write_text(canonical_json(info), encoding="utf-8")
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return 0

    write_part_csvs(out, part_a=part_a, part_b=part_b, part_c=part_c)
    write_csv(out / "batch2_answerable_review.csv", PART_A_FIELDS, part_a)
    write_csv(out / "batch2_unanswerable_review.csv", PART_B_FIELDS, part_b)
    write_csv(out / "batch2_removed_evidence_review.csv", PART_C_FIELDS, part_c)
    all_ex = ex_a.rows + ex_b.rows + ex_c.rows
    write_jsonl(out / "excluded_candidates.jsonl", all_ex)
    qc_paths = write_qc_artifacts(
        out / "qc",
        part_a=part_a,
        part_b=part_b,
        part_c=part_c,
        exclusions=all_ex,
        audits={"part_b_count": len(audits_b), "part_c_count": len(audits_c), "budget": budget.summary()},
    )

    priorities = review_priority_rows(part_a, part_b, part_c)
    write_csv(out / "review_priority.csv", list(priorities[0].keys()) if priorities else [], priorities)
    mandatory = mandatory_review_rows(priorities)
    write_csv(out / "mandatory_review.csv", list(mandatory[0].keys()) if mandatory else [], mandatory)
    sample_rows = []
    for part, rows, n, s in (("A", part_a, 35, SEED), ("B", part_b, 35, SEED + 1), ("C", part_c, 25, SEED + 2)):
        for r in random_review_sample(rows, n=n, seed=s):
            sample_rows.append(
                {
                    "part": part,
                    "batch_id": r.get("batch_id", ""),
                    "question": r.get("question", ""),
                    "question_type": r.get("question_type", ""),
                    "gold_answer": r.get("gold_answer", r.get("original_gold_answer", "")),
                    "gold_passage_id": r.get("gold_passage_id", r.get("original_gold_passage_id", "")),
                    "filter_flags": r.get("filter_flags", ""),
                    "needs_human_check": r.get("needs_human_check", True),
                    "my_decision": "",
                    "my_notes": "",
                }
            )
    write_csv(
        out / "random_review_sample.csv",
        ["part", "batch_id", "question", "question_type", "gold_answer", "gold_passage_id", "filter_flags", "needs_human_check", "my_decision", "my_notes"],
        sample_rows,
    )
    (out / "audits" / "part_b_audit.json").write_text(canonical_json(audits_b), encoding="utf-8")
    (out / "audits" / "part_c_audit.json").write_text(canonical_json(audits_c), encoding="utf-8")
    (out / "token_usage_and_cost.json").write_text(canonical_json(budget.summary()), encoding="utf-8")

    mismatches_after = verify_protected_checksums(PROJECT_ROOT, PROJECT_ROOT / "data/custom_batch2/protected_checksums.json")
    report = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "thinking_budget": THINKING_BUDGET,
        "targets": {"A": TARGET_A, "B": TARGET_B, "C": TARGET_C},
        "accepted": {"A": len(part_a), "B": len(part_b), "C": len(part_c), "total": len(part_a) + len(part_b) + len(part_c)},
        "excluded": len(all_ex),
        "budget": budget.summary(),
        "protected_checksums_before": mismatches,
        "protected_checksums_after": mismatches_after,
        "part_b_policy": "likely_answerable and manual_evidence_review queued for human review, never auto-excluded",
        "max_attempts": MAX_ATTEMPTS,
        "part_b_audits": len(audits_b),
        "part_c_audits": len(audits_c),
        "qc_paths": qc_paths,
    }
    (out / "quality_control_report.json").write_text(canonical_json(report), encoding="utf-8")
    (out / "generation_audit.json").write_text(canonical_json(report), encoding="utf-8")
    (out / "retrieval_audit_summary.json").write_text(
        canonical_json({"part_b_count": len(audits_b), "part_c_count": len(audits_c), "part_b_audit_path": "audits/part_b_audit.json", "part_c_audit_path": "audits/part_c_audit.json"}),
        encoding="utf-8",
    )
    (out / "quality_control_report.md").write_text(
        "\n".join(
            [
                "# Batch 2 Full Quality Control Report",
                "",
                f"- Model: `{MODEL}` temperature={TEMPERATURE} seed={SEED} thinking_budget={THINKING_BUDGET}",
                f"- Accepted: A={len(part_a)}/{TARGET_A}, B={len(part_b)}/{TARGET_B}, C={len(part_c)}/{TARGET_C}, total={len(part_a)+len(part_b)+len(part_c)}",
                f"- Budget: calls={budget.calls} cached={budget.cached_calls} input={budget.input_tokens} output={budget.output_tokens} thinking={budget.thinking_tokens} est=${budget.cost:.6f} cap=${BUDGET_CAP_USD}",
                f"- Protected checksums after: {mismatches_after or 'OK'}",
                "- All accepted rows: needs_human_check=true, my_decision=\"\", my_notes=\"\".",
                "- No generated question was marked human-approved.",
                "- Part C used deterministic removal/audits and no Gemini calls.",
                "- Part B likely-answerable/manual-evidence items are queued for human review, not auto-excluded.",
                "- Review files: batch2_answerable_review.csv, batch2_unanswerable_review.csv, batch2_removed_evidence_review.csv, excluded_candidates.jsonl, mandatory_review.csv, random_review_sample.csv, review_priority.csv, audits/*.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "change_log.md").write_text(
        "\n".join(
            [
                "# Batch 2 Change Log",
                "",
                f"- Generated full approved batch with seed {SEED}, temperature {TEMPERATURE}, thinking_budget {THINKING_BUDGET}.",
                f"- Part A: {len(part_a)} accepted, max {MAX_ATTEMPTS} attempts per target, passage replacement on failure, no accepted passage reuse.",
                f"- Part B: {len(part_b)} accepted; likely-answerable/manual-review candidates queued for human review.",
                f"- Part C: {len(part_c)} accepted via deterministic TyDi rule-based removal; no Gemini calls.",
                "- Protected/frozen files unchanged (checksums verified before/after).",
                "- No row was human-approved; my_decision/my_notes left empty; needs_human_check=true.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        make_word_report(out, report, part_a, part_b, part_c, all_ex)
    except Exception:
        (out / "word_report_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "budget": budget.summary(),
                "checksums_after": mismatches_after,
                "out": str(out),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if mismatches_after:
        raise SystemExit("PROTECTED FILES CHANGED")
    return 0


def make_word_report(out: Path, report: dict, part_a, part_b, part_c, all_ex) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading("Custom Evaluation Dataset — Batch 2", level=0)
    doc.add_paragraph(f"Model: {MODEL}; seed={SEED}; temperature={TEMPERATURE}; thinking_budget={THINKING_BUDGET}")
    doc.add_heading("Accepted Counts", level=1)
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "Part"
    table.rows[0].cells[1].text = "Accepted"
    table.rows[0].cells[2].text = "Target"
    for p, k in (("A", "A"), ("B", "B"), ("C", "C")):
        cells = table.add_row().cells
        cells[0].text = p
        cells[1].text = str(report["accepted"][k])
        cells[2].text = str(report["targets"][k])
    doc.add_heading("Budget / Tokens", level=1)
    b = report["budget"]
    doc.add_paragraph(
        f"Live calls={b['calls']}, cached={b['cached_calls']}, input={b['input_tokens']}, output={b['output_tokens']}, "
        f"thinking={b['thinking_tokens']}, estimated cost=${b['cost_usd_est']:.6f} (cap ${b['cap_usd']})."
    )
    doc.add_heading("Review Notes", level=1)
    for text in (
        "Every accepted row has needs_human_check=true with empty my_decision/my_notes.",
        "Part B likely-answerable and manual-evidence-review candidates are queued for human review rather than auto-excluded.",
        "Part C is deterministic and used no Gemini calls.",
        "Protected/frozen files were checksum-verified before and after; no protected file was modified.",
        f"Excluded candidates recorded: {len(all_ex)}.",
    ):
        doc.add_paragraph(text, style="List Bullet")
    doc.save(out / "Batch2_Full_Results_and_Approach.docx")


if __name__ == "__main__":
    raise SystemExit(main())
