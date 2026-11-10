"""QC reports and review-file generation for Batch 2 (deterministic)."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.batch2_text import BATCH2_SEED, canonical_json, sha256_file


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def summarize_parts(part_a: list[dict[str, Any]], part_b: list[dict[str, Any]], part_c: list[dict[str, Any]], exclusions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    exclusions = exclusions or []
    summary = {
        "seed": BATCH2_SEED,
        "counts": {"part_a": len(part_a), "part_b": len(part_b), "part_c": len(part_c), "total": len(part_a) + len(part_b) + len(part_c), "excluded": len(exclusions)},
        "question_types": dict(Counter([r.get("question_type", "") for r in part_a + part_b + part_c])),
        "exclusion_reasons": dict(Counter([r.get("reason", "") for r in exclusions])),
        "human_checks": {
            "part_a_all_true": all(str(r.get("needs_human_check")).lower() in {"true", "1"} for r in part_a),
            "part_b_all_true": all(str(r.get("needs_human_check")).lower() in {"true", "1"} for r in part_b),
            "part_c_all_true": all(str(r.get("needs_human_check")).lower() in {"true", "1"} for r in part_c),
        },
        "decisions_empty": all((r.get("my_decision", "") == "" and r.get("my_notes", "") == "") for r in part_a + part_b + part_c),
    }
    return summary


def review_priority_rows(part_a: list[dict[str, Any]], part_b: list[dict[str, Any]], part_c: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(part: str, r: dict[str, Any], reason: str, priority: str) -> None:
        rows.append({
            "part": part,
            "batch_id": r.get("batch_id", ""),
            "question_type": r.get("question_type", ""),
            "question": r.get("question", ""),
            "priority": priority,
            "reason": reason,
            "filter_flags": r.get("filter_flags", ""),
            "my_decision": r.get("my_decision", ""),
            "my_notes": r.get("my_notes", ""),
        })
    for r in part_a:
        flags = str(r.get("filter_flags", ""))
        if flags or str(r.get("has_near_duplicate_answer")).lower() == "true":
            add("A", r, flags or "near_duplicate_answer", "mandatory")
        if str(r.get("self_retrieval_gold_rank_bm25", "")).isdigit() and int(r["self_retrieval_gold_rank_bm25"]) > 10:
            add("A", r, "bm25_self_retrieval_gt_10", "high")
    for r in part_b:
        if str(r.get("plausible_answerable", "")).lower() == "true" or r.get("hidden_evidence_candidate"):
            add("B", r, "hidden_evidence_candidate", "mandatory")
        else:
            add("B", r, "unanswerable_manual_review", "mandatory")
    for r in part_c:
        if r.get("hidden_evidence_remaining"):
            add("C", r, "remnant_evidence", "mandatory")
        else:
            add("C", r, "removed_evidence_manual_review", "high")
    dedup = {}
    for row in rows:
        key = (row["part"], row["question"], row["reason"])
        prev = dedup.get(key)
        if prev is None or row["priority"] == "mandatory":
            dedup[key] = row
    return list(dedup.values())


def mandatory_review_rows(priority_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in priority_rows if r.get("priority") == "mandatory"]


def random_review_sample(rows: list[dict[str, Any]], *, n: int, seed: int = BATCH2_SEED) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    rng = random.Random(seed)
    pool = list(rows)
    pool.sort(key=lambda r: (r.get("part", ""), r.get("question", "")))
    rng.shuffle(pool)
    return pool[: min(n, len(pool))]


def write_qc_artifacts(
    output_dir: str | Path,
    *,
    part_a: list[dict[str, Any]],
    part_b: list[dict[str, Any]],
    part_c: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    audits: dict[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_parts(part_a, part_b, part_c, exclusions)
    if audits:
        summary["audits"] = audits
    priorities = review_priority_rows(part_a, part_b, part_c)
    mandatory = mandatory_review_rows(priorities)
    sample = random_review_sample(part_a + part_b + part_c, n=max(10, min(40, (len(part_a) + len(part_b) + len(part_c)) // 10 or 0)), seed=BATCH2_SEED)
    paths = {
        "summary_json": str(out / "qc_summary.json"),
        "summary_md": str(out / "qc_summary.md"),
        "review_priority_csv": str(out / "review_priority.csv"),
        "mandatory_review_csv": str(out / "mandatory_review.csv"),
        "random_review_sample_csv": str(out / "random_review_sample.csv"),
        "exclusions_jsonl": str(out / "exclusions.jsonl"),
    }
    Path(paths["summary_json"]).write_text(canonical_json(summary), encoding="utf-8")
    _write_csv(paths["review_priority_csv"], priorities)
    _write_csv(paths["mandatory_review_csv"], mandatory)
    _write_csv(paths["random_review_sample_csv"], sample)
    Path(paths["exclusions_jsonl"]).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in exclusions), encoding="utf-8"
    )
    lines = ["# Batch 2 QC Summary", "", f"Seed: `{BATCH2_SEED}`", ""]
    lines.append("| part | accepted |")
    lines.append("|---|---:|")
    for k, v in summary["counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Human review", "- `needs_human_check=true` for all accepted rows", "- `my_decision` and `my_notes` left empty", "- no automatic approval", ""]
    Path(paths["summary_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
