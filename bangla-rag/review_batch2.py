#!/usr/bin/env python3
"""Human-in-the-loop review CLI for the Batch 2 custom QA datasets.

Mirrors review.py semantics for the three Batch 2 review CSVs:
  * keep / fix / reject decisions are stored in ``my_decision``
  * free-text notes are stored in ``my_notes``
  * an empty ``my_decision`` means pending
  * every row and column is preserved; only ``my_decision``/``my_notes`` change
  * a timestamped backup is created before a file is first modified

Commands:
    python review_batch2.py status
    python review_batch2.py inspect
    python review_batch2.py review --part A --limit 10
    python review_batch2.py review --part all --pending
    python review_batch2.py export
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/custom_batch2/batch2_full_approved"

PARTS = {
    "A": "batch2_answerable_review.csv",
    "B": "batch2_unanswerable_review.csv",
    "C": "batch2_removed_evidence_review.csv",
}
DECISIONS = {"k": "keep", "f": "fix", "r": "reject"}
VALID_DECISIONS = {"keep", "fix", "reject", ""}
NOTE_REQUIRED = {"fix", "reject"}
READ_ONLY = {"checkpoint.json", "quota_exhausted.json"}


class ReviewError(Exception):
    """Raised for invalid or unsafe review operations."""


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        csv.field_size_limit(sys.maxsize)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ReviewError(f"{path}: missing CSV header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ReviewError(f"{path}: duplicate column names")
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ReviewError(f"{path}:{line_number}: too many values")
                rows.append({k: (v if v is not None else "") for k, v in row.items()})
            return list(reader.fieldnames), rows
    except FileNotFoundError as exc:
        raise ReviewError(f"{path}: not found") from exc


def validate_columns(path: Path, rows_columns: list[str]) -> None:
    required = {"my_decision", "my_notes", "question"}
    missing = required - set(rows_columns)
    if missing:
        raise ReviewError(f"{path}: missing required columns {sorted(missing)}")


def row_key(row: dict[str, str], index: int) -> str:
    anchor = (
        row.get("gold_passage_id")
        or row.get("source_passage_hint")
        or row.get("original_gold_passage_id")
        or "row"
    )
    return f"{row.get('part', '?')}:{row.get('batch_id', '?')}:{anchor}:{index}"


def counts(rows: list[dict[str, str]]) -> Counter:
    normalized = []
    for row in rows:
        decision = (row.get("my_decision") or "").strip().lower()
        if decision not in VALID_DECISIONS:
            raise ReviewError(f"invalid decision {decision!r} (expected keep/fix/reject/empty)")
        normalized.append(decision or "pending")
    return Counter(normalized)


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = path.with_name(f"{path.name}.{stamp}.bak")
    shutil.copy2(path, dest)
    return dest


def save_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    if path.name in READ_ONLY:
        raise ReviewError(f"{path.name} is read-only")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def apply_decision(row: dict[str, str], decision: str, note: str = "") -> None:
    decision = decision.strip().lower()
    if decision not in VALID_DECISIONS:
        raise ReviewError(f"invalid decision {decision!r}")
    if decision in NOTE_REQUIRED and not note.strip():
        raise ReviewError(f"a note is required for {decision}")
    row["my_decision"] = decision
    row["my_notes"] = note.strip()


def _display(row: dict[str, str], part: str, index: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"[{part}] {index}/{total}  {row_key(row, index)}")
    print(f"type: {row.get('question_type', '')}")
    print("Q:", textwrap.shorten(row.get("question", ""), width=200))
    if part == "A":
        print("answer:", textwrap.shorten(row.get("gold_answer", ""), width=160))
        print("passage:", textwrap.shorten(row.get("gold_passage_text", ""), width=220))
        print(
            "retrieval: bm25_rank={} dense_rank={} flags={}".format(
                row.get("self_retrieval_gold_rank_bm25", ""),
                row.get("self_retrieval_gold_rank_dense", ""),
                row.get("filter_flags", ""),
            )
        )
    elif part == "B":
        print("plausible_answerable:", row.get("plausible_answerable", ""))
        print("why_not_answered:", textwrap.shorten(row.get("why_not_answered_from_corpus", ""), width=220))
        print("source_hint:", row.get("source_passage_hint", ""))
        print("top_retrieved:", row.get("top_retrieved_passage_ids", ""))
    else:
        print("original_answer:", textwrap.shorten(row.get("original_gold_answer", ""), width=160))
        print("removed_evidence:", row.get("removed_evidence_ids", ""))
        print("removal_reason:", row.get("removal_reason", ""))
        print("hidden_evidence_remaining:", row.get("hidden_evidence_remaining", "") or "(none)")
        print("top_after_removal:", row.get("top_retrieved_after_removal", ""))
    print(f"current: decision={row.get('my_decision', '') or '(pending)'} notes={row.get('my_notes', '') or ''}")


def review_loop(
    path: Path,
    columns: list[str],
    rows: list[dict[str, str]],
    selected: list[int],
    part: str,
    input_fn=input,
    dry_run: bool = False,
) -> None:
    if not selected:
        print(f"[{part}] nothing to review")
        return
    backed_up = False
    for position, idx in enumerate(selected, start=1):
        row = rows[idx]
        _display(row, part, position, len(selected))
        while True:
            choice = input_fn("keep[k] fix[f] reject[r] skip[s] quit[q]: ").strip().lower()
            if choice == "q":
                print(f"[{part}] saved; stopped early")
                return
            if choice == "s":
                break
            if choice not in DECISIONS:
                print("invalid choice")
                continue
            decision = DECISIONS[choice]
            note = ""
            if decision in NOTE_REQUIRED:
                note = input_fn("note (required): ")
            try:
                apply_decision(row, decision, note)
            except ReviewError as exc:
                print(exc)
                continue
            if dry_run:
                print(f"would set decision={decision}")
            else:
                if not backed_up:
                    dest = backup(path)
                    print(f"backup: {dest}")
                    backed_up = True
                save_csv(path, columns, rows)
                print(f"saved decision={decision}")
            break


def _load_part(part: str) -> tuple[Path, list[str], list[dict[str, str]]]:
    path = DATA / PARTS[part]
    columns, rows = load_csv(path)
    validate_columns(path, columns)
    return path, columns, rows


def status() -> None:
    totals = Counter()
    for part in ("A", "B", "C"):
        _, _, rows = _load_part(part)
        c = counts(rows)
        totals.update(c)
        reviewed = c["keep"] + c["fix"] + c["reject"]
        print(
            f"Part {part}: total={len(rows)} reviewed={reviewed} "
            f"keep={c['keep']} fix={c['fix']} reject={c['reject']} pending={c['pending']}"
        )
    reviewed = totals["keep"] + totals["fix"] + totals["reject"]
    print(
        f"Overall: total={sum(totals.values())} reviewed={reviewed} "
        f"keep={totals['keep']} fix={totals['fix']} reject={totals['reject']} pending={totals['pending']}"
    )


def inspect() -> None:
    for part in ("A", "B", "C"):
        path, columns, rows = _load_part(part)
        keys = [row_key(r, i) for i, r in enumerate(rows)]
        dup = len(keys) - len(set(keys))
        questions = [(r.get("question") or "").strip() for r in rows]
        dup_q = len(questions) - len(set(questions))
        print(
            f"Part {part}: {path.name} rows={len(rows)} columns={len(columns)} "
            f"duplicate_row_keys={dup} duplicate_questions={dup_q}"
        )
        print(f"  columns: {', '.join(columns)}")


def select_rows(rows: list[dict[str, str]], limit: int | None, all_rows: bool, pending_only: bool, start: int) -> list[int]:
    indices = list(range(len(rows)))
    if pending_only:
        indices = [i for i in indices if not (rows[i].get("my_decision") or "").strip()]
    indices = indices[start:]
    if not all_rows:
        indices = indices[: (limit if limit is not None else 10)]
    return indices


def export() -> None:
    combined = []
    for part in ("A", "B", "C"):
        _, columns, rows = _load_part(part)
        kept = [r for r in rows if (r.get("my_decision") or "").strip().lower() == "keep"]
        out = DATA / f"batch2_approved_{ {'A': 'answerable', 'B': 'unanswerable', 'C': 'removed_evidence'}[part] }.csv"
        save_csv(out, columns, kept)
        combined.extend({"part": part, **r} for r in kept)
        print(f"Part {part}: exported {len(kept)} kept rows -> {out}")
    jsonl = DATA / "batch2_approved.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        import json

        for row in combined:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Combined: exported {len(combined)} kept rows -> {jsonl}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review Batch 2 custom QA datasets")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show keep/fix/reject/pending counts")
    sub.add_parser("inspect", help="show schemas and duplicate-key checks")
    review = sub.add_parser("review", help="interactively review rows")
    review.add_argument("--part", choices=["A", "B", "C", "all"], default="A")
    group = review.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=10, help="review at most N pending rows (default 10)")
    group.add_argument("--all", action="store_true", help="review all matching rows")
    review.add_argument("--pending", action="store_true", help="only rows with an empty decision")
    review.add_argument("--start", type=int, default=0, help="skip the first N matching rows")
    review.add_argument("--dry-run", action="store_true", help="show decisions without saving")
    sub.add_parser("export", help="write kept rows to approved CSVs/JSONL")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            status()
        elif args.command == "inspect":
            inspect()
        elif args.command == "review":
            parts = ["A", "B", "C"] if args.part == "all" else [args.part]
            for part in parts:
                path, columns, rows = _load_part(part)
                selected = select_rows(rows, args.limit, args.all, args.pending, args.start)
                review_loop(path, columns, rows, selected, part, dry_run=args.dry_run)
        elif args.command == "export":
            export()
        return 0
    except ReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
