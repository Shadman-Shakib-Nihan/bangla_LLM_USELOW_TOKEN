#!/usr/bin/env python3
"""Human-in-the-loop review of the custom Bengali QA datasets."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
ANSWERABLE = DATA / "custom_drafts_v2.csv"
UNANSWERABLE = DATA / "custom_drafts_unanswerable.csv"
RETRIEVAL = DATA / "custom_drafts_unanswerable_retrieval.csv"
EXPORT = DATA / "custom_questions.csv"
PROTECTED = {"custom_drafts.csv", "custom_drafts_excluded.csv", "question_type_review.csv"}
DECISIONS = {"k": "keep", "f": "fix", "r": "reject"}


class ReviewError(Exception):
    """Raised for invalid or unsafe dataset operations."""


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
                    raise ReviewError(f"{path}: row {line_number} has more fields than the header")
                rows.append({key: value if value is not None else "" for key, value in row.items()})
            if reader.fieldnames is None:
                raise ReviewError(f"{path}: missing CSV header")
        return reader.fieldnames, rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not read {path}: {exc}") from exc


def validate_columns(path: Path, required: set[str]) -> tuple[list[str], list[dict[str, str]]]:
    columns, rows = load_csv(path)
    missing = sorted(required - set(columns))
    if missing:
        raise ReviewError(f"{path}: missing required columns: {', '.join(missing)}")
    return columns, rows


def validate_unique_ids(rows: list[dict[str, str]], path: Path) -> None:
    ids = [row.get("id", "") for row in rows]
    duplicates = sorted(value for value, count in Counter(ids).items() if value and count > 1)
    if duplicates:
        raise ReviewError(f"{path}: duplicate IDs: {', '.join(duplicates[:10])}")


def retrieval_map(questions: list[dict[str, str]], passages: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    question_by_id = {row["id"]: row for row in questions}
    for passage in passages:
        pid = passage.get("id", "")
        if pid not in question_by_id:
            raise ReviewError(f"Retrieval passage has unknown question ID: {pid}")
        if passage.get("question", "") != question_by_id[pid].get("question", ""):
            raise ReviewError(f"Question text mismatch for retrieval ID {pid}")
        grouped[pid].append(passage)
    for q in questions:
        related = grouped.get(q["id"], [])
        if len(related) != 5:
            raise ReviewError(f"Question {q['id']} has {len(related)} retrieval passages; expected 5")
        try:
            related.sort(key=lambda p: int(p.get("rank", "")))
        except ValueError as exc:
            raise ReviewError(f"Question {q['id']} has a nonnumeric retrieval rank") from exc
        if [int(p["rank"]) for p in related] != [1, 2, 3, 4, 5]:
            raise ReviewError(f"Question {q['id']} does not have ranks 1 through 5")
    return dict(grouped)


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = path.with_name(f"{path.name}.{stamp}.bak")
    shutil.copy2(path, destination)
    return destination


def save_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    if path.name in PROTECTED:
        raise ReviewError(f"Refusing to modify protected file: {path}")
    if path.exists():
        backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def counts(rows: list[dict[str, str]]) -> Counter:
    return Counter((row.get("my_decision") or "").strip().lower() for row in rows)


def progress(rows: list[dict[str, str]]) -> None:
    c = counts(rows)
    reviewed = sum(c[x] for x in ("keep", "fix", "reject"))
    print(f"Progress: reviewed {reviewed}/{len(rows)} | remaining {len(rows)-reviewed} | keep {c['keep']} | fix {c['fix']} | reject {c['reject']}")


def ask(prompt: str, input_fn=input) -> str:
    return input_fn(prompt).strip()


def apply_decision(row: dict[str, str], decision: str, note: str = "") -> None:
    normalized = decision.strip().lower()
    if normalized not in {"keep", "fix", "reject"}:
        raise ReviewError(f"Unsupported decision: {decision}")
    if normalized in {"fix", "reject"} and not note.strip():
        raise ReviewError(f"A note is required for {normalized}")
    row["my_decision"] = normalized
    row["my_notes"] = note.strip() if normalized in {"fix", "reject"} else ""


def review_loop(path: Path, columns: list[str], rows: list[dict[str, str]], selected: list[dict[str, str]], kind: str, retrievals=None, input_fn=input) -> None:
    for row in selected:
        previous = (row.get("my_decision") or "").strip()
        if previous:
            confirmation = ask(f"{row['id']} already has decision '{previous}'. Replace it? [y/N] ", input_fn).lower()
            if confirmation != "y":
                print("Keeping existing decision.")
                continue
        print("\n" + "=" * 72)
        print(f"ID: {row.get('id', '')}\nQuestion: {row.get('question', '')}")
        if kind == "answerable":
            print(f"Gold answer: {row.get('gold_answer', '')}\nPassage: {row.get('passage_snippet', '')}")
        else:
            passages = retrievals[row["id"]]
            for passage in passages:
                print(f"\nRank {passage['rank']} | {passage.get('passage_id', '')} | {passage.get('title', '')}\n{passage.get('snippet', '')}")
        while True:
            choice = ask("Decision [k=keep, f=fix, r=reject, s=skip, q=quit]: ", input_fn).lower()
            if choice == "q":
                print("Review saved; exiting.")
                return
            if choice == "s":
                break
            if choice not in DECISIONS:
                print("Enter k, f, r, s, or q.")
                continue
            if kind == "unanswerable" and choice == "f":
                print("Fix is not available for unanswerable questions.")
                continue
            decision = DECISIONS[choice]
            if kind == "unanswerable" and decision == "keep":
                print("Keep only if none of the five passages answers the question.")
            note = ""
            if decision in {"fix", "reject"}:
                note = ask(f"Short {decision} reason (required): ", input_fn)
                if not note:
                    print("A note is required; decision was not saved.")
                    continue
            apply_decision(row, decision, note)
            save_csv(path, columns, rows)
            break
        progress(rows)
        if kind == "answerable" and counts(rows)["keep"] >= 40:
            choice = ask("40 keep decisions reached. Continue reviewing? [y/N] ", input_fn).lower()
            if choice != "y":
                return


def inspect() -> None:
    for path in (ANSWERABLE, UNANSWERABLE, RETRIEVAL):
        columns, rows = load_csv(path)
        print(f"{path.relative_to(BASE)}: {len(rows)} records\n  columns: {', '.join(columns)}")
    _, questions = validate_columns(UNANSWERABLE, {"id", "question", "my_decision", "my_notes"})
    _, passages = validate_columns(RETRIEVAL, {"id", "question", "rank", "passage_id", "snippet"})
    validate_unique_ids(questions, UNANSWERABLE)
    grouped = retrieval_map(questions, passages)
    print(f"Retrieval mapping: valid; {len(grouped)} questions, five ranked passages each (150 total).")


def status() -> None:
    _, rows = validate_columns(ANSWERABLE, {"id", "my_decision", "my_notes"})
    c = counts(rows)
    print("Answerable")
    print(f"  source: {len(rows)} | reviewed: {sum(c[k] for k in ('keep','fix','reject'))} | keep: {c['keep']} | fix: {c['fix']} | reject: {c['reject']} | pending: {len(rows)-sum(c[k] for k in ('keep','fix','reject'))}")
    _, urows = validate_columns(UNANSWERABLE, {"id", "my_decision", "my_notes"})
    uc = counts(urows)
    print("Unanswerable")
    print(f"  source: {len(urows)} | reviewed: {sum(uc[k] for k in ('keep','reject'))} | keep: {uc['keep']} | reject: {uc['reject']} | pending: {len(urows)-uc['keep']-uc['reject']}")


def export(force: bool = False) -> None:
    required = {"id", "question", "gold_answer", "passage_snippet", "my_decision"}
    columns, rows = validate_columns(ANSWERABLE, required)
    validate_unique_ids(rows, ANSWERABLE)
    kept = [row for row in rows if (row.get("my_decision") or "").strip().lower() == "keep"]
    out_columns = [column for column in columns if column != "my_notes"]
    print(f"Total source questions: {len(rows)}")
    c = counts(rows)
    print(f"Total reviewed: {sum(c[k] for k in ('keep','fix','reject'))}")
    print(f"Total keep: {c['keep']}\nTotal fix: {c['fix']}\nTotal reject: {c['reject']}\nTotal pending: {len(rows)-sum(c[k] for k in ('keep','fix','reject'))}\nTotal exported: {len(kept)}")
    if not kept:
        print("No approved answerable questions to export; output was not changed.")
        return
    if EXPORT.exists() and not force:
        raise ReviewError(f"{EXPORT} already exists (and contains user input); use --force to replace it. A backup will be created.")
    save_csv(EXPORT, out_columns, [{column: row.get(column, "") for column in out_columns} for row in kept])
    try:
        display_path = EXPORT.relative_to(BASE)
    except ValueError:
        display_path = EXPORT
    print(f"Exported {len(kept)} unique kept questions to {display_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Show input schemas, counts, and retrieval mapping validity")
    sub.add_parser("status", help="Show review progress for both datasets")
    ans = sub.add_parser("answerable", help="Interactively review answerable questions")
    group = ans.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, help="Review at most N unanswered questions (default: 10)")
    group.add_argument("--all", action="store_true", help="Review all unanswered questions")
    group.add_argument("--ids", nargs="+", help="Review these IDs in source order")
    una = sub.add_parser("unanswerable", help="Review unanswerable questions with five retrieved passages")
    una.add_argument("--limit", type=int, default=30, help="Review at most N unanswered questions (default: 30)")
    exp = sub.add_parser("export", help="Export kept answerable questions to data/custom_questions.csv")
    exp.add_argument("--force", action="store_true", help="Replace existing export after backing it up")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            inspect()
        elif args.command == "status":
            status()
        elif args.command == "export":
            export(args.force)
        elif args.command == "answerable":
            columns, rows = validate_columns(ANSWERABLE, {"id", "question", "gold_answer", "passage_snippet", "my_decision", "my_notes"})
            validate_unique_ids(rows, ANSWERABLE)
            pending = [row for row in rows if not (row.get("my_decision") or "").strip()]
            if args.ids:
                by_id = {row["id"]: row for row in rows}
                missing = [item for item in args.ids if item not in by_id]
                if missing:
                    raise ReviewError(f"Unknown answerable IDs: {', '.join(missing)}")
                selected = [by_id[item] for item in args.ids]
            else:
                selected = pending if args.all else pending[:args.limit or 10]
            print(f"Starting with {len(selected)} question(s); all existing records and columns are retained.")
            review_loop(ANSWERABLE, columns, rows, selected, "answerable")
        elif args.command == "unanswerable":
            columns, rows = validate_columns(UNANSWERABLE, {"id", "question", "my_decision", "my_notes"})
            passages_columns, passages = validate_columns(RETRIEVAL, {"id", "question", "rank", "passage_id", "snippet"})
            validate_unique_ids(rows, UNANSWERABLE)
            grouped = retrieval_map(rows, passages)
            pending = [row for row in rows if not (row.get("my_decision") or "").strip()]
            review_loop(UNANSWERABLE, columns, rows, pending[:args.limit], "unanswerable", grouped)
        return 0
    except (ReviewError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
