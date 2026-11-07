"""Phase 2: build the retrieval corpus and question sets.

Composition (per approved plan):
  * MIRACL Bengali: all passages judged in dev+train qrels (13,705) plus a
    seeded random sample of non-judged passages to reach 100,000 MIRACL passages.
  * TyDi QA Bengali: all unique GoldP contexts, IDs prefixed "tydi_".

Question sets: MIRACL train (tuning) / dev (reporting); TyDi train / validation.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/build_corpus.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus import (
    PROCESSED,
    CorpusManifest,
    build_corpus,
    build_question_sets,
)

SEED = 42


def show_random_passages(path: Path, n: int = 5, seed: int = 7) -> None:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            if len(reservoir) < n:
                reservoir.append(rec)
            else:
                j = rng.randint(0, total - 1)
                if j < n:
                    reservoir[j] = rec
    print(f"\n5 random passages from {path.relative_to(PROJECT_ROOT)} "
          f"(n={total}, sampling seed={seed}):")
    for rec in reservoir:
        print("-" * 70)
        print("passage_id:", rec["passage_id"], "| source:", rec["source"],
              "| title:", rec["title"])
        print("text:", rec["text"][:260].replace("\n", " "))


def main() -> int:
    manifest: CorpusManifest = build_corpus(seed=SEED)
    questions = build_question_sets()

    print("=" * 72)
    print("CORPUS COMPOSITION")
    print("=" * 72)
    print(f"seed: {manifest.seed}")
    print(f"target MIRACL passages: {manifest.target_miracl_passages:,}")
    print(f"MIRACL corpus total (source): {manifest.miracl['corpus_total_passages']:,}")
    print(f"MIRACL judged (dev+train union): {manifest.miracl['judged_unique']:,}")
    print(f"MIRACL non-judged available: {manifest.miracl['non_judged_available']:,}")
    print(f"MIRACL random sample (seed={manifest.seed}): {manifest.miracl['random_sampled']:,}")
    print(f"MIRACL written: {manifest.miracl['written']:,}")
    print(f"TyDi Bengali unique contexts written: {manifest.tydi['unique_contexts_written']:,}")
    print(f"COMBINED corpus passages: {manifest.combined_total:,}")

    print("\n" + "=" * 72)
    print("QUESTION SETS")
    print("=" * 72)
    for name, info in questions.items():
        print(f"{name:16s} questions={info['questions']:5d} "
              f"with_gold={info['with_gold']:5d}  {info['file']}")

    show_random_passages(PROCESSED / "corpus.jsonl")

    stats = {
        "corpus_passages": manifest.combined_total,
        "manifest": str((PROCESSED / "corpus_manifest.json").relative_to(PROJECT_ROOT)),
        "questions": questions,
        "seed": manifest.seed,
    }
    (PROCESSED / "phase2_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[ok] wrote {PROCESSED / 'phase2_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
