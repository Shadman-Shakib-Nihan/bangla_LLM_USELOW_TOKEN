"""Prepare the truncated-vs-split dense ablation.

Writes:
  * data/processed/overlength_parents_truncated.jsonl — the 1,501 over-length
    parents truncated to the e5 512-token input limit (text only; the
    ``passage: `` prefix is added at embedding time).
  * data/processed/overlength_gold_queries.jsonl — queries whose gold parent is
    one of those 1,501 passages, per reporting/tuning set.

The split condition already exists as corpus_chunks.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))

from transformers import AutoTokenizer  # noqa: E402

MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "passage: "
MAX_INPUT = 512
PARENTS = PROJECT_ROOT / "data" / "processed" / "overlength_parents.jsonl"
OUT_PARENTS = PROJECT_ROOT / "data" / "processed" / "overlength_parents_truncated.jsonl"
OUT_QUERIES = PROJECT_ROOT / "data" / "processed" / "overlength_gold_queries.jsonl"
QUESTIONS = {
    "miracl_train": PROJECT_ROOT / "data" / "processed" / "miracl_train_questions.jsonl",
    "miracl_dev": PROJECT_ROOT / "data" / "processed" / "miracl_dev_questions.jsonl",
    "tydi_train": PROJECT_ROOT / "data" / "processed" / "tydi_train_questions.jsonl",
    "tydi_validation": PROJECT_ROOT / "data" / "processed" / "tydi_validation_questions.jsonl",
}


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    prefix_tokens = len(tokenizer.encode(PREFIX, add_special_tokens=False))
    max_text_tokens = MAX_INPUT - prefix_tokens - special_tokens
    print(f"[ok] prefix_tokens={prefix_tokens} special={special_tokens} "
          f"max_text_tokens={max_text_tokens}")

    over_ids: set[str] = set()
    with PARENTS.open(encoding="utf-8") as fin, \
            OUT_PARENTS.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            over_ids.add(rec["passage_id"])
            ids = tokenizer.encode(rec["text"], add_special_tokens=False)
            truncated = tokenizer.decode(
                ids[:max_text_tokens], skip_special_tokens=True
            )
            fout.write(json.dumps({
                "passage_id": rec["passage_id"],
                "parent_id": rec["parent_id"],
                "title": rec["title"],
                "text": truncated,
                "source": rec["source"],
                "original_tokens": len(ids) + prefix_tokens + special_tokens,
                "truncated_tokens": min(len(ids), max_text_tokens)
                + prefix_tokens + special_tokens,
            }, ensure_ascii=False) + "\n")

    counts = Counter()
    by_set = Counter()
    with OUT_QUERIES.open("w", encoding="utf-8") as fout:
        for name, path in QUESTIONS.items():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    hit = sorted(set(row["gold_passage_ids"]) & over_ids)
                    if not hit:
                        continue
                    counts[name] += 1
                    by_set[name] += len(hit)
                    fout.write(json.dumps({
                        "dataset": name,
                        "question_id": row["question_id"],
                        "question": row["question"],
                        "gold_passage_ids": row["gold_passage_ids"],
                        "overlength_gold_ids": hit,
                        "n_overlength_gold": len(hit),
                    }, ensure_ascii=False) + "\n")

    print("[ok] queries with >=1 over-length gold passage:")
    for name in QUESTIONS:
        print(f"     {name:16s} {counts[name]:5d} queries "
              f"({counts[name] / sum(1 for _ in QUESTIONS[name].open()):.2%})")
    print(f"[ok] wrote {OUT_PARENTS.relative_to(PROJECT_ROOT)}")
    print(f"[ok] wrote {OUT_QUERIES.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
