"""Phase 3 gate: check e5 token lengths of every corpus passage.

Tokenizes all passages with the multilingual-e5-small tokenizer, including the
required ``passage: `` prefix and special tokens, without truncation. Reports the
count/share above the model's 512-token limit, median and p95 length, split by
MIRACL vs TyDi, plus a length histogram.

Does NOT embed anything.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/check_token_lengths.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
OUT = PROJECT_ROOT / "results" / "phase3_token_length_check.json"
MODEL_NAME = "intfloat/multilingual-e5-small"
LIMIT = 512
BATCH = 512
PREFIX = "passage: "
BUCKETS = [(0, 128), (129, 256), (257, 384), (385, 512), (513, 10**9)]


def bucket_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}"


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"[ok] tokenizer={MODEL_NAME} | model_max_length={tokenizer.model_max_length}")

    lengths: dict[str, list[int]] = defaultdict(list)
    ids: dict[str, list[str]] = defaultdict(list)
    batch: list[tuple[str, str, str]] = []

    def flush() -> None:
        if not batch:
            return
        texts = [PREFIX + text for _, text, _ in batch]
        enc = tokenizer(
            texts, add_special_tokens=True, truncation=False, return_length=True
        )
        for (source, _, pid), n in zip(batch, enc["length"]):
            lengths[source].append(int(n))
            ids[source].append(pid)
        batch.clear()

    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            batch.append((rec["source"], rec["text"], rec["passage_id"]))
            if len(batch) >= BATCH:
                flush()
    flush()

    def stats(values: list[int]) -> dict:
        arr = np.asarray(values)
        over = int((arr > LIMIT).sum())
        return {
            "passages": int(arr.size),
            "over_512": over,
            "share_over_512": round(over / arr.size, 4),
            "median_tokens": float(np.median(arr)),
            "p95_tokens": float(np.percentile(arr, 95)),
            "mean_tokens": round(float(arr.mean()), 1),
            "max_tokens": int(arr.max()),
        }

    report = {}
    for source in sorted(lengths):
        report[source] = stats(lengths[source])
        arr = np.asarray(lengths[source])
        report[source]["histogram"] = {}
        for lo, hi in BUCKETS:
            n = int(((arr >= lo) & (arr <= hi)).sum())
            report[source]["histogram"][bucket_label(lo, hi)] = {
                "passages": n,
                "share": round(n / arr.size, 4),
            }

    all_lengths = [n for vals in lengths.values() for n in vals]
    report["all"] = stats(all_lengths)
    report["all"]["histogram"] = {}
    for lo, hi in BUCKETS:
        n = int(sum(1 for x in all_lengths if lo <= x <= hi))
        report["all"]["histogram"][bucket_label(lo, hi)] = {
            "passages": n,
            "share": round(n / len(all_lengths), 4),
        }

    report["_meta"] = {
        "model": MODEL_NAME,
        "tokenizer_max_length": int(tokenizer.model_max_length),
        "limit": LIMIT,
        "prefix": PREFIX,
        "special_tokens_included": True,
        "truncation": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== e5 token lengths (prefix '{PREFIX}', special tokens, no truncation) ===")
    print(f"{'source':8s} {'passages':>9s} {'>512':>7s} {'share':>7s} "
          f"{'median':>7s} {'p95':>7s} {'mean':>8s} {'max':>7s}")
    for key in ["miracl", "tydi", "all"]:
        s = report[key]
        print(f"{key:8s} {s['passages']:9,d} {s['over_512']:7,d} "
              f"{s['share_over_512']*100:6.2f}% {s['median_tokens']:7.1f} "
              f"{s['p95_tokens']:7.1f} {s['mean_tokens']:8.1f} {s['max_tokens']:7,d}")

    print("\nHistogram (share of passages):")
    labels = [bucket_label(lo, hi) for lo, hi in BUCKETS]
    print(f"{'source':8s} " + " ".join(f"{l:>12s}" for l in labels))
    for key in ["miracl", "tydi", "all"]:
        h = report[key]["histogram"]
        print(f"{key:8s} " + " ".join(f"{h[l]['share']*100:11.2f}%" for l in labels))

    print(f"\n[ok] wrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
