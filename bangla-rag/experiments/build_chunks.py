"""Phase 3: build the chunked corpus.

Only the 1,501 over-length passages listed in
``results/phase3_over512_ids.json`` are split (sentence-aware, <=400 e5 tokens,
one-sentence overlap). Every other passage stays whole. All units keep
``parent_id``; the chunk file is the shared index for BM25 and dense retrieval.

Also writes the 1,501 original over-length parents for the truncated-vs-split
dense ablation.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/build_chunks.py
"""

from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.chunking import chunk_text  # noqa: E402

CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
OVER_IDS = PROJECT_ROOT / "results" / "phase3_over512_ids.json"
OUT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "corpus_chunks.jsonl"
OUT_PARENTS = PROJECT_ROOT / "data" / "processed" / "overlength_parents.jsonl"
OUT_STATS = PROJECT_ROOT / "results" / "phase3_chunking_stats.json"

MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "passage: "
MAX_TOKENS = 400


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(PREFIX + text, add_special_tokens=True))

    over_ids = set(json.loads(OVER_IDS.read_text(encoding="utf-8")))
    print(f"[ok] over-length parents: {len(over_ids):,}")

    stats = {
        "model": MODEL_NAME,
        "prefix": PREFIX,
        "max_tokens": MAX_TOKENS,
        "overlap_sentences": 1,
        "parents": 0,
        "overlength_parents": 0,
        "whole_passages": 0,
        "new_chunks": 0,
        "net_added": 0,
        "total_units": 0,
        "overlength_with_one_chunk": 0,
        "oversized_sentences": 0,
        "no_overlap_boundaries": 0,
        "post_fix_splits": 0,
        "overlength_by_source": {},
        "new_chunks_by_source": {},
        "chunks_per_parent": [],
        "new_chunk_token_lengths": [],
        "max_chunk_tokens": 0,
    }

    chunk_texts: list[str] = []
    found_over: set[str] = set()
    chunks_per_parent: list[int] = []

    OUT_CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS.open(encoding="utf-8") as fin, \
            OUT_CHUNKS.open("w", encoding="utf-8") as fout, \
            OUT_PARENTS.open("w", encoding="utf-8") as fpar:
        for line in fin:
            rec = json.loads(line)
            stats["parents"] += 1
            pid = rec["passage_id"]

            if pid not in over_ids:
                stats["whole_passages"] += 1
                fout.write(json.dumps({
                    **rec,
                    "parent_id": pid,
                    "chunk_index": 0,
                    "n_chunks": 1,
                    "is_chunked": False,
                    "is_truncated": False,
                }, ensure_ascii=False) + "\n")
                continue

            found_over.add(pid)
            stats["overlength_parents"] += 1
            stats["overlength_by_source"][rec["source"]] = (
                stats["overlength_by_source"].get(rec["source"], 0) + 1
            )
            fpar.write(json.dumps({
                "passage_id": pid,
                "parent_id": pid,
                "title": rec["title"],
                "text": rec["text"],
                "source": rec["source"],
                "token_length": count_tokens(rec["text"]),
            }, ensure_ascii=False) + "\n")

            chunks, cstats = chunk_text(rec["text"], count_tokens, MAX_TOKENS)
            if not chunks:
                raise RuntimeError(f"no chunks produced for {pid}")
            if len(chunks) == 1:
                stats["overlength_with_one_chunk"] += 1

            stats["new_chunks"] += len(chunks)
            stats["oversized_sentences"] += cstats.oversized_sentences
            stats["no_overlap_boundaries"] += cstats.no_overlap_boundaries
            stats["post_fix_splits"] += cstats.post_fix_splits
            stats["new_chunks_by_source"][rec["source"]] = (
                stats["new_chunks_by_source"].get(rec["source"], 0) + len(chunks)
            )
            chunks_per_parent.append(len(chunks))

            for idx, chunk in enumerate(chunks):
                n_tok = count_tokens(chunk)
                stats["max_chunk_tokens"] = max(stats["max_chunk_tokens"], n_tok)
                if n_tok > MAX_TOKENS:
                    raise RuntimeError(f"chunk {pid}::c{idx} exceeds {MAX_TOKENS}: {n_tok}")
                chunk_texts.append(chunk)
                fout.write(json.dumps({
                    "passage_id": f"{pid}::c{idx}",
                    "parent_id": pid,
                    "is_short": rec["is_short"],
                    "title": rec["title"],
                    "text": chunk,
                    "source": rec["source"],
                    "chunk_index": idx,
                    "n_chunks": len(chunks),
                    "is_chunked": True,
                    "is_truncated": False,
                }, ensure_ascii=False) + "\n")

    missing = over_ids - found_over
    if missing:
        raise RuntimeError(f"{len(missing)} over-length IDs not found in corpus")

    stats["net_added"] = stats["new_chunks"] - stats["overlength_parents"]
    stats["total_units"] = stats["whole_passages"] + stats["new_chunks"]

    # Token lengths for every new chunk, in batches.
    lengths: list[int] = []
    for start in range(0, len(chunk_texts), 512):
        batch = chunk_texts[start:start + 512]
        enc = tokenizer(
            [PREFIX + t for t in batch],
            add_special_tokens=True, truncation=False, return_length=True,
        )
        lengths.extend(int(x) for x in enc["length"])
    stats["new_chunk_token_lengths"] = lengths

    cpp = np.asarray(chunks_per_parent)
    tok = np.asarray(lengths)
    summary = {
        "overlength_parents": stats["overlength_parents"],
        "whole_passages": stats["whole_passages"],
        "new_chunks": stats["new_chunks"],
        "net_added_units": stats["net_added"],
        "total_units": stats["total_units"],
        "overlength_with_one_chunk": stats["overlength_with_one_chunk"],
        "oversized_sentences": stats["oversized_sentences"],
        "no_overlap_boundaries": stats["no_overlap_boundaries"],
        "post_fix_splits": stats["post_fix_splits"],
        "overlength_by_source": stats["overlength_by_source"],
        "new_chunks_by_source": stats["new_chunks_by_source"],
        "chunks_per_parent": {
            "min": int(cpp.min()), "median": float(np.median(cpp)),
            "p95": float(np.percentile(cpp, 95)), "max": int(cpp.max()),
            "mean": round(float(cpp.mean()), 2),
            "histogram": {
                "2": int((cpp == 2).sum()), "3-4": int(((cpp >= 3) & (cpp <= 4)).sum()),
                "5-9": int(((cpp >= 5) & (cpp <= 9)).sum()),
                "10+": int((cpp >= 10).sum()),
            },
        },
        "new_chunk_tokens": {
            "min": int(tok.min()), "median": float(np.median(tok)),
            "p95": float(np.percentile(tok, 95)), "max": int(tok.max()),
            "mean": round(float(tok.mean()), 1),
            "share_le_400": round(float((tok <= MAX_TOKENS).mean()), 4),
            "share_le_256": round(float((tok <= 256).mean()), 4),
        },
        "peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3), 2
        ),
        "files": {
            "chunks": str(OUT_CHUNKS.relative_to(PROJECT_ROOT)),
            "overlength_parents": str(OUT_PARENTS.relative_to(PROJECT_ROOT)),
        },
    }
    OUT_STATS.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== chunking summary ===")
    for key in [
        "overlength_parents", "whole_passages", "new_chunks", "net_added_units",
        "total_units", "overlength_with_one_chunk", "oversized_sentences",
        "no_overlap_boundaries", "post_fix_splits", "overlength_by_source",
        "new_chunks_by_source", "chunks_per_parent", "new_chunk_tokens",
        "peak_rss_gib",
    ]:
        print(f"{key}: {summary[key]}")
    print(f"\n[ok] wrote {OUT_STATS.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
