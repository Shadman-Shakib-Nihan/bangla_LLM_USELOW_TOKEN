"""Generate Batch 2 Part C only (deterministic, no Gemini calls).

Resumable with the full run: writes the same Part C CSV/audit/exclusion files
that run_batch2_full_approved.py expects, so a later full-run resume skips the
already-accepted rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS))

import run_batch2_full_approved as base  # noqa: E402


def main() -> int:
    out = base.OUT
    (out / "audits").mkdir(parents=True, exist_ok=True)

    mismatches = base.verify_protected_checksums(
        base.PROJECT_ROOT, base.PROJECT_ROOT / "data/custom_batch2/protected_checksums.json"
    )
    if mismatches:
        raise SystemExit(f"PROTECTED_CHECKSUM_MISMATCH: {mismatches}")

    exclusions = base.load_exclusion_inventory(
        base.PROJECT_ROOT / "data/custom_batch2/source_inventory.json"
    )
    passages = base.load_passages(base.PROJECT_ROOT / "data/processed/corpus.jsonl")
    unit_parents = [
        row.get("parent_id", row["passage_id"])
        for row in (
            base.json.loads(line)
            for line in (base.PROJECT_ROOT / "data/processed/corpus_chunks.jsonl").open(encoding="utf-8")
        )
    ]
    passage_text = {p.passage_id: p.text for p in passages}

    bm25 = base.BM25Index.load(base.PROJECT_ROOT / "data/cache/bm25_index.pkl")
    from sentence_transformers import SentenceTransformer

    emb = np.load(base.PROJECT_ROOT / "data/embeddings/corpus_units.f16.npy", mmap_mode="r")
    encoder = SentenceTransformer(base.E5_MODEL, device="cpu")
    parent_index = base.build_parent_index(unit_parents)

    def bm25_audit(question, top_k=20):
        return base.bm25_search(bm25, parent_index, question, top_k=top_k)

    def dense_audit(question, top_k=20):
        qv = encoder.encode(
            [base.E5_QUERY_PREFIX + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)
        return base.dense_search(emb, qv, parent_index, top_k=top_k)

    part_c, ex_c, audits_c = base.run_part_c(exclusions, passage_text, bm25_audit, dense_audit)
    print(f"PART_C_DONE accepted={len(part_c)} excluded={len(ex_c.rows)} audits={len(audits_c)}", flush=True)

    mismatches_after = base.verify_protected_checksums(
        base.PROJECT_ROOT, base.PROJECT_ROOT / "data/custom_batch2/protected_checksums.json"
    )
    if mismatches_after:
        raise SystemExit("PROTECTED FILES CHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
