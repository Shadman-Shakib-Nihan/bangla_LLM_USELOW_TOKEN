"""Stage 3 executable Batch 2 pipeline (local/mockable; no live Gemini calls).

This script never performs live API calls and never writes protected files.
Live generation remains blocked until GEMINI_API_KEY is set and Stage 4 is
explicitly authorized.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.batch2_generation import MockProvider, build_part_c_removed_evidence, generate_part_a, generate_part_b
from src.batch2_pipeline import (
    BATCH2_SEED,
    bm25_search,
    dense_search,
    load_exclusion_inventory,
    load_passages,
    pilot_selection,
    read_jsonl,
    sample_passages_for_part_a,
    sample_tydi_questions_for_part_c,
    verify_protected_checksums,
    write_part_csvs,
)
from src.batch2_reports import write_qc_artifacts
from src.batch2_text import E5_MODEL, E5_QUERY_PREFIX, build_rare_terms
from src.retrieval import BM25Index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "mock-pilot"], default="check")
    ap.add_argument("--inventory", default="data/custom_batch2/source_inventory.json")
    ap.add_argument("--checksums", default="data/custom_batch2/protected_checksums.json")
    ap.add_argument("--out-dir", default="data/custom_batch2/mock_pilot")
    ap.add_argument("--seed", type=int, default=BATCH2_SEED)
    args = ap.parse_args()

    mismatches = verify_protected_checksums(PROJECT_ROOT, args.checksums)
    print(json.dumps({"protected_checksum_mismatches": mismatches}, ensure_ascii=False))
    if mismatches:
        return 2
    if args.mode == "check":
        print("Stage 3 local checks passed; live Stage 4 remains blocked without GEMINI_API_KEY + explicit authorization.")
        return 0

    exclusions = load_exclusion_inventory(args.inventory)
    passages = load_passages(PROJECT_ROOT / "data/processed/corpus.jsonl")
    units = read_jsonl(PROJECT_ROOT / "data/processed/corpus_chunks.jsonl")
    unit_texts = [u["text"] for u in units]
    unit_parents = [u.get("parent_id", u["passage_id"]) for u in units]
    passage_text = {p.passage_id: p.text for p in passages}
    corpus_pairs = [(p.passage_id, p.text) for p in passages if p.source == "miracl"]
    rare_terms = build_rare_terms([p.text for p in passages[:5000]])

    bm25 = BM25Index.load(PROJECT_ROOT / "data/cache/bm25_index.pkl")

    # Dense audits require local model/files; mock-pilot can run BM25-only unless artifacts/model available.
    dense_available = (PROJECT_ROOT / "data/embeddings/corpus_units.f16.npy").exists()
    unit_emb = np.load(PROJECT_ROOT / "data/embeddings/corpus_units.f16.npy").astype(np.float32) if dense_available else None

    def bm25_audit(query: str, top_k: int = 20):
        return bm25_search(bm25, unit_parents, query, top_k=top_k)

    def dense_audit(query: str, top_k: int = 20):
        if not dense_available:
            return []
        # placeholder zero vector means "audit unavailable" in mock mode unless encoder wired; keep explicit.
        qv = np.zeros((unit_emb.shape[1],), dtype=np.float32)
        return dense_search(unit_emb, qv, unit_parents, top_k=top_k)

    # Deterministic tiny mock pilot candidates (no API, no E5 counting requirement).
    candidates = [p for p in passages if p.source == "miracl" and not p.is_short and len(p.text) > 80][:20]
    part_a_candidates = pilot_selection(candidates, candidates)["answerable"]
    provider_a = MockProvider([{
        "question": f"{c.title or 'এই বিষয়'} সম্পর্কে পাঠ্যে কী বলা আছে?",
        "gold_answer": c.text.split()[0],
        "question_type": "what",
    } for c in part_a_candidates])
    part_a, ex_a = generate_part_a(part_a_candidates, provider=provider_a, corpus=corpus_pairs, rare_terms=rare_terms, bm25_audit=bm25_audit, dense_audit=None if not dense_available else dense_audit)
    provider_b = MockProvider([{"question": "এই বিষয়ে কোন অনুপস্থিত তথ্য?", "question_type": "what", "why_unanswerable": "passage lacks detail", "near_miss_fact": ""}] * 5)
    part_b, ex_b, audits_b = generate_part_b(part_a_candidates[:5], provider=provider_b, passage_text=passage_text, bm25_audit=bm25_audit, dense_audit=dense_audit)
    tydi = read_jsonl(PROJECT_ROOT / "data/processed/tydi_train_questions.jsonl")
    part_c_src = sample_tydi_questions_for_part_c(tydi, exclusions, target=5, seed=args.seed)
    part_c, ex_c, audits_c = build_part_c_removed_evidence(part_c_src, passage_text=passage_text)
    out = PROJECT_ROOT / args.out_dir
    write_part_csvs(out, part_a=part_a, part_b=part_b, part_c=part_c)
    write_qc_artifacts(out / "qc", part_a=part_a, part_b=part_b, part_c=part_c, exclusions=ex_a.rows + ex_b.rows + ex_c.rows, audits={"part_b": audits_b, "part_c": audits_c})
    print(json.dumps({"out_dir": str(out), "counts": {"A": len(part_a), "B": len(part_b), "C": len(part_c)}, "api_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
