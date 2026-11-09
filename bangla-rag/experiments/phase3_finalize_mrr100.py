"""Phase 3 finalization: freeze manifest + metric-only MRR@100 recomputation.

Phase 3 is frozen at tag ``phase3-complete`` (commit 1c37a3c, the Phase 3
implementation commit). This script:

1. records the tag/commit and checksums of the approved Phase 3 result files,
   plus the (preserved) uncommitted working-tree state;
2. replays the deterministic Phase 3 rankings from cached artifacts only -- no
   text is re-embedded, no retriever is retrained, and no retriever parameter
   changes;
3. caps every ranking at 100 candidates and recomputes MRR@100.

The ranking replay is necessary because Phase 3 persisted only the top-20 for
dense/hybrid in ``phase3_per_query.jsonl``. The replay is validated against
both that file and the saved BM25 top-100 rankings before any metric is
reported.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/phase3_finalize_mrr100.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import mrr  # noqa: E402
from src.phase3_artifacts import (  # noqa: E402
    BM25_RANKINGS,
    PER_QUERY,
    RESULTS,
    bm25_rankings,
    build_dense_index,
    build_units_and_parent_index,
    dense_rankings,
    hybrid_rankings,
    load_bm25,
    load_phase3_per_query,
    load_questions,
    load_query_embeddings,
    load_saved_bm25_rankings,
    main_hybrid_params,
)

BM25_CONFIG = RESULTS / "phase3_bm25_config.json"

TAG = "phase3-complete"
DATASETS = ["miracl_dev", "tydi_validation", "tydi_train"]
REPORT_DATASETS = ["miracl_dev", "tydi_validation"]
SYSTEMS = ["BM25", "Dense (e5-small)", "Hybrid (RRF)"]
N_BOOT = 10_000
BOOT_SEED = 42
CAP = 100

PHASE3_FILES = [
    "phase3_dense_metrics.csv",
    "phase3_dense_hybrid_summary.json",
    "phase3_bm25_metrics.csv",
    "phase3_bm25_config.json",
    "phase3_per_query.jsonl",
    "phase3_bm25_rankings.jsonl",
    "phase3_hybrid_tuning.csv",
    "phase3_significance.csv",
    "phase3_tydi_answerstring.csv",
    "phase3_report.md",
    "phase3_error_analysis.md",
]


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT.parent, text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = BOOT_SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def freeze_manifest() -> dict:
    tag_commit = git("rev-parse", f"{TAG}^{{}}")
    tag_object = git("rev-parse", TAG)
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    dirty = [line for line in status.splitlines() if line.strip()]
    manifest = {
        "tag": TAG,
        "tag_object": tag_object,
        "tag_commit": tag_commit,
        "head_at_finalization": head,
        "phase3_commit_message": git("log", "-1", "--format=%s", tag_commit),
        "phase3_commit_date": git("log", "-1", "--format=%cI", tag_commit),
        "working_tree_dirty": bool(dirty),
        "working_tree_changes_preserved": len(dirty),
        "working_tree_changed_paths": dirty,
        "note": (
            "Tag points to the Phase 3 implementation commit. Later commits only "
            "add custom-draft review data. Uncommitted changes were preserved, "
            "not discarded; they are not part of the frozen Phase 3 results."
        ),
        "phase3_result_checksums_sha256": {
            name: sha256(RESULTS / name)
            for name in PHASE3_FILES
            if (RESULTS / name).exists()
        },
    }
    (RESULTS / "phase3_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def verify_against_phase3(
    name: str,
    rows: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    saved_bm25: dict[str, list[str]],
    per_query: dict[tuple[str, str, str], dict],
) -> dict:
    """Fail loudly if a replay does not match the frozen Phase 3 outputs."""
    checks: dict[str, object] = {"dataset": name}
    qids = [r["question_id"] for r in rows]

    bm25_mismatch = 0
    for qid in qids:
        if qid in saved_bm25:
            if rankings["BM25"][qid][:CAP] != saved_bm25[qid][:CAP]:
                bm25_mismatch += 1
    checks["bm25_saved_top100_mismatches"] = bm25_mismatch
    checks["bm25_saved_checks"] = sum(1 for q in qids if q in saved_bm25)

    pq_checks = 0
    pq_mismatch = 0
    for system in SYSTEMS:
        for qid in qids:
            key = (name, system, qid)
            if key not in per_query:
                continue
            pq_checks += 1
            expected = per_query[key]["top20"]
            got = rankings[system][qid][:20]
            if got != expected:
                pq_mismatch += 1
    checks["phase3_top20_checks"] = pq_checks
    checks["phase3_top20_mismatches"] = pq_mismatch

    if bm25_mismatch or pq_mismatch:
        raise RuntimeError(
            f"ranking replay does not match frozen Phase 3 outputs: {checks}"
        )
    return checks


def main() -> int:
    start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("[freeze] recording tag, commit, checksums, working-tree state")
    manifest = freeze_manifest()
    print(f"[freeze] {TAG} -> {manifest['tag_commit'][:12]} "
          f"(dirty files preserved: {manifest['working_tree_changes_preserved']})")

    print("[load] cached Phase 3 artifacts")
    recs, texts, parents, pidx = build_units_and_parent_index()
    bm25 = load_bm25(texts)
    dense_index, n_units = build_dense_index()
    hybrid = main_hybrid_params()
    bm25_cfg = json.loads(BM25_CONFIG.read_text(encoding="utf-8"))["best"]
    saved_bm25 = load_saved_bm25_rankings()
    per_query = load_phase3_per_query()
    print(f"[load] units={len(texts):,} parents={pidx.n_parents:,} "
          f"dense_units={n_units:,}")
    print(f"[load] main hybrid: rrf_k={hybrid['rrf_k']} "
          f"w_bm25={hybrid['w_bm25']} (tuned_on={hybrid['tuned_on']})")

    all_rankings: dict[str, dict[str, dict[str, list[str]]]] = {}
    verification = []
    ranking_records: list[dict] = []

    for name in DATASETS:
        rows = load_questions(name)
        qvec = load_query_embeddings(name, rows)
        t0 = time.time()
        bm25_rank = bm25_rankings(
            bm25, pidx, rows, bm25_cfg["k1"], bm25_cfg["b"], top_k=CAP
        )
        dense_rank = dense_rankings(
            dense_index, None, pidx, qvec, rows, top_k=CAP, full_search=True
        )
        hyb_rank = hybrid_rankings(
            bm25_rank, dense_rank, rows, hybrid["rrf_k"], hybrid["w_bm25"],
            fuse_depth=hybrid["fuse_depth"],
        )
        all_rankings[name] = {
            "BM25": bm25_rank,
            "Dense (e5-small)": dense_rank,
            "Hybrid (RRF)": hyb_rank,
        }
        check = verify_against_phase3(name, rows, all_rankings[name],
                                      saved_bm25, per_query)
        verification.append(check)
        print(f"[replay] {name:16s} n={len(rows):4d} "
              f"{time.time()-t0:5.1f}s  "
              f"top20_mismatch={check['phase3_top20_mismatches']} "
              f"bm25_top100_mismatch={check['bm25_saved_top100_mismatches']}")

        for system in SYSTEMS:
            for row in rows:
                qid = row["question_id"]
                ranking_records.append({
                    "dataset": name,
                    "system": system,
                    "question_id": qid,
                    "ranking": all_rankings[name][system][qid][:CAP],
                    "source": (
                        "saved phase3_bm25_rankings.jsonl"
                        if system == "BM25" and qid in saved_bm25
                        else "deterministic replay from cached Phase 3 artifacts"
                    ),
                })

    rankings_path = RESULTS / "phase3_final_rankings_top100.jsonl"
    with rankings_path.open("w", encoding="utf-8") as f:
        for rec in ranking_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] {rankings_path.name} ({len(ranking_records):,} records)")

    # ---- MRR@100: metric-only recomputation ------------------------------
    rows_out = []
    detail = {}
    for name in REPORT_DATASETS:
        rows = load_questions(name)
        gold = {r["question_id"]: set(r["gold_passage_ids"]) for r in rows}
        detail[name] = {}
        for system in SYSTEMS:
            vals_100 = []
            vals_10 = []
            for row in rows:
                qid = row["question_id"]
                ranking = all_rankings[name][system][qid][:CAP]
                vals_100.append(mrr(ranking, gold[qid]))
                vals_10.append(mrr(ranking[:10], gold[qid]))
            arr_100 = np.asarray(vals_100, dtype=float)
            arr_10 = np.asarray(vals_10, dtype=float)
            lo, hi = bootstrap_ci(arr_100)
            rec = {
                "dataset": name,
                "system": system,
                "n": int(arr_100.size),
                "mrr@100": float(arr_100.mean()),
                "mrr@100_ci_low": lo,
                "mrr@100_ci_high": hi,
                "mrr@10_reference": float(arr_10.mean()),
            }
            rows_out.append(rec)
            detail[name][system] = rec
            print(f"[mrr@100] {name:15s} {system:17s} n={rec['n']:4d} "
                  f"MRR@100={rec['mrr@100']:.4f} [{lo:.4f}, {hi:.4f}]")

    csv_path = RESULTS / "phase3_final_mrr100.csv"
    fields = list(rows_out[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    payload = {
        "description": (
            "Phase 3 final MRR@100. Metric-only recomputation from frozen "
            "retrieval artifacts. Every ranking is capped at 100 candidates "
            "before MRR is computed. No embeddings were recomputed and no "
            "retriever was retrained or reconfigured."
        ),
        "tag": TAG,
        "phase3_commit": manifest["tag_commit"],
        "cap": CAP,
        "bootstrap": {
            "n_resamples": N_BOOT,
            "seed": BOOT_SEED,
            "ci": 0.95,
            "sampling_unit": "question",
        },
        "ranking_replay": {
            "reason": (
                "Phase 3 persisted top-20 only for dense/hybrid; top-100 BM25 "
                "was persisted. Replaying from cached vectors/index was required "
                "to compute MRR@100 without re-embedding."
            ),
            "reembedding": False,
            "retriever_parameters_changed": False,
            "main_hybrid": hybrid,
            "validation_against_phase3": verification,
        },
        "results": rows_out,
    }
    json_path = RESULTS / "phase3_final_mrr100.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[save] {csv_path.name}, {json_path.name}")
    print(f"[done] {time.time()-start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
