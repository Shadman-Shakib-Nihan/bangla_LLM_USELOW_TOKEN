"""Separate TyDi-train-tuned hybrid RRF weight sensitivity experiment.

This is NOT the main hybrid. The primary hybrid remains the MIRACL-train-tuned
setting (RRF k=5, BM25/dense weights 0.50/0.50). Here we tune only the RRF
weights on TyDi train and evaluate the selected weights on TyDi validation.
The RRF constant stays at the main setting's k=5 so this is a weight-only
sensitivity check.

No retrieval is rerun: BM25 and dense top-100 rankings come from the frozen
Phase 3 finalization artifact (``phase3_final_rankings_top100.jsonl``).

Outputs are deliberately separate from the main Phase 3 files:
  results/phase3_tydi_hybrid_sensitivity_grid.csv
  results/phase3_tydi_hybrid_sensitivity.csv
  results/phase3_tydi_hybrid_sensitivity.json
  results/phase3_tydi_hybrid_sensitivity.md

Usage:
    cd bangla-rag
    .venv/bin/python experiments/phase3_tydi_hybrid_sensitivity.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion import rrf_fuse  # noqa: E402
from src.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
RANKINGS = RESULTS / "phase3_final_rankings_top100.jsonl"
MAIN_SUMMARY = RESULTS / "phase3_dense_hybrid_summary.json"
QUESTIONS = {
    "tydi_train": PROJECT_ROOT / "data" / "processed" / "tydi_train_questions.jsonl",
    "tydi_validation": PROJECT_ROOT / "data" / "processed" / "tydi_validation_questions.jsonl",
}

LABEL = "TyDi-train-tuned hybrid sensitivity"
RRF_K = 5
WEIGHTS = [round(x / 20, 2) for x in range(21)]
CAP = 100
N_BOOT = 10_000
BOOT_SEED = 42


def load_questions(name: str) -> list[dict]:
    with QUESTIONS[name].open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_rankings(dataset: str) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    with RANKINGS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["dataset"] != dataset:
                continue
            out.setdefault(rec["system"], {})[rec["question_id"]] = rec["ranking"]
    return out


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = values.size
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(fused: dict[str, list[str]], rows: list[dict]) -> dict:
    gold = {r["question_id"]: set(r["gold_passage_ids"]) for r in rows}
    keys = ["recall@1", "recall@3", "recall@5", "recall@10",
            "mrr@100", "ndcg@10"]
    vals = {k: [] for k in keys}
    for r in rows:
        qid = r["question_id"]
        g = gold[qid]
        ranking = fused[qid][:CAP]
        for k in [1, 3, 5, 10]:
            vals[f"recall@{k}"].append(recall_at_k(ranking, g, k))
        vals["mrr@100"].append(mrr(ranking, g))
        vals["ndcg@10"].append(ndcg_at_k(ranking, g, 10))
    out = {"n": len(rows)}
    for key, v in vals.items():
        out[key] = float(np.mean(v))
    return out


def main() -> int:
    start = time.time()
    train_rows = load_questions("tydi_train")
    val_rows = load_questions("tydi_validation")
    train_rank = load_rankings("tydi_train")
    val_rank = load_rankings("tydi_validation")

    bm25_train, dense_train = train_rank["BM25"], train_rank["Dense (e5-small)"]
    bm25_val, dense_val = val_rank["BM25"], val_rank["Dense (e5-small)"]

    grid = []
    best = None
    for w in WEIGHTS:
        fused = {
            r["question_id"]: rrf_fuse(
                [bm25_train[r["question_id"]][:CAP],
                 dense_train[r["question_id"]][:CAP]],
                weights=[w, 1.0 - w], k=RRF_K,
            )
            for r in train_rows
        }
        metrics = evaluate(fused, train_rows)
        row = {"label": LABEL, "rrf_k": RRF_K, "w_bm25": w,
               "w_dense": round(1.0 - w, 2), "n_train": len(train_rows), **metrics}
        grid.append(row)
        key = (metrics["ndcg@10"], metrics["mrr@100"], metrics["recall@10"])
        if best is None or key > best[0]:
            best = (key, row)

    selected = best[1]
    w = selected["w_bm25"]

    fused_val = {
        r["question_id"]: rrf_fuse(
            [bm25_val[r["question_id"]][:CAP],
             dense_val[r["question_id"]][:CAP]],
            weights=[w, 1.0 - w], k=RRF_K,
        )
        for r in val_rows
    }
    val_metrics = evaluate(fused_val, val_rows)

    # Per-question values for bootstrap CIs on the selected validation result.
    gold_val = {r["question_id"]: set(r["gold_passage_ids"]) for r in val_rows}
    per_q = {k: [] for k in ["recall@1", "recall@3", "recall@5", "recall@10",
                             "mrr@100", "ndcg@10"]}
    for r in val_rows:
        qid = r["question_id"]
        ranking = fused_val[qid][:CAP]
        g = gold_val[qid]
        for k in [1, 3, 5, 10]:
            per_q[f"recall@{k}"].append(recall_at_k(ranking, g, k))
        per_q["mrr@100"].append(mrr(ranking, g))
        per_q["ndcg@10"].append(ndcg_at_k(ranking, g, 10))

    val_row = {
        "label": LABEL,
        "rrf_k": RRF_K,
        "w_bm25": w,
        "w_dense": round(1.0 - w, 2),
        "tuned_on": "tydi_train",
        "evaluated_on": "tydi_validation",
        "n": len(val_rows),
        **{k: float(np.mean(v)) for k, v in per_q.items()},
    }
    for k, v in per_q.items():
        lo, hi = bootstrap_ci(np.asarray(v, dtype=float))
        val_row[f"{k}_ci_low"] = lo
        val_row[f"{k}_ci_high"] = hi

    # Main (MIRACL-tuned) hybrid validation reference, read from Phase 3.
    main_payload = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    main_hybrid = main_payload["hybrid"]
    main_val = next(
        r for r in main_payload["main_results"]
        if r["dataset"] == "tydi_validation" and r["system"] == "Hybrid (RRF)"
    )
    main_ref = {
        "label": "Main hybrid (MIRACL-train-tuned, primary; reference only)",
        "rrf_k": main_hybrid["rrf_k"],
        "w_bm25": main_hybrid["w_bm25"],
        "w_dense": main_hybrid["w_dense"],
        "tuned_on": main_hybrid["tuned_on"],
        "evaluated_on": "tydi_validation",
        "n": main_val["n"],
        "recall@1": main_val["recall@1"],
        "recall@3": main_val["recall@3"],
        "recall@5": main_val["recall@5"],
        "recall@10": main_val["recall@10"],
        "mrr@100": main_val["mrr"],
        "ndcg@10": main_val["ndcg@10"],
    }

    with (RESULTS / "phase3_tydi_hybrid_sensitivity_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        writer.writeheader()
        writer.writerows(grid)

    with (RESULTS / "phase3_tydi_hybrid_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(val_row.keys()), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerow(val_row)

    payload = {
        "label": LABEL,
        "description": (
            "Sensitivity only. RRF weights tuned on TyDi train, evaluated on "
            "TyDi validation, with rrf_k fixed at the main setting. This is "
            "NOT the primary hybrid and does not replace it."
        ),
        "primary_hybrid_unchanged": True,
        "primary_hybrid": {
            "rrf_k": main_hybrid["rrf_k"],
            "w_bm25": main_hybrid["w_bm25"],
            "w_dense": main_hybrid["w_dense"],
            "tuned_on": main_hybrid["tuned_on"],
        },
        "selection": {
            "tuned_on": "tydi_train",
            "n_train": len(train_rows),
            "grid": {"w_bm25": WEIGHTS, "rrf_k": [RRF_K]},
            "metric": "ndcg@10",
            "tie_break": ["mrr@100", "recall@10"],
            "selected": {"w_bm25": w, "w_dense": round(1.0 - w, 2),
                         "rrf_k": RRF_K},
        },
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED, "ci": 0.95,
                      "sampling_unit": "question"},
        "validation_result": val_row,
        "main_hybrid_validation_reference": main_ref,
        "runtime_seconds": round(time.time() - start, 1),
        "source_rankings": RANKINGS.name,
    }
    (RESULTS / "phase3_tydi_hybrid_sensitivity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        f"# {LABEL} (separate from the main hybrid)",
        "",
        "RRF weights tuned on **TyDi train only** with rrf_k fixed at 5; "
        "evaluated on **TyDi validation**. The primary hybrid remains the "
        "MIRACL-train-tuned setting.",
        "",
        "| setting | rrf_k | w_BM25 | w_dense | n | R@1 | R@3 | R@5 | R@10 | MRR@100 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    md.append(
        f"| {LABEL} | {RRF_K} | {w:.2f} | {1.0-w:.2f} | {val_row['n']} | "
        f"{val_row['recall@1']:.4f} | {val_row['recall@3']:.4f} | "
        f"{val_row['recall@5']:.4f} | {val_row['recall@10']:.4f} | "
        f"{val_row['mrr@100']:.4f} | {val_row['ndcg@10']:.4f} |"
    )
    md.append(
        f"| {main_ref['label']} | {main_ref['rrf_k']} | {main_ref['w_bm25']:.2f} | "
        f"{main_ref['w_dense']:.2f} | {main_ref['n']} | "
        f"{main_ref['recall@1']:.4f} | {main_ref['recall@3']:.4f} | "
        f"{main_ref['recall@5']:.4f} | {main_ref['recall@10']:.4f} | "
        f"{main_ref['mrr@100']:.4f} | {main_ref['ndcg@10']:.4f} |"
    )
    md += [
        "",
        f"Selected weights: **w_BM25={w:.2f}, w_dense={1.0-w:.2f}** "
        f"(train nDCG@10={selected['ndcg@10']:.4f}).",
        "",
        f"Bootstrap 95% CIs for the sensitivity row (10,000 question-level "
        f"resamples, seed {BOOT_SEED}):",
        "",
        "| metric | value | 95% CI |",
        "|---|---:|---|",
    ]
    for k in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@100",
              "ndcg@10"]:
        md.append(
            f"| {k} | {val_row[k]:.4f} | "
            f"[{val_row[k + '_ci_low']:.4f}, {val_row[k + '_ci_high']:.4f}] |"
        )
    md += ["", "No retrieval was rerun; rankings were reused from the frozen "
           "Phase 3 finalization artifact."]
    (RESULTS / "phase3_tydi_hybrid_sensitivity.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    print(f"[ok] selected w_BM25={w:.2f} w_dense={1.0-w:.2f} "
          f"(train nDCG@10={selected['ndcg@10']:.4f})")
    print(f"[ok] validation: R@10={val_row['recall@10']:.4f} "
          f"MRR@100={val_row['mrr@100']:.4f} "
          f"nDCG@10={val_row['ndcg@10']:.4f}")
    print(f"[done] {time.time()-start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
