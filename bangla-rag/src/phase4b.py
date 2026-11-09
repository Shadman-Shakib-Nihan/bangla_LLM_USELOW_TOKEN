"""Statistical helpers for Phase 4b reporting.

The functions here operate on the frozen per-query Phase 4 outputs. They do not
read embeddings, rebuild retrievers, or call external services.
"""

from __future__ import annotations

import statistics
from typing import Iterable, Mapping

import numpy as np

DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Question-level bootstrap mean and percentile 95% CI."""
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
    }


def paired_bootstrap_diff(
    values_a: Mapping[str, float],
    values_b: Mapping[str, float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    scale: float = 100.0,
) -> dict:
    """Paired bootstrap mean difference for the same question IDs.

    Both mappings must contain exactly the same question IDs. A single set of
    bootstrap replicate indices is applied to both methods. The result is
    scaled by ``scale`` (100 gives percentage points for 0/1 metrics).
    """
    keys_a = set(values_a)
    keys_b = set(values_b)
    if keys_a != keys_b:
        missing_a = sorted(keys_b - keys_a)[:5]
        missing_b = sorted(keys_a - keys_b)[:5]
        raise ValueError(
            "paired comparison requires identical question IDs; "
            f"missing from A={missing_a}, missing from B={missing_b}"
        )
    qids = sorted(keys_a)
    if not qids:
        raise ValueError("paired comparison has no questions")
    a = np.asarray([values_a[q] for q in qids], dtype=float)
    b = np.asarray([values_b[q] for q in qids], dtype=float)
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    includes_zero = bool(lo <= 0.0 <= hi)
    return {
        "n_questions": int(n),
        "mean_diff": float((a.mean() - b.mean()) * scale),
        "ci_low": float(lo * scale),
        "ci_high": float(hi * scale),
        "includes_zero": includes_zero,
        "significant": not includes_zero,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def metric_values(
    records: Iterable[dict],
    *,
    dataset: str,
    retriever: str,
    method: str,
    metric: str,
    budget: int | None = None,
    require_gold_in_top10: bool = False,
) -> dict[str, float]:
    """Extract one metric per question for a method/setting.

    ``budget=None`` selects the M0 rows, whose ``budget_tokens`` is null.
    """
    out: dict[str, float] = {}
    for record in records:
        if record["dataset"] != dataset:
            continue
        if record["retriever"] != retriever:
            continue
        if record["method"] != method:
            continue
        rec_budget = record.get("budget_tokens")
        if budget is None:
            if rec_budget is not None:
                continue
        elif rec_budget != budget:
            continue
        if require_gold_in_top10 and not record.get("gold_in_top10"):
            continue
        value = record.get(metric)
        if value is None:
            continue
        out[record["question_id"]] = float(value)
    return out


def summarize_values(values: Iterable[float]) -> dict:
    """Mean, median, minimum, and maximum for a token distribution."""
    arr = [float(v) for v in values]
    if not arr:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(arr),
        "mean": float(statistics.fmean(arr)),
        "median": float(statistics.median(arr)),
        "min": float(min(arr)),
        "max": float(max(arr)),
    }


def summarize_m4(
    rows: Iterable[dict],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Summarize recomputed M4 oracle rows.

    Each row must provide ``o200k_tokens``, ``e5_tokens``, and ``n_selected``.
    """
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {
            "n_questions": 0,
            "o200k": summarize_values([]),
            "e5": summarize_values([]),
            "o200k_mean_ci": bootstrap_mean_ci([], n_boot=n_boot, seed=seed),
            "e5_mean_ci": bootstrap_mean_ci([], n_boot=n_boot, seed=seed),
            "mean_selected_sentences": None,
            "zero_match_count": 0,
            "zero_match_share": None,
            "multi_match_count": 0,
            "multi_match_share": None,
        }
    o200k = [r["o200k_tokens"] for r in rows]
    e5 = [r["e5_tokens"] for r in rows]
    selected = [r["n_selected"] for r in rows]
    zero = sum(1 for value in selected if value == 0)
    multi = sum(1 for value in selected if value > 1)
    return {
        "n_questions": n,
        "o200k": summarize_values(o200k),
        "e5": summarize_values(e5),
        "o200k_mean_ci": bootstrap_mean_ci(o200k, n_boot=n_boot, seed=seed),
        "e5_mean_ci": bootstrap_mean_ci(e5, n_boot=n_boot, seed=seed),
        "mean_selected_sentences": float(statistics.fmean(selected)),
        "zero_match_count": zero,
        "zero_match_share": zero / n,
        "multi_match_count": multi,
        "multi_match_share": multi / n,
    }
