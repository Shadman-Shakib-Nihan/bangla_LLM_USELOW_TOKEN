#!/usr/bin/env python3
"""Phase 4b: statistical reporting, oracle validation, and finalization.

This script is read-only with respect to retrieval and embeddings. It reads the
frozen Phase 4 per-query outputs and frozen Phase 3 rankings, then writes:

  phase4b_tydi_m0.csv
  phase4b_tydi_budget.csv
  phase4b_miracl_m0.csv
  phase4b_miracl_budget.csv
  phase4b_paired_bootstrap.csv
  phase4b_m4_validation.csv / .json / _examples.md
  phase4b_recall_vs_tokens.png
  phase4b_miracl_retention_vs_tokens.png
  phase4b_report.md
  phase4b_config.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer  # noqa: E402

from src.compression import (  # noqa: E402
    build_candidates,
    contains_answer,
    m4_oracle,
    normalize_for_match,
)
from src.phase4b import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_mean_ci,
    metric_values,
    paired_bootstrap_diff,
    summarize_m4,
)

RESULTS = PROJECT_ROOT / "results"
PROCESSED = PROJECT_ROOT / "data" / "processed"
CORPUS = PROCESSED / "corpus.jsonl"
RANKINGS = RESULTS / "phase3_final_rankings_top100.jsonl"
PER_QUERY = RESULTS / "phase4_per_query_metrics.jsonl"
EXISTING = RESULTS / "phase4_results_all.csv"
PHASE4_CONFIG = RESULTS / "phase4_config.json"
M2_EXAMPLES = RESULTS / "phase4_m2_answer_loss_examples.json"
FREEZE_MANIFEST = RESULTS / "phase3_freeze_manifest.json"

MODEL_NAME = "intfloat/multilingual-e5-small"
SYSTEMS = ["Dense (e5-small)", "Hybrid (RRF)"]
SYSTEM_SHORT = {"Dense (e5-small)": "Dense", "Hybrid (RRF)": "Hybrid"}
TYDI_DATASETS = ["tydi_heldout", "tydi_validation"]
MIRACL_DATASET = "miracl_dev"
BUDGETS = [128, 256, 512, 1024]
M0_KS = [1, 3, 5, 10]
METHODS = ["M1", "M2", "M3", "M5"]
COMPARISONS = [("M2", "M1"), ("M2", "M5"), ("M3", "M2")]
METHOD_ORDER = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M5": 5}
N_BOOT = DEFAULT_BOOTSTRAP_REPLICATES
BOOT_SEED = DEFAULT_BOOTSTRAP_SEED
TYDI_SPLIT_SEED = 42
TYDI_TUNING_N = 1000
START = time.time()


class O200k:
    def __init__(self) -> None:
        import tiktoken

        os.environ.setdefault(
            "TIKTOKEN_CACHE_DIR", str(PROJECT_ROOT / "data" / "cache" / "tiktoken")
        )
        self.enc = tiktoken.get_encoding("o200k_base")
        self.version = __import__("importlib.metadata", fromlist=["version"]).version(
            "tiktoken"
        )

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))


class E5Counter:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(
            self.tokenizer.encode(text, add_special_tokens=True, verbose=False)
        )


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_questions(name: str) -> list[dict]:
    return load_jsonl(PROCESSED / f"{name}_questions.jsonl")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_ci(mean, low, high, digits: int) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def build_aggregates(records: list[dict]) -> list[dict]:
    buckets: dict = defaultdict(
        lambda: {
            "o200k": [],
            "e5": [],
            "recall": [],
            "gold": [],
            "loss": [],
        }
    )
    for record in records:
        method = record["method"]
        if method not in METHOD_ORDER or method == "M4":
            continue
        key = (
            record["dataset"],
            record["retriever"],
            method,
            record["setting"],
            record["budget_tokens"],
            record["top_k"],
        )
        bucket = buckets[key]
        bucket["o200k"].append(record["o200k_tokens"])
        bucket["e5"].append(record["e5_tokens"])
        if record["answer_evidence_recall"] is not None:
            bucket["recall"].append(record["answer_evidence_recall"])
        if record["gold_in_top10"]:
            bucket["gold"].append(record["gold_passage_retained"])
        if record["evidence_loss"] is not None:
            bucket["loss"].append(record["evidence_loss"])

    rows: list[dict] = []
    for key, bucket in sorted(
        buckets.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            METHOD_ORDER[item[0][2]],
            item[0][4] if item[0][4] is not None else 0,
            item[0][5] if item[0][5] is not None else 0,
        ),
    ):
        dataset, retriever, method, setting, budget, top_k = key
        o = bootstrap_mean_ci(bucket["o200k"], n_boot=N_BOOT, seed=BOOT_SEED)
        e = bootstrap_mean_ci(bucket["e5"], n_boot=N_BOOT, seed=BOOT_SEED)
        recall = (
            bootstrap_mean_ci(bucket["recall"], n_boot=N_BOOT, seed=BOOT_SEED)
            if bucket["recall"]
            else None
        )
        gold = (
            bootstrap_mean_ci(bucket["gold"], n_boot=N_BOOT, seed=BOOT_SEED)
            if bucket["gold"]
            else None
        )
        loss = (
            bootstrap_mean_ci(bucket["loss"], n_boot=N_BOOT, seed=BOOT_SEED)
            if bucket["loss"]
            else None
        )
        rows.append({
            "dataset": dataset,
            "dataset_kind": "miracl" if dataset == MIRACL_DATASET else "tydi",
            "retriever": retriever,
            "method": method,
            "setting": setting,
            "budget_tokens": budget,
            "top_k": top_k,
            "n_questions": o["n"],
            "n_answer_denominator": recall["n"] if recall else None,
            "n_gold_denominator": gold["n"] if gold else None,
            "n_loss_denominator": loss["n"] if loss else None,
            "mean_o200k_tokens": o["mean"],
            "o200k_ci_low": o["ci_low"],
            "o200k_ci_high": o["ci_high"],
            "mean_e5_tokens": e["mean"],
            "e5_ci_low": e["ci_low"],
            "e5_ci_high": e["ci_high"],
            "answer_evidence_recall": recall["mean"] if recall else None,
            "recall_ci_low": recall["ci_low"] if recall else None,
            "recall_ci_high": recall["ci_high"] if recall else None,
            "gold_passage_retention": gold["mean"] if gold else None,
            "gold_ci_low": gold["ci_low"] if gold else None,
            "gold_ci_high": gold["ci_high"] if gold else None,
            "evidence_loss_rate": loss["mean"] if loss else None,
            "loss_ci_low": loss["ci_low"] if loss else None,
            "loss_ci_high": loss["ci_high"] if loss else None,
        })
    return rows


def check_against_phase4(aggregates: list[dict]) -> dict:
    with EXISTING.open(encoding="utf-8") as stream:
        existing = list(csv.DictReader(stream))
    keyed = {
        (r["dataset"], r["retriever"], r["method"], r["setting"]): r
        for r in existing
        if r["method"] != "M4"
    }
    compared = 0
    mismatches = []
    for row in aggregates:
        key = (
            row["dataset"], row["retriever"], row["method"], row["setting"]
        )
        old = keyed.get(key)
        if old is None:
            mismatches.append((key, "missing in phase4_results_all.csv"))
            continue
        for field, source in [
            ("mean_o200k_tokens", "mean_o200k_tokens"),
            ("mean_e5_tokens", "mean_e5_tokens"),
            ("answer_evidence_recall", "answer_evidence_recall"),
            ("gold_passage_retention", "gold_passage_retention"),
            ("evidence_loss_rate", "evidence_loss_rate"),
        ]:
            new_value = row[field]
            old_value = old[source]
            if new_value is None and old_value in ("", "None"):
                continue
            if new_value is None or old_value in ("", "None"):
                mismatches.append((key, f"{field} missing one side"))
                continue
            if abs(float(new_value) - float(old_value)) > 1e-9:
                mismatches.append((key, f"{field}: {new_value} != {old_value}"))
        compared += 1
    return {"compared": compared, "mismatches": mismatches}


def build_paired_rows(
    records: list[dict],
    datasets: list[str],
    metric: str,
    require_gold: bool,
    metric_label: str,
) -> list[dict]:
    out: list[dict] = []
    for dataset in datasets:
        for retriever in SYSTEMS:
            for budget in BUDGETS:
                values = {
                    method: metric_values(
                        records,
                        dataset=dataset,
                        retriever=retriever,
                        method=method,
                        metric=metric,
                        budget=budget,
                        require_gold_in_top10=require_gold,
                    )
                    for method in ["M1", "M2", "M3", "M5"]
                }
                for first, second in COMPARISONS:
                    result = paired_bootstrap_diff(
                        values[first],
                        values[second],
                        n_boot=N_BOOT,
                        seed=BOOT_SEED,
                    )
                    comparison = f"{first}-{second}"
                    if result["significant"]:
                        direction = "higher" if result["mean_diff"] > 0 else "lower"
                        interpretation = (
                            f"{first} is {abs(result['mean_diff']):.2f} pp "
                            f"{direction} than {second}; the 95% CI "
                            f"[{result['ci_low']:.2f}, {result['ci_high']:.2f}] "
                            "excludes zero, so the paired difference is "
                            "statistically significant at the 5% level."
                        )
                    else:
                        interpretation = (
                            f"The paired difference {comparison} is "
                            f"{result['mean_diff']:+.2f} pp with 95% CI "
                            f"[{result['ci_low']:.2f}, {result['ci_high']:.2f}], "
                            "which includes zero; it is not statistically "
                            "significant at the 5% level."
                        )
                    out.append({
                        "dataset": dataset,
                        "retriever": retriever,
                        "metric": metric_label,
                        "comparison": comparison,
                        "budget_tokens": budget,
                        "n_questions": result["n_questions"],
                        "mean_diff_pp": result["mean_diff"],
                        "ci_low_pp": result["ci_low"],
                        "ci_high_pp": result["ci_high"],
                        "includes_zero": result["includes_zero"],
                        "significant": result["significant"],
                        "n_boot": result["n_boot"],
                        "bootstrap_seed": result["seed"],
                        "interpretation": interpretation,
                    })
    return out


def load_rankings() -> dict:
    out: dict = defaultdict(lambda: defaultdict(dict))
    allowed = {"tydi_train", "tydi_validation"}
    with RANKINGS.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["system"] not in SYSTEMS:
                continue
            if record["dataset"] not in allowed:
                continue
            out[record["dataset"]][record["system"]][record["question_id"]] = (
                record["ranking"][:10]
            )
    return out


def load_parent_texts(needed: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    with CORPUS.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            pid = record["parent_id"]
            if pid in needed:
                texts[pid] = record["text"]
    missing = needed - set(texts)
    if missing:
        raise RuntimeError(f"missing parent texts for {len(missing)} IDs")
    return texts


def tydi_split() -> tuple[list[dict], list[dict], list[dict]]:
    train = load_questions("tydi_train")
    validation = load_questions("tydi_validation")
    indices = list(range(len(train)))
    rng = random.Random(TYDI_SPLIT_SEED)
    tuning = set(rng.sample(indices, TYDI_TUNING_N))
    heldout = [row for i, row in enumerate(train) if i not in tuning]
    return train, heldout, validation


def compute_m4_rows(
    rankings: dict,
    parent_texts: dict[str, str],
    heldout: list[dict],
    validation: list[dict],
    o200k: O200k,
    e5: E5Counter,
) -> list[dict]:
    sources = {
        "tydi_heldout": ("tydi_train", heldout),
        "tydi_validation": ("tydi_validation", validation),
    }
    rows: list[dict] = []
    for dataset, (source, questions) in sources.items():
        for retriever in SYSTEMS:
            for question in questions:
                qid = question["question_id"]
                top10 = rankings[source][retriever].get(qid, [])[:10]
                passages = [parent_texts[pid] for pid in top10]
                candidates = build_candidates(passages)
                answers = [
                    answer for answer in question.get("gold_answers", [])
                    if normalize_for_match(answer)
                ]
                result = m4_oracle(candidates, answers)
                context = result.context
                has_answer = (
                    contains_answer(context, answers)[0] if answers else False
                )
                rows.append({
                    "dataset": dataset,
                    "retriever": retriever,
                    "question_id": qid,
                    "question": question["question"],
                    "gold_answers": answers,
                    "top10_ids": top10,
                    "o200k_tokens": o200k.count(context),
                    "e5_tokens": e5.count(context),
                    "n_selected": len(result.selected),
                    "selected_sentences": [c.text for c in result.selected],
                    "selected_passage_ranks": [
                        c.passage_rank + 1 for c in result.selected
                    ],
                    "answer_evidence_recall": 1.0 if has_answer else 0.0,
                    "context": context,
                    "fallback": result.fallback,
                })
    return rows


def build_m4_stats(m4_rows: list[dict]) -> list[dict]:
    existing = {
        (r["dataset"], r["retriever"]): r
        for r in csv.DictReader(EXISTING.open(encoding="utf-8"))
        if r["method"] == "M4"
    }
    groups: dict = defaultdict(list)
    for row in m4_rows:
        groups[(row["dataset"], row["retriever"])].append(row)
    stats = []
    for (dataset, retriever), rows in sorted(groups.items()):
        summary = summarize_m4(rows, n_boot=N_BOOT, seed=BOOT_SEED)
        recall = bootstrap_mean_ci(
            [r["answer_evidence_recall"] for r in rows],
            n_boot=N_BOOT,
            seed=BOOT_SEED,
        )
        previous = existing.get((dataset, retriever))
        previous_o = float(previous["mean_o200k_tokens"]) if previous else None
        previous_e = float(previous["mean_e5_tokens"]) if previous else None
        with_answers = sum(1 for r in rows if r["gold_answers"])
        stats.append({
            "dataset": dataset,
            "retriever": retriever,
            "n_questions": summary["n_questions"],
            "n_with_gold_answers": with_answers,
            "o200k_mean": summary["o200k"]["mean"],
            "o200k_median": summary["o200k"]["median"],
            "o200k_min": summary["o200k"]["min"],
            "o200k_max": summary["o200k"]["max"],
            "o200k_ci_low": summary["o200k_mean_ci"]["ci_low"],
            "o200k_ci_high": summary["o200k_mean_ci"]["ci_high"],
            "e5_mean": summary["e5"]["mean"],
            "e5_median": summary["e5"]["median"],
            "e5_min": summary["e5"]["min"],
            "e5_max": summary["e5"]["max"],
            "e5_ci_low": summary["e5_mean_ci"]["ci_low"],
            "e5_ci_high": summary["e5_mean_ci"]["ci_high"],
            "mean_selected_sentences": summary["mean_selected_sentences"],
            "zero_match_count": summary["zero_match_count"],
            "zero_match_share": summary["zero_match_share"],
            "multi_match_count": summary["multi_match_count"],
            "multi_match_share": summary["multi_match_share"],
            "answer_recall": recall["mean"],
            "answer_recall_ci_low": recall["ci_low"],
            "answer_recall_ci_high": recall["ci_high"],
            "previous_mean_o200k": previous_o,
            "difference_o200k": (
                summary["o200k"]["mean"] - previous_o
                if previous_o is not None else None
            ),
            "previous_mean_e5": previous_e,
            "difference_e5": (
                summary["e5"]["mean"] - previous_e
                if previous_e is not None else None
            ),
        })
    return stats


def select_m4_examples(m4_rows: list[dict], limit: int = 5) -> list[dict]:
    grouped: dict = defaultdict(list)
    for row in m4_rows:
        if row["n_selected"] > 1:
            grouped[(row["dataset"], row["retriever"])].append(row)
    selected: list[dict] = []
    for key in sorted(grouped):
        for row in grouped[key][:2]:
            selected.append(row)
            if len(selected) == limit:
                return selected
    return selected


def plot_tydi(aggregates: list[dict], m4_stats: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "M1": "#ff7f0e",
        "M2": "#2ca02c",
        "M3": "#d62728",
        "M5": "#7f7f7f",
        "M0": "#111111",
        "M4": "#9467bd",
    }
    markers = {"M1": "s", "M2": "^", "M3": "D", "M5": "v"}
    labels = {
        "M1": "M1 hard budget",
        "M2": "M2 extractive e5",
        "M3": "M3 extractive BM25",
        "M5": "M5 random",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, retriever in zip(axes, SYSTEMS):
        m0 = [
            r for r in aggregates
            if r["dataset"] == "tydi_heldout"
            and r["retriever"] == retriever
            and r["method"] == "M0"
        ]
        m0.sort(key=lambda r: r["top_k"])
        x = [r["mean_o200k_tokens"] for r in m0]
        y = [r["answer_evidence_recall"] for r in m0]
        xerr = [
            [r["mean_o200k_tokens"] - r["o200k_ci_low"] for r in m0],
            [r["o200k_ci_high"] - r["mean_o200k_tokens"] for r in m0],
        ]
        yerr = [
            [r["answer_evidence_recall"] - r["recall_ci_low"] for r in m0],
            [r["recall_ci_high"] - r["answer_evidence_recall"] for r in m0],
        ]
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt="D", color=colors["M0"],
            markersize=6, capsize=2, linestyle="none",
            label="M0 baseline (top-k, not connected)",
        )
        for row in m0:
            ax.annotate(
                f"k={row['top_k']}",
                (row["mean_o200k_tokens"], row["answer_evidence_recall"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
            )

        for method in ["M1", "M2", "M3", "M5"]:
            points = [
                r for r in aggregates
                if r["dataset"] == "tydi_heldout"
                and r["retriever"] == retriever
                and r["method"] == method
            ]
            points.sort(key=lambda r: r["budget_tokens"])
            px = [r["mean_o200k_tokens"] for r in points]
            py = [r["answer_evidence_recall"] for r in points]
            pxerr = [
                [r["mean_o200k_tokens"] - r["o200k_ci_low"] for r in points],
                [r["o200k_ci_high"] - r["mean_o200k_tokens"] for r in points],
            ]
            pyerr = [
                [r["answer_evidence_recall"] - r["recall_ci_low"] for r in points],
                [r["recall_ci_high"] - r["answer_evidence_recall"] for r in points],
            ]
            ax.errorbar(
                px, py, xerr=pxerr, yerr=pyerr,
                fmt=markers[method] + "-", color=colors[method],
                markersize=6, capsize=2, linewidth=1.6, label=labels[method],
            )

        oracle = [
            s for s in m4_stats
            if s["dataset"] == "tydi_heldout" and s["retriever"] == retriever
        ]
        if oracle:
            s = oracle[0]
            ax.errorbar(
                [s["o200k_mean"]], [s["answer_recall"]],
                xerr=[[s["o200k_mean"] - s["o200k_ci_low"]],
                      [s["o200k_ci_high"] - s["o200k_mean"]]],
                yerr=[[s["answer_recall"] - s["answer_recall_ci_low"]],
                      [s["answer_recall_ci_high"] - s["answer_recall"]]],
                fmt="*", color=colors["M4"], markersize=13, capsize=2,
                linestyle="none", label="M4 oracle (reference, not a method)",
            )
        ax.set_title(f"{SYSTEM_SHORT[retriever]} retrieval")
        ax.set_xlabel("Mean context tokens (o200k_base)")
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Answer-evidence recall (TyDi held-out)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Phase 4b: TyDi held-out answer-evidence recall vs context cost "
        "(error bars are 95% bootstrap CIs)"
    )
    fig.tight_layout()
    path = RESULTS / "phase4b_recall_vs_tokens.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_miracl(aggregates: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "M1": "#ff7f0e",
        "M2": "#2ca02c",
        "M3": "#d62728",
        "M5": "#7f7f7f",
        "M0": "#111111",
    }
    markers = {"M1": "s", "M2": "^", "M3": "D", "M5": "v"}
    labels = {
        "M1": "M1 hard budget",
        "M2": "M2 extractive e5",
        "M3": "M3 extractive BM25",
        "M5": "M5 random",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, retriever in zip(axes, SYSTEMS):
        m0 = [
            r for r in aggregates
            if r["dataset"] == MIRACL_DATASET
            and r["retriever"] == retriever
            and r["method"] == "M0"
        ]
        m0.sort(key=lambda r: r["top_k"])
        x = [r["mean_o200k_tokens"] for r in m0]
        y = [r["gold_passage_retention"] for r in m0]
        xerr = [
            [r["mean_o200k_tokens"] - r["o200k_ci_low"] for r in m0],
            [r["o200k_ci_high"] - r["mean_o200k_tokens"] for r in m0],
        ]
        yerr = [
            [r["gold_passage_retention"] - r["gold_ci_low"] for r in m0],
            [r["gold_ci_high"] - r["gold_passage_retention"] for r in m0],
        ]
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt="D", color=colors["M0"],
            markersize=6, capsize=2, linestyle="none",
            label="M0 baseline (top-k, not connected)",
        )
        for row in m0:
            ax.annotate(
                f"k={row['top_k']}",
                (row["mean_o200k_tokens"], row["gold_passage_retention"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
            )

        for method in ["M1", "M2", "M3", "M5"]:
            points = [
                r for r in aggregates
                if r["dataset"] == MIRACL_DATASET
                and r["retriever"] == retriever
                and r["method"] == method
            ]
            points.sort(key=lambda r: r["budget_tokens"])
            px = [r["mean_o200k_tokens"] for r in points]
            py = [r["gold_passage_retention"] for r in points]
            pxerr = [
                [r["mean_o200k_tokens"] - r["o200k_ci_low"] for r in points],
                [r["o200k_ci_high"] - r["mean_o200k_tokens"] for r in points],
            ]
            pyerr = [
                [r["gold_passage_retention"] - r["gold_ci_low"] for r in points],
                [r["gold_ci_high"] - r["gold_passage_retention"] for r in points],
            ]
            ax.errorbar(
                px, py, xerr=pxerr, yerr=pyerr,
                fmt=markers[method] + "-", color=colors[method],
                markersize=6, capsize=2, linewidth=1.6, label=labels[method],
            )
        ax.set_title(f"{SYSTEM_SHORT[retriever]} retrieval")
        ax.set_xlabel("Mean context tokens (o200k_base)")
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Gold-passage retention (MIRACL dev)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Phase 4b: MIRACL dev gold-passage retention vs context cost "
        "(not answer-evidence recall; error bars are 95% bootstrap CIs)"
    )
    fig.tight_layout()
    path = RESULTS / "phase4b_miracl_retention_vs_tokens.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def m0_rows_for(aggregates, dataset, retriever):
    rows = [
        r for r in aggregates
        if r["dataset"] == dataset and r["retriever"] == retriever
        and r["method"] == "M0"
    ]
    return sorted(rows, key=lambda r: r["top_k"])


def budget_rows_for(aggregates, dataset, retriever):
    rows = [
        r for r in aggregates
        if r["dataset"] == dataset and r["retriever"] == retriever
        and r["method"] in METHODS
    ]
    return sorted(
        rows,
        key=lambda r: (METHOD_ORDER[r["method"]], r["budget_tokens"]),
    )


def m0_md_table(rows, include_answer=True):
    headers = ["k", "n", "o200k tokens (95% CI)", "e5 tokens (95% CI)"]
    if include_answer:
        headers += ["Answer recall (95% CI)", "Gold retention (95% CI)",
                    "Evidence loss (95% CI)"]
    else:
        headers += ["Gold retention (95% CI)"]
    table_rows = []
    for r in rows:
        row = [
            r["top_k"],
            r["n_questions"],
            fmt_ci(r["mean_o200k_tokens"], r["o200k_ci_low"],
                   r["o200k_ci_high"], 1),
            fmt_ci(r["mean_e5_tokens"], r["e5_ci_low"], r["e5_ci_high"], 1),
        ]
        if include_answer:
            row += [
                fmt_ci(r["answer_evidence_recall"], r["recall_ci_low"],
                       r["recall_ci_high"], 4),
                fmt_ci(r["gold_passage_retention"], r["gold_ci_low"],
                       r["gold_ci_high"], 4),
                fmt_ci(r["evidence_loss_rate"], r["loss_ci_low"],
                       r["loss_ci_high"], 4),
            ]
        else:
            row += [
                fmt_ci(r["gold_passage_retention"], r["gold_ci_low"],
                       r["gold_ci_high"], 4),
            ]
        table_rows.append(row)
    return md_table(headers, table_rows)


def budget_md_table(rows, include_answer=True):
    headers = ["Method", "Budget", "n", "o200k tokens (95% CI)",
               "e5 tokens (95% CI)"]
    if include_answer:
        headers += ["Answer recall (95% CI)", "Gold retention (95% CI)",
                    "Evidence loss (95% CI)"]
    else:
        headers += ["Gold retention (95% CI)"]
    table_rows = []
    for r in rows:
        row = [
            r["method"], r["budget_tokens"], r["n_questions"],
            fmt_ci(r["mean_o200k_tokens"], r["o200k_ci_low"],
                   r["o200k_ci_high"], 1),
            fmt_ci(r["mean_e5_tokens"], r["e5_ci_low"], r["e5_ci_high"], 1),
        ]
        if include_answer:
            row += [
                fmt_ci(r["answer_evidence_recall"], r["recall_ci_low"],
                       r["recall_ci_high"], 4),
                fmt_ci(r["gold_passage_retention"], r["gold_ci_low"],
                       r["gold_ci_high"], 4),
                fmt_ci(r["evidence_loss_rate"], r["loss_ci_low"],
                       r["loss_ci_high"], 4),
            ]
        else:
            row += [
                fmt_ci(r["gold_passage_retention"], r["gold_ci_low"],
                       r["gold_ci_high"], 4),
            ]
        table_rows.append(row)
    return md_table(headers, table_rows)


def paired_md_table(rows):
    headers = ["Comparison", "Budget", "n", "Mean diff (pp)",
               "95% CI (pp)", "Includes zero", "Significant"]
    table_rows = []
    for r in rows:
        table_rows.append([
            r["comparison"], r["budget_tokens"], r["n_questions"],
            f"{r['mean_diff_pp']:+.2f}",
            f"[{r['ci_low_pp']:.2f}, {r['ci_high_pp']:.2f}]",
            "yes" if r["includes_zero"] else "no",
            "no" if r["includes_zero"] else "yes",
        ])
    return md_table(headers, table_rows)


def nonsignificant_bullets(rows, metric_label):
    out = []
    for r in rows:
        if not r["includes_zero"]:
            continue
        out.append(
            f"- {r['dataset']} / {SYSTEM_SHORT[r['retriever']]} / "
            f"{r['comparison']} at B{r['budget_tokens']}: "
            f"{r['mean_diff_pp']:+.2f} pp (95% CI "
            f"[{r['ci_low_pp']:.2f}, {r['ci_high_pp']:.2f}]) includes zero; "
            f"not statistically significant ({metric_label})."
        )
    return out


def significant_bullets(rows):
    out = []
    for r in rows:
        if r["includes_zero"]:
            continue
        direction = "higher" if r["mean_diff_pp"] > 0 else "lower"
        out.append(
            f"- {r['dataset']} / {SYSTEM_SHORT[r['retriever']]} / "
            f"{r['comparison']} at B{r['budget_tokens']}: "
            f"{r['mean_diff_pp']:+.2f} pp (95% CI "
            f"[{r['ci_low_pp']:.2f}, {r['ci_high_pp']:.2f}]) excludes zero; "
            f"{r['comparison'].split('-')[0]} is {direction} than "
            f"{r['comparison'].split('-')[1]}."
        )
    return out


def pareto_frontier(rows):
    points = sorted(
        rows,
        key=lambda r: (r["mean_o200k_tokens"], -r["answer_evidence_recall"]),
    )
    frontier = []
    best = -1.0
    for row in points:
        if row["answer_evidence_recall"] > best:
            frontier.append(row)
            best = row["answer_evidence_recall"]
    return frontier


def best_at_budget(aggregates, dataset, retriever, budget):
    rows = [
        r for r in aggregates
        if r["dataset"] == dataset and r["retriever"] == retriever
        and r["method"] in METHODS and r["budget_tokens"] == budget
    ]
    return max(rows, key=lambda r: r["answer_evidence_recall"])


def check_phase3_checksums() -> dict:
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in manifest["phase3_result_checksums_sha256"].items():
        path = RESULTS / name
        if not path.exists():
            mismatches.append((name, "missing"))
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append((name, "mismatch"))
    return {
        "checked": len(manifest["phase3_result_checksums_sha256"]),
        "mismatches": mismatches,
    }


def write_m2_examples(lines: list[str], examples: list[dict]) -> None:
    lines += [
        "## Five Verbatim M2 Answer-Loss Examples",
        "",
        "These are the same five examples retained from Phase 4. Each is a TyDi "
        "question where M0 top-10 contained a gold answer but the M2 context "
        "did not. Text is copied verbatim from the dataset and experiment "
        "output.",
        "",
    ]
    for i, example in enumerate(examples, start=1):
        lines += [
            f"### Example {i} ({example['dataset']}, "
            f"{SYSTEM_SHORT[example['retriever']]}, B{example['budget']})",
            "",
            f"- Question ID: `{example['question_id']}`",
            f"- Question: {example['question']}",
            f"- Gold answer(s): {', '.join(example['gold_answers'])}",
            f"- Answer-bearing sentence: "
            f"{example['answer_sentence'] or 'n/a'}",
            f"- Parent passage: `{example['answer_passage_id']}` at retrieval "
            f"rank {example['answer_passage_rank']}",
            "",
            "**M2 compressed context (verbatim)**",
            "",
            "```text",
            example["m2_context"],
            "```",
            "",
            f"**Why the answer was lost:** {example['reason']}",
            "",
        ]


def main() -> int:
    records = load_jsonl(PER_QUERY)
    aggregates = build_aggregates(records)
    consistency = check_against_phase4(aggregates)
    if consistency["mismatches"]:
        raise RuntimeError(
            "aggregate recomputation does not match phase4_results_all.csv: "
            f"{consistency['mismatches'][:5]}"
        )

    tydi_m0 = [
        r for r in aggregates
        if r["dataset_kind"] == "tydi" and r["method"] == "M0"
    ]
    tydi_budget = [
        r for r in aggregates
        if r["dataset_kind"] == "tydi" and r["method"] in METHODS
    ]
    miracl_m0 = [
        r for r in aggregates
        if r["dataset_kind"] == "miracl" and r["method"] == "M0"
    ]
    miracl_budget = [
        r for r in aggregates
        if r["dataset_kind"] == "miracl" and r["method"] in METHODS
    ]

    aggregate_fields = [
        "dataset", "dataset_kind", "retriever", "method", "setting",
        "budget_tokens", "top_k", "n_questions", "n_answer_denominator",
        "n_gold_denominator", "n_loss_denominator", "mean_o200k_tokens",
        "o200k_ci_low", "o200k_ci_high", "mean_e5_tokens", "e5_ci_low",
        "e5_ci_high", "answer_evidence_recall", "recall_ci_low",
        "recall_ci_high", "gold_passage_retention", "gold_ci_low",
        "gold_ci_high", "evidence_loss_rate", "loss_ci_low", "loss_ci_high",
    ]
    write_csv(RESULTS / "phase4b_tydi_m0.csv", tydi_m0, aggregate_fields)
    write_csv(RESULTS / "phase4b_tydi_budget.csv", tydi_budget, aggregate_fields)
    write_csv(RESULTS / "phase4b_miracl_m0.csv", miracl_m0, aggregate_fields)
    write_csv(
        RESULTS / "phase4b_miracl_budget.csv", miracl_budget, aggregate_fields
    )

    paired_tydi = build_paired_rows(
        records, TYDI_DATASETS, "answer_evidence_recall", False,
        "answer_evidence_recall",
    )
    paired_miracl = build_paired_rows(
        records, [MIRACL_DATASET], "gold_passage_retained", True,
        "gold_passage_retention",
    )
    paired_all = paired_tydi + paired_miracl
    paired_fields = [
        "dataset", "retriever", "metric", "comparison", "budget_tokens",
        "n_questions", "mean_diff_pp", "ci_low_pp", "ci_high_pp",
        "includes_zero", "significant", "n_boot", "bootstrap_seed",
        "interpretation",
    ]
    write_csv(
        RESULTS / "phase4b_paired_bootstrap.csv", paired_all, paired_fields
    )

    _, heldout, validation = tydi_split()
    rankings = load_rankings()
    needed_parents: set[str] = set()
    for source, questions in [
        ("tydi_train", heldout),
        ("tydi_validation", validation),
    ]:
        for retriever in SYSTEMS:
            for question in questions:
                needed_parents.update(
                    rankings[source][retriever].get(question["question_id"], [])[
                        :10
                    ]
                )
    parent_texts = load_parent_texts(needed_parents)

    o200k = O200k()
    e5 = E5Counter()
    m4_rows = compute_m4_rows(
        rankings, parent_texts, heldout, validation, o200k, e5
    )
    m4_stats = build_m4_stats(m4_rows)
    m4_examples = select_m4_examples(m4_rows)
    m4_fields = [
        "dataset", "retriever", "n_questions", "n_with_gold_answers",
        "o200k_mean", "o200k_median", "o200k_min", "o200k_max",
        "o200k_ci_low", "o200k_ci_high", "e5_mean", "e5_median", "e5_min",
        "e5_max", "e5_ci_low", "e5_ci_high", "mean_selected_sentences",
        "zero_match_count", "zero_match_share", "multi_match_count",
        "multi_match_share", "answer_recall", "answer_recall_ci_low",
        "answer_recall_ci_high", "previous_mean_o200k", "difference_o200k",
        "previous_mean_e5", "difference_e5",
    ]
    write_csv(RESULTS / "phase4b_m4_validation.csv", m4_stats, m4_fields)
    (RESULTS / "phase4b_m4_validation.json").write_text(
        json.dumps({
            "rule": (
                "Minimum unique answer-bearing sentence evidence: split with "
                "the established splitter, match gold answers with NFC and "
                "Bangla/ASCII digit normalization, deduplicate identical "
                "normalized sentences, keep original text and order, keep all "
                "distinct matching sentences."
            ),
            "stats": m4_stats,
            "examples": m4_examples,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_tydi_path = plot_tydi(aggregates, m4_stats)
    plot_miracl_path = plot_miracl(aggregates)

    m2_examples = json.loads(M2_EXAMPLES.read_text(encoding="utf-8"))
    if len(m2_examples) != 5:
        raise RuntimeError(
            f"expected 5 M2 answer-loss examples, found {len(m2_examples)}"
        )

    phase3_checksums = check_phase3_checksums()
    if phase3_checksums["mismatches"]:
        raise RuntimeError(f"Phase 3 checksum mismatch: {phase3_checksums}")

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    memory = {
        "peak_rss_mib": peak_bytes / (1024 ** 2),
        "measurement": (
            "resource.getrusage(RUSAGE_SELF).ru_maxrss; macOS reports bytes, "
            "Linux KiB"
        ),
    }
    phase4_config = json.loads(PHASE4_CONFIG.read_text(encoding="utf-8"))

    non_significant = nonsignificant_bullets(paired_tydi, "answer evidence") + \
        nonsignificant_bullets(
            paired_miracl, "gold-passage retention, not answer evidence"
        )
    significant = significant_bullets(paired_tydi) + significant_bullets(
        paired_miracl
    )

    lines: list[str] = [
        "# Phase 4b: Statistical Reporting, Oracle Validation, and Finalization",
        "",
        "## Scope and integrity",
        "",
        "- This phase is a read-only analysis of the frozen Phase 4 per-query "
        "outputs and frozen Phase 3 rankings. No retrieval, embedding "
        "generation, embedding cache, model, dataset, or external API was "
        "used or rebuilt.",
        "- Dense (multilingual-e5-small) is the primary retriever and the main "
        "MIRACL-train-tuned Hybrid (RRF) is secondary.",
        "- TyDi answer-evidence results and MIRACL gold-passage-retention "
        "results are kept separate throughout. MIRACL has no answer strings, "
        "so no answer recall or answer-loss value is reported or inferred.",
        f"- Bootstrap: {N_BOOT:,} question-level resamples, seed {BOOT_SEED}, "
        "percentile 95% confidence intervals. Paired comparisons use the same "
        "bootstrap replicate indices for both methods over the same question "
        "IDs.",
        f"- Source per-query records: {len(records):,}. Aggregate rows "
        f"recomputed: {len(aggregates)}; all matched the approved Phase 4 "
        f"aggregates for {consistency['compared']} non-M4 rows.",
        "",
        "## 1. TyDi held-out M0 baselines (Dense)",
        "",
        m0_md_table(m0_rows_for(aggregates, "tydi_heldout", SYSTEMS[0])),
        "",
        "## 2. TyDi held-out M0 baselines (Hybrid, MIRACL-tuned)",
        "",
        m0_md_table(m0_rows_for(aggregates, "tydi_heldout", SYSTEMS[1])),
        "",
        "M0 points are uncompressed top-k contexts. They are shown separately "
        "from the compression-method curves in the plot and are not connected "
        "as a compression curve.",
        "",
        "## 3. TyDi held-out budget tables (Dense)",
        "",
        budget_md_table(budget_rows_for(aggregates, "tydi_heldout", SYSTEMS[0])),
        "",
        "## 4. TyDi held-out budget tables (Hybrid, MIRACL-tuned)",
        "",
        budget_md_table(budget_rows_for(aggregates, "tydi_heldout", SYSTEMS[1])),
        "",
        "Evidence-loss denominators are questions where M0 top-10 contained the "
        "gold answer. Answer-recall denominators are all TyDi held-out "
        "questions. Gold-retention denominators are questions where a gold "
        "passage is in M0 top-10.",
        "",
        "## 5. Paired bootstrap differences (TyDi held-out, answer evidence)",
        "",
        "Positive differences mean the first method has higher answer-evidence "
        "recall. A difference is called statistically significant only when "
        "its two-sided 95% CI excludes zero.",
        "",
        "### Dense",
        "",
        paired_md_table(
            [r for r in paired_tydi
             if r["dataset"] == "tydi_heldout" and r["retriever"] == SYSTEMS[0]]
        ),
        "",
        "### Hybrid",
        "",
        paired_md_table(
            [r for r in paired_tydi
             if r["dataset"] == "tydi_heldout" and r["retriever"] == SYSTEMS[1]]
        ),
        "",
        "## 6. TyDi validation (secondary)",
        "",
        "### M0 (Dense)",
        "",
        m0_md_table(m0_rows_for(aggregates, "tydi_validation", SYSTEMS[0])),
        "",
        "### M0 (Hybrid)",
        "",
        m0_md_table(m0_rows_for(aggregates, "tydi_validation", SYSTEMS[1])),
        "",
        "### Budgets (Dense)",
        "",
        budget_md_table(
            budget_rows_for(aggregates, "tydi_validation", SYSTEMS[0])
        ),
        "",
        "### Budgets (Hybrid)",
        "",
        budget_md_table(
            budget_rows_for(aggregates, "tydi_validation", SYSTEMS[1])
        ),
        "",
        "### Paired differences (Dense)",
        "",
        paired_md_table(
            [r for r in paired_tydi
             if r["dataset"] == "tydi_validation"
             and r["retriever"] == SYSTEMS[0]]
        ),
        "",
        "### Paired differences (Hybrid)",
        "",
        paired_md_table(
            [r for r in paired_tydi
             if r["dataset"] == "tydi_validation"
             and r["retriever"] == SYSTEMS[1]]
        ),
        "",
        "## 7. MIRACL dev (Dense)",
        "",
        "MIRACL reports only gold-passage retention and token cost. Answer "
        "recall and answer-evidence loss are not reported.",
        "",
        "### M0",
        "",
        m0_md_table(
            m0_rows_for(aggregates, MIRACL_DATASET, SYSTEMS[0]),
            include_answer=False,
        ),
        "",
        "### Budgets",
        "",
        budget_md_table(
            budget_rows_for(aggregates, MIRACL_DATASET, SYSTEMS[0]),
            include_answer=False,
        ),
        "",
        "## 8. MIRACL dev (Hybrid)",
        "",
        "### M0",
        "",
        m0_md_table(
            m0_rows_for(aggregates, MIRACL_DATASET, SYSTEMS[1]),
            include_answer=False,
        ),
        "",
        "### Budgets",
        "",
        budget_md_table(
            budget_rows_for(aggregates, MIRACL_DATASET, SYSTEMS[1]),
            include_answer=False,
        ),
        "",
        "## 9. MIRACL paired gold-passage-retention differences",
        "",
        "These are **gold-passage-retention differences**, not "
        "answer-evidence differences. MIRACL has no gold answer strings.",
        "",
        "Paired MIRACL retention comparisons are restricted to questions "
        "whose gold passage was in the M0 top-10, because retention is "
        "undefined otherwise; n is therefore the gold-retention denominator "
        "(378 for Dense, 375 for Hybrid), not the full 411-question set.",
        "",
        "### Dense",
        "",
        paired_md_table(
            [r for r in paired_miracl if r["retriever"] == SYSTEMS[0]]
        ),
        "",
        "### Hybrid",
        "",
        paired_md_table(
            [r for r in paired_miracl if r["retriever"] == SYSTEMS[1]]
        ),
        "",
        "## 10. Oracle M4 validation (TyDi only)",
        "",
        "M4 was recomputed with the minimum unique answer-bearing sentence "
        "rule: split with the established splitter, normalize with NFC and "
        "Bangla/ASCII digit mapping, deduplicate identical normalized "
        "sentences, preserve original text and order, and keep all genuinely "
        "distinct answer-bearing sentences. M4 remains an oracle lower bound "
        "and is not an ordinary budgeted method.",
        "",
        "### Token statistics",
        "",
        md_table(
            ["Dataset", "Retriever", "n", "o200k mean [95% CI]",
             "o200k median", "o200k min", "o200k max",
             "e5 mean [95% CI]", "e5 median", "e5 min", "e5 max",
             "Previous mean o200k", "Difference"],
            [[
                s["dataset"], SYSTEM_SHORT[s["retriever"]], s["n_questions"],
                fmt_ci(s["o200k_mean"], s["o200k_ci_low"], s["o200k_ci_high"], 1),
                f"{s['o200k_median']:.1f}", f"{s['o200k_min']:.0f}",
                f"{s['o200k_max']:.0f}",
                fmt_ci(s["e5_mean"], s["e5_ci_low"], s["e5_ci_high"], 1),
                f"{s['e5_median']:.1f}", f"{s['e5_min']:.0f}",
                f"{s['e5_max']:.0f}",
                f"{s['previous_mean_o200k']:.1f}"
                if s["previous_mean_o200k"] is not None else "n/a",
                f"{s['difference_o200k']:+.1f}"
                if s["difference_o200k"] is not None else "n/a",
            ] for s in m4_stats],
        ),
        "",
        "### Sentence-selection statistics",
        "",
        md_table(
            ["Dataset", "Retriever", "n", "Mean selected unique sentences",
             "Zero matches n (%)", ">1 unique sentence n (%)",
             "Questions with gold answers"],
            [[
                s["dataset"], SYSTEM_SHORT[s["retriever"]], s["n_questions"],
                f"{s['mean_selected_sentences']:.3f}",
                f"{s['zero_match_count']} ({s['zero_match_share']:.2%})",
                f"{s['multi_match_count']} ({s['multi_match_share']:.2%})",
                s["n_with_gold_answers"],
            ] for s in m4_stats],
        ),
        "",
        "### Comparison with the previous 167.5-token result",
        "",
    ]
    for s in m4_stats:
        if s["previous_mean_o200k"] is None:
            continue
        change = s["difference_o200k"]
        if abs(change) < 0.05:
            text = (
                f"- {s['dataset']} / {SYSTEM_SHORT[s['retriever']]}: "
                f"recomputed mean {s['o200k_mean']:.1f} vs previous "
                f"{s['previous_mean_o200k']:.1f} o200k tokens. The value is "
                "unchanged under the minimum-unique-sentence rule."
            )
        else:
            text = (
                f"- {s['dataset']} / {SYSTEM_SHORT[s['retriever']]}: "
                f"recomputed mean {s['o200k_mean']:.1f} vs previous "
                f"{s['previous_mean_o200k']:.1f} o200k tokens "
                f"({change:+.1f}). The difference is due to deduplicating "
                "identical normalized answer-bearing sentences across the ten "
                "retrieved passages, not to a change in answer matching."
            )
        lines.append(text)
    lines += [
        "",
        "### Examples with multiple unique answer-bearing sentences",
        "",
    ]
    if m4_examples:
        for i, example in enumerate(m4_examples, start=1):
            lines += [
                f"**Example {i}** ({example['dataset']} / "
                f"{SYSTEM_SHORT[example['retriever']]} / "
                f"`{example['question_id']}`)",
                "",
                f"- Question: {example['question']}",
                f"- Gold answer(s): {', '.join(example['gold_answers'])}",
                f"- Selected unique sentences: {example['n_selected']}",
                f"- o200k tokens: {example['o200k_tokens']}; "
                f"e5 tokens: {example['e5_tokens']}",
                "",
                "```text",
                example["context"],
                "```",
                "",
            ]
    else:
        lines.append(
            "No question selected more than one unique answer-bearing "
            "sentence."
        )
    lines += [
        "",
        "## 11. Non-significant paired comparisons",
        "",
        "The following comparisons have a 95% CI that includes zero and are "
        "therefore **not statistically significant** at the 5% level:",
        "",
    ]
    lines += non_significant if non_significant else [
        "- None. Every reported paired comparison excluded zero."
    ]
    lines += [
        "",
        "## 12. Significant paired comparisons",
        "",
        "These comparisons exclude zero at the 5% level:",
        "",
    ]
    lines += significant if significant else [
        "- None. No reported paired comparison excluded zero."
    ]
    lines += [
        "",
        "## 13. Interpretation",
        "",
    ]
    for retriever in SYSTEMS:
        frontier = pareto_frontier([
            r for r in aggregates
            if r["dataset"] == "tydi_heldout"
            and r["retriever"] == retriever
            and r["method"] in ["M0", "M1", "M2", "M3", "M5"]
        ])
        points = []
        for row in frontier:
            label = (
                f"M0 k={row['top_k']}" if row["method"] == "M0"
                else f"{row['method']} B{row['budget_tokens']}"
            )
            points.append(
                f"{label} ({row['mean_o200k_tokens']:.1f} tokens, "
                f"recall {row['answer_evidence_recall']:.4f})"
            )
        lines += [
            f"- **{SYSTEM_SHORT[retriever]} held-out Pareto frontier** among "
            "M0/M1/M2/M3/M5 point estimates: " + "; ".join(points) + ".",
        ]
        winners = []
        for budget in BUDGETS:
            best = best_at_budget(
                aggregates, "tydi_heldout", retriever, budget
            )
            winners.append(
                f"B{budget}: {best['method']} "
                f"({best['answer_evidence_recall']:.4f})"
            )
        lines.append(
            f"- **{SYSTEM_SHORT[retriever]} highest point estimate at each "
            "budget**: " + "; ".join(winners) + ". These are point-estimate "
            "rankings; significance claims are limited to the paired "
            "comparisons reported above."
        )
    lines += [
        "- M2 generally has the strongest recall/token trade-off among "
        "deployable extractive methods, but the paired tables should be used "
        "for any specific M2-versus-M1 or M3-versus-M2 claim.",
        "- M2 versus M5 is the clearest separation: random sentence selection "
        "is far below extractive e5 selection at every budget.",
        "- M4 is an oracle reference only. It shows the remaining headroom but "
        "is not a deployable compression method and is excluded from the "
        "budgeted method comparisons.",
        "- MIRACL conclusions are limited to gold-passage retention. No "
        "answer-correctness claim is made from MIRACL.",
        "",
        "## 14. Plots",
        "",
        f"![TyDi answer-evidence recall vs context cost]"
        f"({plot_tydi_path.name})",
        "",
        f"![MIRACL gold-passage retention vs context cost]"
        f"({plot_miracl_path.name})",
        "",
        "Error bars show 95% bootstrap confidence intervals for both mean "
        "o200k tokens and the y-axis metric. M0 baseline points are not "
        "connected. MIRACL is plotted only on a gold-passage-retention axis.",
        "",
    ]

    write_m2_examples(lines, m2_examples)
    lines += [
        "## 15. Tests, runtime, and memory",
        "",
        "Targeted tests: `results/phase4b_test_results.txt`.",
        "",
        "Full suite: `results/phase4b_test_results_all.txt`.",
        "",
        f"- Phase 4b analysis runtime: {time.time() - START:.1f} seconds.",
        f"- Phase 4b peak RSS: {memory['peak_rss_mib']:.1f} MiB "
        f"({memory['measurement']}).",
        f"- Phase 4 approved run runtime: "
        f"{phase4_config['runtime_seconds']:.1f} seconds; peak RSS "
        f"{phase4_config['memory']['peak_rss_mib']:.1f} MiB; device "
        f"{phase4_config['hardware']['device']}.",
        "",
        "## 16. Validation checks",
        "",
        f"- Per-query records read: {len(records):,}.",
        f"- Aggregate rows recomputed: {len(aggregates)}; compared with "
        f"approved Phase 4 aggregates: {consistency['compared']} rows, "
        "0 mismatches.",
        f"- Paired comparisons: {len(paired_all)} rows; every comparison "
        "validated identical question-ID sets.",
        f"- Phase 3 checksums rechecked: {phase3_checksums['checked']} files, "
        f"{len(phase3_checksums['mismatches'])} mismatches.",
        f"- M2 answer-loss examples retained: {len(m2_examples)}.",
        "- No dataset, embedding file, cache, model, index, or large derived "
        "artifact was written or staged by this phase.",
        "",
        "## 17. Output files",
        "",
        "- `phase4b_tydi_m0.csv`",
        "- `phase4b_tydi_budget.csv`",
        "- `phase4b_miracl_m0.csv`",
        "- `phase4b_miracl_budget.csv`",
        "- `phase4b_paired_bootstrap.csv`",
        "- `phase4b_m4_validation.csv` / `.json` / `_examples.md`",
        "- `phase4b_recall_vs_tokens.png`",
        "- `phase4b_miracl_retention_vs_tokens.png`",
        "- `phase4b_report.md`",
        "- `phase4b_config.json`",
        "",
    ]
    (RESULTS / "phase4b_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    example_lines = ["# Phase 4b: M4 multi-sentence examples", ""]
    for i, example in enumerate(m4_examples, start=1):
        example_lines += [
            f"## Example {i}",
            "",
            f"- Dataset: `{example['dataset']}`",
            f"- Retriever: `{example['retriever']}`",
            f"- Question ID: `{example['question_id']}`",
            f"- Question: {example['question']}",
            f"- Gold answer(s): {', '.join(example['gold_answers'])}",
            f"- Selected unique sentences: {example['n_selected']}",
            f"- o200k tokens: {example['o200k_tokens']}",
            f"- e5 tokens: {example['e5_tokens']}",
            "",
            "```text",
            example["context"],
            "```",
            "",
        ]
    (RESULTS / "phase4b_m4_examples.md").write_text(
        "\n".join(example_lines) + "\n", encoding="utf-8"
    )

    config = {
        "phase": "4b",
        "source_per_query": str(PER_QUERY.relative_to(PROJECT_ROOT)),
        "source_rankings": str(RANKINGS.relative_to(PROJECT_ROOT)),
        "source_corpus": str(CORPUS.relative_to(PROJECT_ROOT)),
        "bootstrap": {
            "resamples": N_BOOT,
            "seed": BOOT_SEED,
            "ci": 0.95,
            "unit": "question",
            "paired": "same replicate indices for both methods",
        },
        "datasets": {
            "tydi_datasets": TYDI_DATASETS,
            "miracl_dataset": MIRACL_DATASET,
            "tydi_split_seed": TYDI_SPLIT_SEED,
            "tydi_tuning_n": TYDI_TUNING_N,
            "tydi_heldout_n": len(heldout),
        },
        "methods": {
            "m0_ks": M0_KS,
            "budgeted": METHODS,
            "budgets": BUDGETS,
            "m4": "TyDi-only oracle, minimum unique answer-bearing sentences",
        },
        "m4_rule": {
            "splitter": "established Bangla danda / question mark / newline",
            "matching": "NFC, Bangla/ASCII digits, exact normalized substring",
            "dedupe": "identical normalized sentences keep first text and order",
            "distinct_sentences": "all genuinely distinct answer sentences kept",
        },
        "runtime_seconds": round(time.time() - START, 1),
        "memory": memory,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu analysis; no model inference",
        },
        "checks": {
            "records": len(records),
            "aggregate_rows": len(aggregates),
            "aggregate_comparison": consistency,
            "paired_rows": len(paired_all),
            "phase3_checksums": phase3_checksums,
            "m2_examples": len(m2_examples),
        },
        "outputs": [
            "phase4b_tydi_m0.csv",
            "phase4b_tydi_budget.csv",
            "phase4b_miracl_m0.csv",
            "phase4b_miracl_budget.csv",
            "phase4b_paired_bootstrap.csv",
            "phase4b_m4_validation.csv",
            "phase4b_m4_validation.json",
            "phase4b_m4_examples.md",
            "phase4b_recall_vs_tokens.png",
            "phase4b_miracl_retention_vs_tokens.png",
            "phase4b_report.md",
            "phase4b_config.json",
        ],
        "read_only_retrieval_and_embeddings": True,
        "external_api_calls": False,
        "new_models": False,
    }
    (RESULTS / "phase4b_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[done] records={len(records):,} aggregate_rows={len(aggregates)}")
    print(f"[done] paired_rows={len(paired_all)} m4_rows={len(m4_rows)}")
    print(f"[done] report={RESULTS / 'phase4b_report.md'}")
    print(f"[done] runtime={time.time() - START:.1f}s "
          f"peak={memory['peak_rss_mib']:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
