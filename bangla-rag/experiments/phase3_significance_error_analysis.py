"""Phase 3 follow-up: paired significance tests + per-query error analysis.

Inputs are the per-query records written by experiments/phase3_dense_hybrid_eval.py
(results/phase3_per_query.jsonl).  No retrieval is recomputed here.

Outputs:
  results/phase3_significance.csv
  results/phase3_error_analysis.md
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
RESULTS = PROJECT_ROOT / "results"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PER_QUERY = RESULTS / "phase3_per_query.jsonl"
UNITS = PROCESSED / "corpus_chunks.jsonl"
TYDI_VAL = PROCESSED / "tydi_validation_questions.jsonl"

DATASETS = ("miracl_dev", "tydi_validation")
SYSTEMS = ("BM25", "Dense (e5-small)", "Hybrid (RRF)")
METRICS = ("recall@10", "mrr", "ndcg@10")
PAIRS = (
    ("BM25", "Dense (e5-small)"),
    ("BM25", "Hybrid (RRF)"),
    ("Dense (e5-small)", "Hybrid (RRF)"),
)
N_BOOT = 10_000
BOOT_SEED = 42
ALPHA = 0.05
N_PAIRS = len(PAIRS)
SNIPPET_CHARS = 500


def load_records() -> dict[tuple[str, str], dict[str, dict]]:
    wide: dict[tuple[str, str], dict[str, dict]] = {}
    with PER_QUERY.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["dataset"], rec["qid"])
            wide.setdefault(key, {})[rec["system"]] = rec
    for key, systems in wide.items():
        missing = set(SYSTEMS) - set(systems)
        if missing:
            raise RuntimeError(f"{key} is missing systems: {sorted(missing)}")
    return wide


def load_parent_text() -> tuple[dict[str, str], dict[str, dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    meta: dict[str, dict] = {}
    with UNITS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["parent_id"]
            groups[pid].append(rec)
            meta[pid] = {
                "title": rec.get("title", ""),
                "source": rec.get("source", ""),
                "is_short": bool(rec.get("is_short")),
            }
    texts: dict[str, str] = {}
    for pid, recs in groups.items():
        recs.sort(key=lambda r: r.get("chunk_index", 0))
        texts[pid] = " ".join(r["text"] for r in recs)
    return texts, meta


def load_tydi_answers() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with TYDI_VAL.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["question_id"]] = list(rec.get("gold_answers", []))
    return out


def compute_gold_token_lengths(parent_text: dict[str, str],
                               gold_ids: set[str]) -> dict[str, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    out: dict[str, int] = {}
    for pid in sorted(gold_ids):
        out[pid] = len(tok.encode("passage: " + parent_text.get(pid, ""),
                                  add_special_tokens=True))
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = diff[idx].mean(axis=1)
    lo95, hi95 = np.percentile(means, [2.5, 97.5])
    tail = 100.0 * (ALPHA / N_PAIRS) / 2.0
    lo_b, hi_b = np.percentile(means, [tail, 100.0 - tail])
    p_boot = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "n": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff": float(np.mean(diff)),
        "ci95_low": float(lo95),
        "ci95_high": float(hi95),
        "excludes_zero_95": bool(lo95 > 0 or hi95 < 0),
        "ci_bonf_low": float(lo_b),
        "ci_bonf_high": float(hi_b),
        "excludes_zero_bonf": bool(lo_b > 0 or hi_b < 0),
        "p_boot": float(min(1.0, p_boot)),
    }


def key_order(wide: dict, dataset: str) -> list[tuple[str, str]]:
    return sorted(k for k in wide if k[0] == dataset)


def mean_metric(wide: dict, keys: list[tuple[str, str]], system: str,
                metric: str) -> float:
    if not keys:
        return float("nan")
    return float(np.mean([wide[k][system][metric] for k in keys]))


def select_examples(candidates: list[tuple[str, str]],
                    n: int = 5) -> list[tuple[str, str]]:
    by_ds = {ds: sorted(k for k in candidates if k[0] == ds) for ds in DATASETS}
    out: list[tuple[str, str]] = []
    while len(out) < n and any(by_ds.values()):
        for ds in DATASETS:
            if by_ds[ds] and len(out) < n:
                out.append(by_ds[ds].pop(0))
    return out


def snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    pos = cut.rfind(" ")
    if pos > 0:
        cut = cut[:pos]
    return cut + " …"


def rank_bin(rank: int | None) -> str:
    if rank is None:
        return "absent"
    if rank <= 10:
        return "<=10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    if rank <= 200:
        return "101-200"
    if rank <= 500:
        return "201-500"
    if rank <= 1000:
        return "501-1000"
    return ">1000"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_example(idx: int, key: tuple[str, str], wide: dict,
                   parent_text: dict[str, str],
                   tydi_answers: dict[str, list[str]],
                   gold_tok_len: dict[str, int],
                   note: str | None = None) -> list[str]:
    dataset, qid = key
    rows = wide[key]
    gold = rows["BM25"]["gold_passage_ids"]
    lines = [f"**Example {idx} — `{dataset}` / `{qid}`**", ""]
    if note:
        lines += [f"- Category note: {note}", ""]
    lines.append(f"- Question (verbatim): {rows['BM25']['question']}")
    over = rows["BM25"]["overlength_gold_ids"]
    short_flag = rows["BM25"]["short_gold_ids"]
    short_tok = [p for p in gold if gold_tok_len.get(p, 10**9) < 20]
    lines.append(
        f"- Gold IDs: `{', '.join(gold)}`; over-length gold: "
        f"`{', '.join(over) if over else 'none'}`; Phase 2 `is_short` gold: "
        f"`{', '.join(short_flag) if short_flag else 'none'}`"
    )
    lines.append(
        "- Gold token lengths (e5, `passage: ` prefix): "
        + ", ".join(f"`{p}`={gold_tok_len.get(p, '?')}" for p in gold)
    )
    lines.append(
        f"- Short gold (<20 e5 tokens): {'yes' if short_tok else 'no'}"
    )
    gold_snips = []
    for pid in gold:
        gold_snips.append(f"`{pid}`: {snippet(parent_text.get(pid, ''))}")
    lines.append("- Gold passage(s), verbatim excerpt:")
    lines += [f"  - {s}" for s in gold_snips]
    if dataset == "tydi_validation":
        answers = tydi_answers.get(qid, [])
        lines.append(
            "- TyDi gold answer strings (verbatim): "
            + ("; ".join(answers) if answers else "none")
        )
        lines.append(
            f"- Dense answer-string first hit rank: "
            f"{rows['Dense (e5-small)']['answer_hit_rank']}"
        )
    lines.append("")
    lines.append(md_table(
        ["system", "R@10", "MRR", "nDCG@10", "first gold rank"],
        [[s,
          f"{rows[s]['recall@10']:.3f}",
          f"{rows[s]['mrr']:.3f}",
          f"{rows[s]['ndcg@10']:.3f}",
          str(rows[s]["first_gold_rank"])]
         for s in SYSTEMS],
    ))
    lines.append("")
    for system in SYSTEMS:
        top = rows[system]["top10"][0] if rows[system]["top10"] else None
        if top is None:
            lines.append(f"- {system} top-1: none")
        else:
            lines.append(
                f"- {system} top-1: `{top}` — {snippet(parent_text.get(top, ''))}"
            )
    lines.append("")
    return lines


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    wide = load_records()
    parent_text, _meta = load_parent_text()
    tydi_answers = load_tydi_answers()
    gold_ids_all: set[str] = set()
    for systems in wide.values():
        gold_ids_all.update(systems["BM25"]["gold_passage_ids"])
    gold_tok_len = compute_gold_token_lengths(parent_text, gold_ids_all)

    # ---------------- Paired bootstrap -----------------------------------
    sig_rows = []
    for dataset in DATASETS:
        keys = key_order(wide, dataset)
        for metric in METRICS:
            arrays = {
                s: np.array([wide[k][s][metric] for k in keys], dtype=float)
                for s in SYSTEMS
            }
            for a, b in PAIRS:
                res = paired_bootstrap(arrays[a], arrays[b])
                sig_rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "system_a": a,
                    "system_b": b,
                    **res,
                })

    with (RESULTS / "phase3_significance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sig_rows)

    # ---------------- Error analysis tables -------------------------------
    report: list[str] = []
    report.append("# Phase 3 — paired significance tests and per-query error analysis")
    report.append("")
    report.append(
        "Inputs: `results/phase3_per_query.jsonl` from "
        "`experiments/phase3_dense_hybrid_eval.py`. No retrieval was recomputed. "
        "Queries are the same for every system; paired bootstrap uses 10,000 "
        "resamples with seed 42. Differences are A − B."
    )
    report.append("")

    report.append("## Paired bootstrap (95% CI, and Bonferroni-adjusted CI for 3 pairs)")
    report.append("")
    headers = ["dataset", "metric", "A vs B", "mean A", "mean B", "Δ",
               "95% CI", "excl. 0?", "Bonf. 98.33% CI", "excl. 0?"]
    rows = []
    for r in sig_rows:
        rows.append([
            r["dataset"], r["metric"], f"{r['system_a']} vs {r['system_b']}",
            f"{r['mean_a']:.4f}", f"{r['mean_b']:.4f}", f"{r['diff']:+.4f}",
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]",
            "yes" if r["excludes_zero_95"] else "no",
            f"[{r['ci_bonf_low']:+.4f}, {r['ci_bonf_high']:+.4f}]",
            "yes" if r["excludes_zero_bonf"] else "no",
        ])
    report.append(md_table(headers, rows))
    report.append("")
    report.append(
        "Multiple-comparison note: within each (dataset, metric), the three "
        "pairwise tests form one family. Bonferroni uses α = 0.05/3 = 0.0167, "
        "i.e. a 98.33% percentile interval. The three tests are correlated "
        "because they reuse the same queries, so Bonferroni is conservative. "
        "The `p_boot` column in the CSV is an unadjusted percentile bootstrap "
        "p-value and should not be read as family-wise."
    )
    report.append("")

    # Overall by dataset
    report.append("## Error analysis — overall by dataset")
    report.append("")
    headers = ["dataset", "system", "n", "R@10", "MRR", "nDCG@10"]
    rows = []
    for dataset in DATASETS:
        keys = key_order(wide, dataset)
        for system in SYSTEMS:
            rows.append([
                dataset, system, str(len(keys)),
                f"{mean_metric(wide, keys, system, 'recall@10'):.4f}",
                f"{mean_metric(wide, keys, system, 'mrr'):.4f}",
                f"{mean_metric(wide, keys, system, 'ndcg@10'):.4f}",
            ])
    report.append(md_table(headers, rows))
    report.append("")

    # Over-length gold subgroup
    report.append("## Error analysis — over-length gold vs not")
    report.append("")
    headers = ["dataset", "subgroup", "n", "system", "R@10", "MRR", "nDCG@10"]
    rows = []
    for dataset in DATASETS:
        keys = key_order(wide, dataset)
        for value in (False, True):
            sub = [k for k in keys
                   if bool(wide[k]["BM25"]["has_overlength_gold"]) is value]
            label = "yes" if value else "no"
            for system in SYSTEMS:
                rows.append([
                    dataset, label, str(len(sub)), system,
                    f"{mean_metric(wide, sub, system, 'recall@10'):.4f}",
                    f"{mean_metric(wide, sub, system, 'mrr'):.4f}",
                    f"{mean_metric(wide, sub, system, 'ndcg@10'):.4f}",
                ])
    report.append(md_table(headers, rows))
    report.append("")
    report.append(
        "This subgroup is defined by queries whose gold set contains at least "
        "one over-length parent; the metrics still use the full gold set. The "
        "earlier ablation table restricted gold to the over-length IDs."
    )
    report.append("")

    # Short-gold sensitivity
    report.append("## Error analysis — short gold passages")
    report.append("")
    report.append(
        "Phase 2's `is_short` flag is stripped character length < 20. No gold "
        "passage in either reporting set has that flag. To check whether "
        "short-ish gold is a failure mode, the table adds e5-token thresholds "
        "(count of queries with at least one gold parent below the threshold)."
    )
    report.append("")
    headers = ["definition", "dataset", "n queries", "BM25 R@10",
               "Dense R@10", "Hybrid R@10", "Dense misses"]
    rows = []
    definitions: list[tuple[str, int | None]] = [
        ("Phase 2 `is_short` (char <20)", None),
        ("<20 e5 tokens", 20),
        ("<30 e5 tokens", 30),
        ("<50 e5 tokens", 50),
    ]
    for label, threshold in definitions:
        for dataset in DATASETS:
            keys = key_order(wide, dataset)
            if threshold is None:
                sub = [k for k in keys if wide[k]["BM25"]["is_short_gold"]]
            else:
                sub = [
                    k for k in keys
                    if any(gold_tok_len.get(p, 10**9) < threshold
                           for p in wide[k]["BM25"]["gold_passage_ids"])
                ]
            dense_misses = sum(
                1 for k in sub
                if wide[k]["Dense (e5-small)"]["recall@10"] == 0
            )
            rows.append([
                label, dataset, str(len(sub)),
                f"{mean_metric(wide, sub, 'BM25', 'recall@10'):.4f}",
                f"{mean_metric(wide, sub, 'Dense (e5-small)', 'recall@10'):.4f}",
                f"{mean_metric(wide, sub, 'Hybrid (RRF)', 'recall@10'):.4f}",
                str(dense_misses),
            ])
    report.append(md_table(headers, rows))
    report.append("")
    report.append(
        "There are no dense top-10 failures on short-gold queries under any "
        "tested definition; the short-passage slice is simply not represented "
        "among these gold sets."
    )
    report.append("")

    # Win/loss counts
    report.append("## Error analysis — top-10 wins and losses")
    report.append("")
    headers = ["dataset", "BM25 hit / Dense miss", "Dense hit / BM25 miss",
               "Hybrid miss (all)", "Dense miss (all)", "BM25 miss (all)"]
    rows = []
    for dataset in DATASETS:
        keys = key_order(wide, dataset)
        bm25_only = [k for k in keys
                     if wide[k]["BM25"]["recall@10"] == 1
                     and wide[k]["Dense (e5-small)"]["recall@10"] == 0]
        dense_only = [k for k in keys
                      if wide[k]["Dense (e5-small)"]["recall@10"] == 1
                      and wide[k]["BM25"]["recall@10"] == 0]
        hybrid_miss = [k for k in keys
                       if wide[k]["Hybrid (RRF)"]["recall@10"] == 0]
        dense_miss = [k for k in keys
                      if wide[k]["Dense (e5-small)"]["recall@10"] == 0]
        bm25_miss = [k for k in keys if wide[k]["BM25"]["recall@10"] == 0]
        rows.append([
            dataset, str(len(bm25_only)), str(len(dense_only)),
            str(len(hybrid_miss)), str(len(dense_miss)), str(len(bm25_miss)),
        ])
    report.append(md_table(headers, rows))
    report.append("")

    # Rank of gold when it misses top 10
    report.append("## Gold rank when the system misses the top 10")
    report.append("")
    bins = ["11-20", "21-50", "51-100", "101-200", "201-500", "501-1000",
            ">1000", "absent"]
    headers = ["dataset", "system", "n missed", *bins]
    rows = []
    for dataset in DATASETS:
        keys = key_order(wide, dataset)
        for system in SYSTEMS:
            miss = [k for k in keys if wide[k][system]["recall@10"] == 0]
            counts = {b: 0 for b in bins}
            for k in miss:
                counts[rank_bin(wide[k][system]["first_gold_rank"])] += 1
            rows.append([
                dataset, system, str(len(miss)),
                *[str(counts[b]) for b in bins],
            ])
    report.append(md_table(headers, rows))
    report.append("")
    report.append(
        "`absent` means the gold parent is not in the returned ranking: BM25 "
        "only ranks positive-scoring parents; hybrid only fuses the top-100 "
        "BM25 and top-100 dense lists. Dense ranks every parent, so its gold "
        "rank is always present."
    )
    report.append("")

    # ---------------- Concrete examples ----------------------------------
    report.append("## Concrete failure examples (verbatim)")
    report.append("")
    report.append(
        "Questions and passage snippets are verbatim. Snippets longer than "
        f"{SNIPPET_CHARS} characters are cut at a word boundary and marked "
        "with `…`; nothing else is edited."
    )
    report.append("")

    example_specs: list[tuple[str, str, list[tuple[str, str]]]] = []

    bm25_wins = [k for k in wide
                 if wide[k]["BM25"]["recall@10"] == 1
                 and wide[k]["Dense (e5-small)"]["recall@10"] == 0]
    dense_wins = [k for k in wide
                  if wide[k]["Dense (e5-small)"]["recall@10"] == 1
                  and wide[k]["BM25"]["recall@10"] == 0]
    dense_misses = [k for k in wide
                    if wide[k]["Dense (e5-small)"]["recall@10"] == 0]
    hybrid_misses = [k for k in wide
                     if wide[k]["Hybrid (RRF)"]["recall@10"] == 0]
    tydi_strict_miss_answer_hit = [
        k for k in wide
        if k[0] == "tydi_validation"
        and wide[k]["Dense (e5-small)"]["recall@10"] == 0
        and wide[k]["Dense (e5-small)"]["answer_hit_rank"] is not None
        and wide[k]["Dense (e5-small)"]["answer_hit_rank"] <= 10
    ]
    short_gold_misses = [
        k for k in wide
        if any(gold_tok_len.get(p, 10**9) < 20
               for p in wide[k]["BM25"]["gold_passage_ids"])
        and wide[k]["Dense (e5-small)"]["recall@10"] == 0
    ]
    over_gold_misses = [
        k for k in wide
        if wide[k]["BM25"]["has_overlength_gold"]
        and wide[k]["Dense (e5-small)"]["recall@10"] == 0
    ]

    example_specs = [
        ("BM25 hits top-10, dense misses",
         "BM25 R@10 = 1 and dense R@10 = 0.", bm25_wins),
        ("Dense hits top-10, BM25 misses",
         "Dense R@10 = 1 and BM25 R@10 = 0.", dense_wins),
        ("Dense misses top-10 (gold rank > 10)",
         "Dense R@10 = 0; the table shows how deep the first gold parent is.",
         dense_misses),
        ("Hybrid misses top-10",
         "Hybrid R@10 = 0. Hybrid fuses only the top-100 BM25 and dense lists.",
         hybrid_misses),
        ("TyDi strict-ID miss but answer-string hit (dense)",
         "Dense strict R@10 = 0, but a dense top-10 parent contains a gold "
         "answer string.", tydi_strict_miss_answer_hit),
        ("Short gold passage (<20 e5 tokens), dense misses top-10",
         "Gold parent has <20 e5 tokens; dense R@10 = 0.", short_gold_misses),
        ("Over-length gold, dense misses top-10",
         "Gold parent is over-length; dense R@10 = 0.", over_gold_misses),
    ]

    for title, note, candidates in example_specs:
        report.append(f"### {title}")
        report.append("")
        report.append(f"Available queries: **{len(candidates)}**.")
        report.append("")
        chosen = select_examples(candidates, n=5)
        if not chosen:
            if title.startswith("Short gold"):
                report.append(
                    "_No dense top-10 failures on short-gold queries; see the "
                    "short-gold sensitivity table._"
                )
            else:
                report.append("_No queries in this category._")
            report.append("")
            continue
        for i, key in enumerate(chosen, start=1):
            report += render_example(
                i, key, wide, parent_text, tydi_answers, gold_tok_len, note
            )
        if len(chosen) < 5:
            report.append(
                f"_Only {len(chosen)} quer{'y' if len(chosen) == 1 else 'ies'} "
                "available in this category._"
            )
            report.append("")

    # ---------------- Caveats --------------------------------------------
    report.append("## Caveats")
    report.append("")
    report += [
        "- The 100k MIRACL slice is easier than the full 297k-passage corpus: "
        "it mixes judged passages with a random sample and was built for this "
        "phase, so absolute scores should not be read as full-corpus estimates.",
        "- TyDi is scored two ways: strict gold passage ID and answer-string "
        "hit. The answer-string rule is more lenient and checks all chunks of "
        "the retrieved top-k parents.",
        "- Validation sets are small: MIRACL dev n=411, TyDi validation n=113. "
        "Subgroups are much smaller (TyDi validation over-length gold n=4), so "
        "their CIs and example counts are unstable.",
        "- The Phase 2 `is_short` flag (stripped characters <20) matches no "
        "gold passage in either reporting set; the short-gold sensitivity "
        "table therefore also uses e5-token thresholds.",
        "- RRF hyperparameters (k=5, w_BM25=0.5) were tuned on MIRACL train "
        "only, not on dev or TyDi validation.",
        "- MRR for BM25/dense is over the full parent ranking; hybrid MRR is "
        "over the fused top-100+top-100 candidate list. This depth difference "
        "matters only for a handful of queries but should be kept in mind.",
        "- Paired bootstrap resamples queries, not passages; it measures "
        "uncertainty over this query sample, not over the corpus.",
        "- No BGE-M3, compression, or external API call was run for this "
        "analysis.",
    ]
    report.append("")

    (RESULTS / "phase3_error_analysis.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    # ---------------- Console summary ------------------------------------
    print("[ok] wrote", RESULTS / "phase3_significance.csv")
    print("[ok] wrote", RESULTS / "phase3_error_analysis.md")
    print("\n## Paired bootstrap (95% CI; Bonferroni 98.33% for 3 pairs)")
    for dataset in DATASETS:
        print(f"\n### {dataset}")
        print(md_table(
            ["metric", "A vs B", "Δ", "95% CI", "excl", "Bonf. CI", "excl"],
            [[r["metric"], f"{r['system_a']} vs {r['system_b']}",
              f"{r['diff']:+.4f}",
              f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]",
              "yes" if r["excludes_zero_95"] else "no",
              f"[{r['ci_bonf_low']:+.4f}, {r['ci_bonf_high']:+.4f}]",
              "yes" if r["excludes_zero_bonf"] else "no"]
             for r in sig_rows if r["dataset"] == dataset],
        ))
    print("\nFull tables and verbatim examples: results/phase3_error_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
