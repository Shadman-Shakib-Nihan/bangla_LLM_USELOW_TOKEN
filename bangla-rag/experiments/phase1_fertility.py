"""Phase 1: tokenizer fertility on parallel Bangla/English sentences.

Computes tokens-per-sentence and the Bangla/English ratio for Gemini
(count_tokens API), OpenAI o200k (tiktoken) and Qwen2.5 (local tokenizer).

Usage:
    cd bangla-rag
    .venv/bin/python experiments/phase1_fertility.py
    .venv/bin/python experiments/phase1_fertility.py --limit 100 --skip-gemini
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.tokenizers import (
    GeminiCounter,
    QwenCounter,
    TiktokenCounter,
    compute_fertility,
    load_flores_pairs,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "phase1.yaml",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="override data.limit (number of sentences)")
    parser.add_argument("--split", default=None,
                        help="override data.split (dev or devtest)")
    parser.add_argument("--skip-gemini", action="store_true",
                        help="skip the Gemini API tokenizer")
    return parser.parse_args()


def build_counters(cfg: dict, skip_gemini: bool) -> tuple[list, bool]:
    """Return (usable_counters, gemini_skipped)."""
    counters = []
    gemini_skipped = False
    tok_cfg = cfg["tokenizers"]

    if tok_cfg.get("use_gemini", True) and not skip_gemini:
        try:
            counters.append(GeminiCounter(model=tok_cfg["gemini_model"]))
            print(f"[ok] Gemini counter ready: {tok_cfg['gemini_model']}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[skip] Gemini counter unavailable: {exc}")
            gemini_skipped = True

    if tok_cfg.get("use_tiktoken", True):
        counters.append(TiktokenCounter())
        print("[ok] tiktoken o200k_base ready")

    if tok_cfg.get("use_qwen", True):
        counters.append(QwenCounter(repo_id=tok_cfg["qwen_repo"]))
        print(f"[ok] Qwen tokenizer ready: {tok_cfg['qwen_repo']}")

    return counters, gemini_skipped


def save_versions(path: Path, cfg: dict, n_sentences: int, counters: list) -> None:
    import importlib.metadata as md

    packages = [
        "google-genai",
        "tiktoken",
        "tokenizers",
        "huggingface-hub",
        "numpy",
        "pandas",
        "matplotlib",
        "pyyaml",
    ]
    lines = [
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"python: {platform.python_version()} ({platform.platform()})",
        f"machine: {platform.machine()}",
        f"seed: {cfg.get('seed')}",
        f"split: {cfg['data']['split']}",
        f"n_sentences: {n_sentences}",
        f"gemini_model: {cfg['tokenizers']['gemini_model']}",
        f"qwen_repo: {cfg['tokenizers']['qwen_repo']}",
        f"counters_used: {[getattr(c, 'name', 'unavailable') for c in counters]}",
        "packages:",
    ]
    for pkg in packages:
        try:
            lines.append(f"  {pkg}=={md.version(pkg)}")
        except md.PackageNotFoundError:
            lines.append(f"  {pkg}: NOT INSTALLED")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(summary: pd.DataFrame, out_path: Path, n_sentences: int) -> None:
    fig, (ax_counts, ax_ratio) = plt.subplots(1, 2, figsize=(13, 5.5))

    x = range(len(summary))
    width = 0.38
    ax_counts.bar(
        [i - width / 2 for i in x], summary["mean_bn_tokens"],
        width=width, label="Bangla", color="#c0392b",
    )
    ax_counts.bar(
        [i + width / 2 for i in x], summary["mean_en_tokens"],
        width=width, label="English", color="#2980b9",
    )
    ax_counts.set_xticks(list(x))
    ax_counts.set_xticklabels(summary["tokenizer"], rotation=15)
    ax_counts.set_ylabel("Mean tokens per sentence")
    ax_counts.set_title(f"Tokens per sentence (FLORES-200, n={n_sentences})")
    ax_counts.legend()
    ax_counts.grid(axis="y", alpha=0.3)

    ax_ratio.bar(
        [i - width / 2 for i in x], summary["mean_ratio_per_sentence"],
        width=width, label="mean per-sentence ratio", color="#8e44ad",
    )
    ax_ratio.bar(
        [i + width / 2 for i in x], summary["corpus_ratio"],
        width=width, label="corpus-total ratio", color="#f39c12",
    )
    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_ratio.set_xticks(list(x))
    ax_ratio.set_xticklabels(summary["tokenizer"], rotation=15)
    ax_ratio.set_ylabel("Bangla / English token ratio")
    ax_ratio.set_title("Bangla/English tokenizer fertility ratio")
    ax_ratio.legend()
    ax_ratio.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.limit is not None:
        cfg["data"]["limit"] = args.limit
    if args.split is not None:
        cfg["data"]["split"] = args.split

    pairs = load_flores_pairs(
        split=cfg["data"]["split"],
        root=PROJECT_ROOT / cfg["data"]["root"],
        limit=cfg["data"]["limit"],
        bn_lang=cfg["data"]["bn_lang"],
        en_lang=cfg["data"]["en_lang"],
    )
    n_sentences = len(pairs)
    print(f"[ok] loaded {n_sentences} Bangla/English pairs "
          f"(split={cfg['data']['split']})")

    counters, gemini_skipped = build_counters(cfg, args.skip_gemini)
    usable = counters

    rows = []
    for counter in usable:
        print(f"[run] counting with {counter.name} ...")
        rows.append(compute_fertility(pairs, counter))

    # Per-sentence table.
    per_sentence = pd.DataFrame({"bn": [b for b, _ in pairs],
                                 "en": [e for _, e in pairs]})
    for row in rows:
        tag = row["tokenizer"].replace(":", "_")
        per_sentence[f"tokens_bn_{tag}"] = row["bn_tokens"]
        per_sentence[f"tokens_en_{tag}"] = row["en_tokens"]

    out_cfg = cfg["output"]
    per_path = PROJECT_ROOT / out_cfg["per_sentence_csv"]
    per_path.parent.mkdir(parents=True, exist_ok=True)
    per_sentence.to_csv(per_path, index=False, encoding="utf-8")
    print(f"[ok] wrote {per_path.relative_to(PROJECT_ROOT)}")

    summary = pd.DataFrame(summarize(rows))
    summary_path = PROJECT_ROOT / out_cfg["summary_csv"]
    summary.to_csv(summary_path, index=False)
    print(f"[ok] wrote {summary_path.relative_to(PROJECT_ROOT)}")

    make_plot(summary, PROJECT_ROOT / out_cfg["plot_path"], n_sentences)
    print(f"[ok] wrote {out_cfg['plot_path']}")

    save_versions(PROJECT_ROOT / out_cfg["versions_path"], cfg, n_sentences, counters)
    print(f"[ok] wrote {out_cfg['versions_path']}")

    print("\n=== Phase 1 fertility summary ===")
    display = summary.copy()
    for col in ["mean_bn_tokens", "mean_en_tokens",
                "mean_ratio_per_sentence", "corpus_ratio"]:
        display[col] = display[col].map(lambda v: round(float(v), 3))
    print(display.to_string(index=False))

    if gemini_skipped:
        print("\n[warn] Gemini tokenizer was skipped; see message above. "
              "The table is incomplete until GEMINI_API_KEY is set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
