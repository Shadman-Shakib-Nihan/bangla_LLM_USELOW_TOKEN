"""Generate the Phase 2-3 Word report from the real result files.

The report is assembled from:
  * data/processed/corpus_manifest.json and phase2_stats.json
  * results/phase3_*.json / .csv and results/phase3_per_query.jsonl
  * the Phase 1 report, when building the combined Phases 1-3 document

No retrieval is recomputed and no API call is made. The only model artifact
loaded is the local e5 tokenizer, used to reproduce the short-gold sensitivity
table from the saved per-query records.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/make_phase2_phase3_report.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "results"
PROCESSED = PROJECT_ROOT / "data" / "processed"
UNITS = PROCESSED / "corpus_chunks.jsonl"

OUT_STANDALONE = RESULTS / "Phase2_Phase3_RAG_Report.docx"
OUT_COMBINED = RESULTS / "Bangla_RAG_Report_Phases_1-3.docx"
PHASE1_DOCX = RESULTS / "Phase1_Tokenizer_Fertility_Report.docx"

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
DATASETS = ("miracl_dev", "tydi_validation")
SYSTEMS = ("BM25", "Dense (e5-small)", "Hybrid (RRF)")


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def f4(value: float) -> str:
    return f"{value:.4f}"


def fci(row: pd.Series, metric: str) -> str:
    return (f"{row[metric]:.4f} "
            f"[{row[metric + '_ci_low']:.4f}, {row[metric + '_ci_high']:.4f}]")


def shade_cell(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_table(document: Document, headers: list[str], rows: list[list[str]],
              font_size: float = 8.5, style: str = "Light Grid Accent 1"):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = style
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    hdr = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(header)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(font_size)
        shade_cell(hdr[i], "1F3A5F")
    set_repeat_table_header(table.rows[0])
    for values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(values):
            cells[i].text = str(value)
            for paragraph in cells[i].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
    document.add_paragraph()
    return table


def add_bullet(document: Document, text: str) -> None:
    document.add_paragraph(text, style="List Bullet")


def add_number(document: Document, text: str) -> None:
    document.add_paragraph(text, style="List Number")


def add_code(document: Document, lines: list[str]) -> None:
    for line in lines:
        p = document.add_paragraph()
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)


def add_caption(document: Document, text: str) -> None:
    p = document.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def setup_document(document: Document, margins: float = 0.7) -> None:
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for section in document.sections:
        section.top_margin = Inches(margins)
        section.bottom_margin = Inches(margins)
        section.left_margin = Inches(margins)
        section.right_margin = Inches(margins)
    footer = document.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Phase 2-3 Report - Bangla/English token-efficient RAG  |  page ")
    add_page_number(p)


# ---------------------------------------------------------------------------
# Data loading and derived tables
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_per_query() -> dict[tuple[str, str], dict[str, dict]]:
    wide: dict[tuple[str, str], dict[str, dict]] = {}
    with (RESULTS / "phase3_per_query.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["dataset"], rec["qid"])
            wide.setdefault(key, {})[rec["system"]] = rec
    for key, systems in wide.items():
        missing = set(SYSTEMS) - set(systems)
        if missing:
            raise RuntimeError(f"{key} is missing systems: {sorted(missing)}")
    return wide


def load_gold_token_lengths(wide: dict) -> dict[str, int]:
    """Reproduce the e5-token length of every gold parent from the units file."""
    os.environ.setdefault(
        "HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface")
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "intfloat/multilingual-e5-small", local_files_only=True
    )
    gold_ids: set[str] = set()
    for systems in wide.values():
        gold_ids.update(systems["BM25"]["gold_passage_ids"])
    parts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with UNITS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["parent_id"]
            if pid in gold_ids:
                parts[pid].append((int(rec.get("chunk_index", 0)), rec["text"]))
    lengths: dict[str, int] = {}
    for pid, chunks in parts.items():
        chunks.sort()
        text = " ".join(chunk for _, chunk in chunks)
        lengths[pid] = len(
            tokenizer.encode("passage: " + text, add_special_tokens=True)
        )
    return lengths


def mean_metric(wide: dict, keys: list[tuple[str, str]], system: str,
                metric: str) -> float:
    if not keys:
        return float("nan")
    return float(np.mean([wide[k][system][metric] for k in keys]))


def rank_bin(rank: int | None) -> str:
    if rank is None:
        return "absent"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    if rank <= 200:
        return "51-200"
    if rank <= 1000:
        return "201-1000"
    return ">1000"


def first_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.loads(f.readline())


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def add_title_page(document: Document) -> None:
    title = document.add_heading(
        "Phase 2-3 Report: Bangla/English Retrieval Corpus and "
        "Dense/Hybrid Evaluation", level=0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        "Token-efficient Bangla/English RAG - corpus construction, chunking, "
        "BM25, dense retrieval, hybrid RRF, significance tests, and error analysis"
    )
    run.italic = True
    subtitle.add_run("\n\nGenerated from the project's result artifacts; "
                     "no retrieval and no API call was run for this report.")
    add_table(
        document,
        ["Property", "Value"],
        [
            ["Report scope", "Phase 2 (corpus and question sets) and Phase 3 (retrieval)"],
            ["Project", "Bangla/English token-efficient RAG"],
            ["Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")],
            ["Git HEAD at build", git_commit()],
            ["Evaluation sets", "MIRACL dev (n=411); TyDi QA validation (n=113)"],
            ["Retrieval units", "104,045 units over 101,870 parent passages"],
            ["Dense model", "intfloat/multilingual-e5-small (384-d, fp16)"],
        ],
        font_size=9,
    )


def add_executive_summary(document: Document, data: dict) -> None:
    document.add_heading("Executive summary", level=1)
    add_bullet(
        document,
        "Phase 2 built a 101,870-parent retrieval corpus: 100,000 MIRACL "
        "Bengali passages (13,705 judged plus 86,295 seeded random non-judged) "
        "and 1,870 unique TyDi QA Bengali contexts. All four question sets "
        "have 100% gold coverage."
    )
    add_bullet(
        document,
        "Phase 3 split only the 1,501 passages longer than e5's 512-token "
        "limit into 3,676 chunks, giving 104,045 retrieval units. Every other "
        "passage stayed whole; all units retain parent IDs."
    )
    add_bullet(
        document,
        "Dense e5-small and hybrid RRF both beat BM25 on R@10 and nDCG@10 "
        "on both reporting sets. On MIRACL dev, the paired R@10 advantage over "
        "BM25 is about 0.19-0.20; on TyDi validation it is about 0.33. The "
        "95% bootstrap intervals exclude zero."
    )
    add_bullet(
        document,
        "Hybrid is the best ranker on MIRACL dev for MRR and nDCG@10 "
        "(significantly above both BM25 and dense), while R@10 is statistically "
        "tied with dense. On TyDi validation, dense is significantly better "
        "than hybrid on MRR/nDCG@10, while R@10 is tied."
    )
    add_bullet(
        document,
        "Splitting over-length passages is necessary: on the over-length-gold "
        "slice, split chunks beat a single truncated parent by 0.08-0.15 R@10 "
        "on the non-degenerate sets."
    )
    add_bullet(
        document,
        "Short gold passages are not a failure mode in these reporting sets: "
        "the Phase 2 is_short flag matches no gold passage, and dense has zero "
        "top-10 misses on <50-token gold queries."
    )
    add_bullet(
        document,
        "No BGE-M3, compression, or external API call was run for Phase 2-3 "
        "or for this report."
    )


def add_datasets(document: Document, data: dict) -> None:
    manifest = data["manifest"]
    phase2 = data["phase2_stats"]
    first_miracl = data["first_miracl"]
    first_tydi = data["first_tydi"]

    document.add_page_break()
    document.add_heading("1. Datasets used", level=1)
    document.add_paragraph(
        "Phase 1 used FLORES-200 for tokenizer fertility and is reported "
        "separately. This report covers Phase 2 and Phase 3, whose retrieval "
        "data are MIRACL Bengali and TyDi QA GoldP Bengali."
    )

    document.add_heading("1.1 Project dataset context", level=2)
    add_table(
        document,
        ["Dataset", "Role in project", "Size / split", "Source and license"],
        [
            ["FLORES-200 (ben_Beng + eng_Latn)",
             "Phase 1 tokenizer fertility; not used in Phase 2-3 retrieval",
             "997 aligned sentence pairs (dev)",
             "Meta AI NLLB release; archive as downloaded; license not bundled"],
            ["MIRACL Bengali",
             "Phase 2-3 retrieval corpus; query sets for tuning/reporting",
             "297,265 source passages; 100,000 selected; train 1,631 / dev 411 questions",
             "Apache-2.0 MIRACL release; text derived from bnwiki-20190201 "
             "(verify exact CC BY-SA version)"],
            ["TyDi QA GoldP Bengali",
             "Phase 2-3 retrieval corpus and secondary evaluation set",
             "1,870 unique contexts; train 2,390 / validation 113 questions",
             "Apache-2.0 TyDi QA release"],
            ["data/custom_questions.csv",
             "Small manually curated sanity set; not used in Phase 3 metrics",
             "6 rows (4 answerable, 2 unanswerable)",
             "User-provided project file; never rewritten"],
        ],
        font_size=8,
    )

    document.add_heading("1.2 Phase 2 corpus composition", level=2)
    m = manifest["miracl"]
    t = manifest["tydi"]
    add_table(
        document,
        ["Property", "Value"],
        [
            ["Seed", str(manifest["seed"])],
            ["MIRACL source passages", f"{m['corpus_total_passages']:,}"],
            ["MIRACL judged in dev+train qrels", f"{m['judged_unique']:,}"],
            ["MIRACL non-judged available", f"{m['non_judged_available']:,}"],
            ["MIRACL random non-judged sample", f"{m['random_sampled']:,}"],
            ["MIRACL selected / written", f"{m['written']:,}"],
            ["TyDi unique Bengali contexts", f"{t['unique_contexts_written']:,}"],
            ["TyDi rows contributing contexts",
             f"train {t['from_train_rows']:,}; validation {t['from_validation_rows']:,}"],
            ["Combined parent passages", f"{manifest['combined_total']:,}"],
            ["Short passages kept and flagged (<20 stripped chars)",
             f"{m['short_passages'] + t['short_contexts']:,} "
             f"(MIRACL {m['short_passages']:,}; TyDi {t['short_contexts']:,})"],
        ],
        font_size=8.5,
    )

    document.add_heading("1.3 Question sets", level=2)
    rows = []
    for name, info in phase2["questions"].items():
        rows.append([
            name.replace("_", " "),
            f"{info['questions']:,}",
            f"{info['with_gold']:,}",
            "tuning only" if name.endswith("train") else "reporting only",
            f"data/processed/{Path(info['file']).name}",
        ])
    add_table(
        document,
        ["Question set", "Questions", "With gold", "Purpose", "File"],
        rows,
        font_size=8.5,
    )
    document.add_paragraph(
        "Gold coverage is 100% in every set. MIRACL train and TyDi train are "
        "used only for hyperparameter selection; MIRACL dev and TyDi validation "
        "are held out for reporting."
    )

    document.add_heading("1.4 Source files and licenses", level=2)
    add_table(
        document,
        ["Source", "Files", "License"],
        [
            ["MIRACL corpus",
             "data/raw/miracl_bn/corpus/docs-0.jsonl.gz",
             "Apache-2.0 release; text from bnwiki-20190201 (verify CC BY-SA)"],
            ["MIRACL qrels/topics",
             "data/raw/miracl_bn/qrels/*.tsv; data/raw/miracl_bn/topics/*.tsv",
             "Apache-2.0"],
            ["TyDi QA GoldP",
             "data/raw/tydiqa_goldp/train.parquet; validation.parquet",
             "Apache-2.0"],
            ["FLORES-200 (Phase 1)",
             "data/raw/flores200_dataset/; flores200_dataset.tar.gz",
             "License not bundled in archive; unverified"],
        ],
        font_size=8,
    )

    document.add_heading("1.5 Dataset examples (verbatim)", level=2)
    add_table(
        document,
        ["Dataset / split", "Question", "Gold evidence"],
        [
            ["MIRACL dev", first_miracl["question"],
             "passage ID " + ", ".join(first_miracl["gold_passage_ids"])],
            ["TyDi validation", first_tydi["question"],
             "answers: " + "; ".join(first_tydi.get("gold_answers", [])) +
             "; passage ID " + ", ".join(first_tydi["gold_passage_ids"])],
        ],
        font_size=8,
    )


def add_methods(document: Document, data: dict) -> None:
    manifest = data["manifest"]
    chunking = data["chunking"]
    chunk_tokens = data["chunk_tokens"]
    bm25 = data["bm25_config"]
    summary = data["summary"]
    embed_checks = data["embed_checks"]
    embed_timing = data["embed_timing"]
    best_hybrid = data["best_hybrid"]

    document.add_page_break()
    document.add_heading("2. Methods", level=1)

    document.add_heading("2.1 Corpus construction (Phase 2)", level=2)
    add_number(
        document,
        "Loaded the official MIRACL Bengali corpus (297,265 passages) and the "
        "official dev+train qrels/topics."
    )
    add_number(
        document,
        "Computed the union of judged passages in dev+train qrels: 13,705 "
        "unique passage IDs."
    )
    add_number(
        document,
        "Combined the judged passages with a seeded random sample of 86,295 "
        "non-judged passages (seed 42), producing exactly 100,000 MIRACL "
        "passages."
    )
    add_number(
        document,
        "Deduplicated all TyDi QA Bengali GoldP contexts by content hash and "
        "wrote 1,870 unique contexts with tydi_ prefixed IDs."
    )
    add_number(
        document,
        "Kept every selected passage as a parent passage; the combined corpus "
        "has 101,870 parents. Passages shorter than 20 stripped characters "
        "were kept and flagged is_short, not deleted."
    )
    add_number(
        document,
        "Built question sets from the official files only: MIRACL train/dev "
        "and TyDi train/validation. No questions or gold labels were invented "
        "or rewritten."
    )

    document.add_heading("2.2 Retrieval units and chunking (Phase 3)", level=2)
    document.add_paragraph(
        "Only parents whose e5 token length exceeds 512 (including the "
        "passage: prefix and special tokens) are split. All other parents stay "
        "whole. Splitting is sentence-aware on the Bangla danda (।) and "
        "newlines, with one-sentence overlap and a 400-e5-token ceiling. A "
        "single sentence that alone exceeds the limit is split by words as a "
        "fallback. Every unit keeps its parent_id, chunk_index, n_chunks, and "
        "is_chunked flags."
    )
    cp = chunking["chunks_per_parent"]
    ct = chunking["new_chunk_tokens"]
    add_table(
        document,
        ["Chunking statistic", "Value"],
        [
            ["Parent passages", f"{manifest['combined_total']:,}"],
            ["Whole passages kept", f"{chunking['whole_passages']:,}"],
            ["Over-length parents", f"{chunking['overlength_parents']:,}"],
            ["Over-length by source",
             f"MIRACL {chunking['overlength_by_source'].get('miracl', 0):,}; "
             f"TyDi {chunking['overlength_by_source'].get('tydi', 0):,}"],
            ["New chunks", f"{chunking['new_chunks']:,}"],
            ["New chunks by source",
             f"MIRACL {chunking['new_chunks_by_source'].get('miracl', 0):,}; "
             f"TyDi {chunking['new_chunks_by_source'].get('tydi', 0):,}"],
            ["Net added units", f"{chunking['net_added_units']:,}"],
            ["Total retrieval units", f"{chunking['total_units']:,}"],
            ["Chunks per over-length parent",
             f"min {cp['min']}, median {cp['median']:.1f}, p95 {cp['p95']:.1f}, "
             f"max {cp['max']}, mean {cp['mean']:.2f}"],
            ["New chunk e5 tokens",
             f"min {ct['min']}, median {ct['median']:.1f}, p95 {ct['p95']:.1f}, "
             f"max {ct['max']}, mean {ct['mean']:.1f}"],
            ["Chunks <=400 tokens", f"{ct['share_le_400']:.1%}"],
            ["Oversized-sentence fallbacks", str(chunking["oversized_sentences"])],
            ["No-overlap boundaries", str(chunking["no_overlap_boundaries"])],
            ["Post-fix splits", str(chunking["post_fix_splits"])],
            ["Over-length parents with one chunk",
             str(chunking["overlength_with_one_chunk"])],
        ],
        font_size=8.5,
    )
    document.add_paragraph(
        "Parent-level evaluation counts a parent once: its score is the maximum "
        "over all of its units. This applies to BM25, dense, and hybrid alike."
    )

    document.add_heading("2.3 BM25 baseline", level=2)
    add_table(
        document,
        ["Property", "Value"],
        [
            ["Units file", bm25["units_file"]],
            ["Units", f"{bm25['n_units']:,}"],
            ["Parents", f"{bm25['n_parents']:,}"],
            ["Vocabulary", f"{bm25['vocab_size']:,}"],
            ["Average document length", f"{bm25['avgdl']:.3f} tokens"],
            ["Tokenization", bm25["tokenization"]],
            ["Indexed field", bm25["indexed_field"]],
            ["IDF", bm25["bm25"]["idf"]],
            ["Score", bm25["bm25"]["score"]],
            ["Query term frequency", bm25["bm25"]["query_term_frequency"]],
            ["Zero-score documents", bm25["bm25"]["zero_score_docs"]],
            ["Tuning set", "MIRACL train only"],
            ["Selection metric", bm25["selection_metric"]],
            ["Best k1 / b", f"{bm25['best']['k1']} / {bm25['best']['b']}"],
            ["Grid",
             f"k1 {bm25['grid']['k1']}; b {bm25['grid']['b']}"],
            ["Index build (cached)", f"{bm25['build_seconds']:.1f} s"],
            ["Total wall time", f"{bm25['total_seconds']:.1f} s"],
            ["Peak RSS", f"{bm25['peak_rss_gib']:.2f} GiB"],
        ],
        font_size=8.5,
    )
    add_caption(
        document,
        "BM25 train nDCG@10 at the selected parameters: "
        f"{data['bm25_best_train']['ndcg@10']:.4f}; train MRR "
        f"{data['bm25_best_train']['mrr']:.4f}; train R@10 "
        f"{data['bm25_best_train']['recall@10']:.4f}."
    )

    document.add_heading("2.4 Dense retrieval", level=2)
    checks = embed_checks["checks"]
    add_table(
        document,
        ["Property", "Value"],
        [
            ["Model", embed_checks["model"]],
            ["Embedding dimension", str(checks["dim"])],
            ["Device / dtype / batch", "MPS / fp16 / 16"],
            ["Max sequence length", str(embed_checks["max_seq_length"])],
            ["Pooling", embed_checks["pooling"]],
            ["Normalization", embed_checks["normalization"]],
            ["Prefixes",
             f"passage: (units); query: (queries)"],
            ["Index", "FAISS IndexFlatIP (inner product on L2-normalized vectors)"],
            ["Indexed units", f"{embed_checks['units']['n_units']:,}"],
            ["Unit encode time (after resume)",
             f"{embed_checks['units']['encode_seconds']:.1f} s "
             f"(started at {embed_checks['units']['start_count']:,})"],
            ["Truncated-parent encode time",
             f"{embed_checks['truncated']['encode_seconds']:.1f} s "
             f"for {embed_checks['truncated']['n_truncated']:,} parents"],
            ["Query encode time (total)",
             f"{sum(v['encode_seconds'] for v in embed_checks['queries'].values()):.1f} s"],
            ["Non-finite values", str(checks["nonfinite_values"])],
            ["Vectors with norm <0.5", str(checks["vectors_with_norm_<0.5"])],
            ["Determinism check",
             f"n={checks['determinism_check_n']}, "
             f"max abs diff {checks['determinism_max_abs_diff']:.2e}, "
             f"min cosine {checks['determinism_min_cosine']:.6f}"],
            ["Peak RSS", f"{embed_checks['peak_rss_gib']:.2f} GiB"],
            ["Torch", embed_timing["torch"]],
        ],
        font_size=8.5,
    )
    document.add_paragraph(
        f"The timing probe embedded {embed_timing['n_timed']:,} units in "
        f"{embed_timing['encode_seconds_1000']:.3f} s "
        f"({embed_timing['throughput_units_per_s']:.1f} units/s) on MPS fp16. "
        "The full run was resumed from a checkpoint; the saved actual unit "
        f"encode time is {embed_checks['units']['encode_seconds']:.1f} s, plus "
        f"{embed_checks['truncated']['encode_seconds']:.1f} s for the "
        f"{embed_checks['truncated']['n_truncated']:,} truncated parents used "
        "in the ablation."
    )

    document.add_heading("2.5 Hybrid retrieval (weighted RRF)", level=2)
    add_table(
        document,
        ["Property", "Value"],
        [
            ["Fusion", "Weighted reciprocal rank fusion: "
                       "score(d) = sum_i w_i / (k + rank_i(d))"],
            ["Fusion depth", f"top {summary['hybrid']['fuse_depth']} from each list"],
            ["Tuned on", summary["hybrid"]["tuned_on"]],
            ["RRF k", str(summary["hybrid"]["rrf_k"])],
            ["BM25 weight", str(summary["hybrid"]["w_bm25"])],
            ["Dense weight", str(summary["hybrid"]["w_dense"])],
            ["Grid size",
             f"{len(data['hybrid_tuning']):,} combinations",
             ],
            ["Best train nDCG@10", f"{best_hybrid['ndcg@10']:.4f}"],
            ["Best train MRR", f"{best_hybrid['mrr']:.4f}"],
            ["Best train R@10", f"{best_hybrid['recall@10']:.4f}"],
        ],
        font_size=8.5,
    )
    add_caption(
        document,
        "RRF k grid: {1,2,3,5,8,10,15,20,30,40,60,80,100}; BM25 weight grid: "
        "0.00-1.00 in steps of 0.05 (dense weight = 1 - BM25 weight). "
        "Selection: nDCG@10, then MRR, then R@10."
    )

    document.add_heading("2.6 Evaluation protocol", level=2)
    add_table(
        document,
        ["Item", "Protocol"],
        [
            ["Reporting sets", "MIRACL dev (n=411); TyDi validation (n=113)"],
            ["Unit of evaluation", "Parent passage, counted once at its best unit"],
            ["Recall@k", "1 if any gold parent is in the top k; else 0"],
            ["MRR", "1 / rank of the first gold parent (0 if absent)"],
            ["nDCG@10", "Binary relevance; ideal DCG uses min(|gold|, 10)"],
            ["Confidence intervals", "1,000 query bootstrap draws; seed 42; 95%"],
            ["Paired significance", "10,000 paired query bootstrap draws; seed 42"],
            ["Multiple comparisons", "Bonferroni over 3 system pairs per "
                                     "(dataset, metric): alpha = 0.05/3 = 0.0167"],
            ["TyDi secondary metric", "Answer-string hit in any chunk of the "
                                      "top-k retrieved parents"],
            ["Ablation", "Split chunks vs one truncated parent on queries with "
                         "over-length gold"],
            ["Tuning isolation", "MIRACL train only; no dev/validation tuning"],
        ],
        font_size=8.5,
    )
    document.add_paragraph(
        "For TyDi answer-string scoring, both the gold answers and passages are "
        "normalized with Unicode NFC, lowercasing, and whitespace collapsing. "
        "A query is a hit at k if any retrieved parent's chunks contain any "
        "gold answer string."
    )

    document.add_heading("2.7 Software, hardware, and runtime", level=2)
    add_table(
        document,
        ["Item", "Value"],
        [
            ["Python", "3.12.13"],
            ["Platform", "macOS arm64 (Apple Silicon), MPS"],
            ["Torch", embed_timing["torch"]],
            ["Dense dtype / batch", "fp16 / 16"],
            ["BM25 total wall time", f"{bm25['total_seconds']:.1f} s"],
            ["Dense unit encode time", f"{embed_checks['units']['encode_seconds']:.1f} s (resumed)"],
            ["Dense truncated encode time", f"{embed_checks['truncated']['encode_seconds']:.1f} s"],
            ["Dense peak RSS", f"{embed_checks['peak_rss_gib']:.2f} GiB"],
            ["Full evaluation wall time", f"{summary['total_seconds']:.1f} s"],
        ],
        font_size=8.5,
    )


def add_results(document: Document, data: dict) -> None:
    metrics = data["metrics"]
    tydi_ans = data["tydi_answer"]
    sig = data["significance"]
    ablation = data["ablation"]
    chunk_tokens = data["chunk_tokens"]
    chunking = data["chunking"]
    wide = data["wide"]
    gold_tok_len = data["gold_tok_len"]

    document.add_page_break()
    document.add_heading("3. Results", level=1)

    document.add_heading("3.1 Main retrieval results (strict gold passage IDs)", level=2)
    for dataset in DATASETS:
        document.add_heading(
            "MIRACL dev (n=411)" if dataset == "miracl_dev"
            else "TyDi validation (n=113)", level=3
        )
        sub = metrics[metrics["dataset"] == dataset]
        rows = []
        for system in SYSTEMS:
            r = sub[sub["system"] == system].iloc[0]
            rows.append([
                system,
                f4(r["recall@1"]), f4(r["recall@3"]), f4(r["recall@5"]),
                f4(r["recall@10"]), f4(r["mrr"]), f4(r["ndcg@10"]),
            ])
        add_table(
            document,
            ["System", "R@1", "R@3", "R@5", "R@10", "MRR", "nDCG@10"],
            rows,
            font_size=9,
        )
        add_caption(
            document,
            "Means only. 95% query-bootstrap CIs are in "
            "results/phase3_dense_metrics.csv. BM25/dense MRR uses the full "
            "parent ranking; hybrid MRR uses the fused top-100+top-100 list."
        )

    document.add_heading("3.2 Key metrics with 95% bootstrap CIs", level=2)
    rows = []
    for dataset in DATASETS:
        sub = metrics[metrics["dataset"] == dataset]
        for system in SYSTEMS:
            r = sub[sub["system"] == system].iloc[0]
            rows.append([
                "MIRACL dev" if dataset == "miracl_dev" else "TyDi val",
                system,
                fci(r, "recall@10"),
                fci(r, "mrr"),
                fci(r, "ndcg@10"),
            ])
    add_table(
        document,
        ["Dataset", "System", "R@10 [95% CI]", "MRR [95% CI]",
         "nDCG@10 [95% CI]"],
        rows,
        font_size=8,
    )

    document.add_heading("3.3 TyDi validation: strict ID vs answer-string", level=2)
    rows = []
    for system in SYSTEMS:
        strict = metrics[
            (metrics["dataset"] == "tydi_validation")
            & (metrics["system"] == system)
        ].iloc[0]
        ans = tydi_ans[
            (tydi_ans["dataset"] == "tydi_validation_answerstring")
            & (tydi_ans["system"] == system)
        ].iloc[0]
        rows.append([
            system,
            f4(strict["recall@10"]),
            f4(ans["recall@10"]),
            f"{ans['recall@10'] - strict['recall@10']:+.4f}",
            f4(strict["mrr"]),
            f4(ans["mrr"]),
        ])
    add_table(
        document,
        ["System", "Strict R@10", "Answer-string R@10", "Delta",
         "Strict MRR", "Answer-string MRR"],
        rows,
        font_size=8.5,
    )
    add_caption(
        document,
        "The answer-string rule checks whether any chunk of a retrieved parent "
        "contains a gold TyDi answer string. It is more lenient than exact gold "
        "passage ID matching."
    )

    document.add_heading("3.4 RRF tuning", level=2)
    best = data["best_hybrid"]
    add_table(
        document,
        ["rrf_k", "w_BM25", "w_dense", "Train n", "Train R@10", "Train MRR",
         "Train nDCG@10"],
        [[
            str(int(best["rrf_k"])), f"{best['w_bm25']:.2f}",
            f"{best['w_dense']:.2f}", str(int(best["n"])),
            f4(best["recall@10"]), f4(best["mrr"]), f4(best["ndcg@10"]),
        ]],
        font_size=9,
    )
    document.add_paragraph(
        "The best configuration is interior to the grid (k=5, equal weights), "
        "not a grid boundary. The same configuration is used for both "
        "reporting sets; no dev or TyDi validation tuning was performed."
    )

    document.add_heading("3.5 Paired bootstrap significance (10,000 resamples)", level=2)
    document.add_paragraph(
        "Differences are A - B. The 95% interval is the percentile bootstrap "
        "interval; the Bonferroni interval is 98.33% for the three-pair family "
        "within each (dataset, metric). The three tests reuse the same queries "
        "and are correlated, so Bonferroni is conservative."
    )
    for dataset in DATASETS:
        document.add_heading(
            "MIRACL dev" if dataset == "miracl_dev" else "TyDi validation",
            level=3,
        )
        sub = sig[sig["dataset"] == dataset]
        rows = []
        for _, r in sub.iterrows():
            rows.append([
                r["metric"],
                f"{r['system_a']} vs {r['system_b']}",
                f"{r['diff']:+.4f}",
                f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]",
                "yes" if r["excludes_zero_95"] else "no",
                f"[{r['ci_bonf_low']:+.4f}, {r['ci_bonf_high']:+.4f}]",
                "yes" if r["excludes_zero_bonf"] else "no",
            ])
        add_table(
            document,
            ["Metric", "A vs B", "Delta", "95% CI", "Excl. 0",
             "Bonf. 98.33% CI", "Excl. 0"],
            rows,
            font_size=7.5,
        )
    document.add_paragraph(
        "The unadjusted percentile p-value is in "
        "results/phase3_significance.csv. It should not be read as family-wise."
    )

    document.add_heading("3.6 Truncated-vs-split ablation", level=2)
    rows = []
    for _, r in ablation.iterrows():
        rows.append([
            r["dataset"],
            str(int(r["n_queries"])),
            fci(r, "split_R@10_over_gold"),
            fci(r, "trunc_R@10_over_gold"),
            f"{r['delta_over_gold']:+.4f}",
        ])
    add_table(
        document,
        ["Dataset", "n queries", "Split R@10 (over-length gold)",
         "Truncated R@10 (over-length gold)", "Delta"],
        rows,
        font_size=8,
    )
    add_caption(
        document,
        "Over-length gold means the query has at least one over-length gold "
        "parent. The split condition uses the normal 104,045-unit index; the "
        "truncated condition uses one 512-token embedding per over-length "
        "parent (101,870 units)."
    )

    document.add_heading("3.7 Error analysis", level=2)
    add_heading39 = document.add_heading("Top-10 wins and losses", level=3)
    rows = []
    for dataset in DATASETS:
        keys = sorted(k for k in wide if k[0] == dataset)
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
            "MIRACL dev" if dataset == "miracl_dev" else "TyDi val",
            str(len(bm25_only)), str(len(dense_only)), str(len(hybrid_miss)),
            str(len(dense_miss)), str(len(bm25_miss)),
        ])
    add_table(
        document,
        ["Dataset", "BM25 hit / Dense miss", "Dense hit / BM25 miss",
         "Hybrid miss", "Dense miss", "BM25 miss"],
        rows,
        font_size=8.5,
    )

    document.add_heading("Gold rank when a system misses top-10", level=3)
    bins = ["11-20", "21-50", "51-200", "201-1000", ">1000", "absent"]
    rows = []
    for dataset in DATASETS:
        keys = sorted(k for k in wide if k[0] == dataset)
        for system in SYSTEMS:
            miss = [k for k in keys if wide[k][system]["recall@10"] == 0]
            counts = {b: 0 for b in bins}
            for k in miss:
                counts[rank_bin(wide[k][system]["first_gold_rank"])] += 1
            rows.append([
                "MIRACL dev" if dataset == "miracl_dev" else "TyDi val",
                system, str(len(miss)),
                *[str(counts[b]) for b in bins],
            ])
    add_table(
        document,
        ["Dataset", "System", "n missed", *bins],
        rows,
        font_size=8,
    )
    add_caption(
        document,
        "The full 8-bin table (11-20, 21-50, 51-100, 101-200, 201-500, "
        "501-1000, >1000, absent) is in results/phase3_error_analysis.md. "
        "Absent means the gold parent is not in the returned ranking: BM25 "
        "only ranks positive-scoring parents; hybrid fuses only the top-100 "
        "lists."
    )

    document.add_heading("Over-length gold subgroup", level=3)
    rows = []
    for dataset in DATASETS:
        keys = sorted(k for k in wide if k[0] == dataset)
        for value in (False, True):
            sub = [k for k in keys
                   if bool(wide[k]["BM25"]["has_overlength_gold"]) is value]
            label = "yes" if value else "no"
            rows.append([
                "MIRACL dev" if dataset == "miracl_dev" else "TyDi val",
                label, str(len(sub)),
                f4(mean_metric(wide, sub, "BM25", "recall@10")),
                f4(mean_metric(wide, sub, "Dense (e5-small)", "recall@10")),
                f4(mean_metric(wide, sub, "Hybrid (RRF)", "recall@10")),
            ])
    add_table(
        document,
        ["Dataset", "Over-length gold", "n", "BM25 R@10", "Dense R@10",
         "Hybrid R@10"],
        rows,
        font_size=8.5,
    )

    document.add_heading("Short-gold sensitivity", level=3)
    document.add_paragraph(
        "The Phase 2 is_short flag is stripped character length <20. No gold "
        "passage in either reporting set has that flag, so the table also uses "
        "e5-token thresholds for gold parents."
    )
    rows = []
    for label, threshold in [
        ("Phase 2 is_short (char <20)", None),
        ("<20 e5 tokens", 20),
        ("<30 e5 tokens", 30),
        ("<50 e5 tokens", 50),
    ]:
        for dataset in DATASETS:
            keys = sorted(k for k in wide if k[0] == dataset)
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
            n_sub = len(sub)
            rows.append([
                label,
                "MIRACL dev" if dataset == "miracl_dev" else "TyDi val",
                str(n_sub),
                f4(mean_metric(wide, sub, "BM25", "recall@10")) if n_sub else "n/a",
                f4(mean_metric(wide, sub, "Dense (e5-small)", "recall@10")) if n_sub else "n/a",
                f4(mean_metric(wide, sub, "Hybrid (RRF)", "recall@10")) if n_sub else "n/a",
                str(dense_misses),
            ])
    add_table(
        document,
        ["Definition", "Dataset", "n queries", "BM25 R@10", "Dense R@10",
         "Hybrid R@10", "Dense misses"],
        rows,
        font_size=8,
    )
    document.add_paragraph(
        "There are no dense top-10 failures on short-gold queries under any "
        "tested definition; the short-passage slice is not represented among "
        "these gold sets."
    )

    document.add_heading("3.8 Token-length profile of retrieval units", level=2)
    a = chunk_tokens["all_units"]
    c = chunk_tokens["split_chunks_only"]
    ct = chunking["new_chunk_tokens"]
    add_table(
        document,
        ["Slice", "Units", "min", "median", "p95", "max", " <20 tokens",
         " <10 tokens", " <5 tokens"],
        [
            ["All units", f"{chunk_tokens['total_units']:,}",
             str(a["min"]), f"{a['median']:.1f}", f"{a['p95']:.1f}",
             str(a["max"]), f"{a['lt_20']:,} ({a['lt_20']/chunk_tokens['total_units']:.2%})",
             f"{a['lt_10']:,}", f"{a['lt_5']:,}"],
            ["Split chunks only", f"{chunk_tokens['split_chunks']:,}",
             str(c["min"]), f"{c['median']:.1f}", f"{ct['p95']:.1f}",
             str(ct["max"]), f"{c['lt_20']:,} ({c['lt_20']/max(chunk_tokens['split_chunks'],1):.2%})",
             f"{c['lt_10']:,}", f"{c['lt_5']:,}"],
        ],
        font_size=8,
    )


def add_interpretation(document: Document) -> None:
    document.add_heading("4. Interpretation", level=1)
    add_bullet(
        document,
        "Dense retrieval is the strongest standalone system on these sets. "
        "It significantly improves R@10 and nDCG@10 over BM25 on both "
        "reporting sets. Its MRR advantage over BM25 is not significant on "
        "MIRACL dev but is large and significant on TyDi validation."
    )
    add_bullet(
        document,
        "Equal-weight RRF is the best ranker on MIRACL dev for MRR and "
        "nDCG@10, but it does not improve R@10 over dense. On TyDi, dense is "
        "significantly better than hybrid on MRR/nDCG@10, while R@10 is tied. "
        "The likely reason is that BM25 is much weaker on TyDi and equal "
        "weights dilute dense's strong first hit."
    )
    add_bullet(
        document,
        "BM25-only wins are rare (12 queries across both sets vs 132 "
        "dense-only wins), but they exist. They are mostly exact-name/date "
        "queries where lexical matching is decisive."
    )
    add_bullet(
        document,
        "Dense misses are mostly shallow: 32 of 45 dense misses across both "
        "sets are at rank 11-50 (24 of 33 on MIRACL dev, 8 of 12 on TyDi). "
        "Hybrid inherits similar misses because it only fuses the top 100 lists."
    )
    add_bullet(
        document,
        "Over-length passages are not a dense weakness in the main metric, "
        "but the ablation shows that truncating them costs 0.08-0.15 R@10 on "
        "the over-length-gold slice. Sentence-aware chunking is the safer "
        "choice."
    )
    add_bullet(
        document,
        "TyDi strict gold IDs undercount success: 6 strict-ID misses are "
        "answer-string hits within the dense top-10. The answer-string rule "
        "is a useful secondary metric, not a replacement for strict IDs."
    )


def add_limitations(document: Document) -> None:
    document.add_heading("5. Limitations and caveats", level=1)
    add_bullet(
        document,
        "The 100k MIRACL slice is easier than the full 297k-passage corpus: "
        "it mixes all judged passages with a random sample. Absolute scores "
        "should not be read as full-corpus estimates."
    )
    add_bullet(
        document,
        "TyDi is scored two ways: strict gold passage ID and answer-string "
        "hit. The answer-string rule is more lenient and checks all chunks of "
        "retrieved parents."
    )
    add_bullet(
        document,
        "Validation sets are small: MIRACL dev n=411 and TyDi validation "
        "n=113. Subgroups are much smaller (TyDi over-length gold n=4), so "
        "their CIs and example counts are unstable."
    )
    add_bullet(
        document,
        "The Phase 2 is_short flag (stripped characters <20) matches no gold "
        "passage in either reporting set. The short-gold sensitivity table "
        "adds e5-token thresholds, but even <50 tokens has no dense misses."
    )
    add_bullet(
        document,
        "RRF hyperparameters (k=5, w_BM25=0.5) were tuned on MIRACL train "
        "only. There is no guarantee that the same weights are optimal for "
        "TyDi or for other corpora."
    )
    add_bullet(
        document,
        "MRR for BM25/dense is over the full parent ranking; hybrid MRR is "
        "over the fused top-100+top-100 candidate list. This depth difference "
        "matters only for a handful of queries but should be kept in mind."
    )
    add_bullet(
        document,
        "Paired bootstrap resamples queries, not passages. It measures "
        "uncertainty over this query sample, not over the corpus."
    )
    add_bullet(
        document,
        "Only multilingual-e5-small was evaluated as the dense model. "
        "BGE-M3, compression, and external API generation/judging were not "
        "run."
    )
    add_bullet(
        document,
        "The MIRACL text derives from bnwiki-20190201; the exact CC BY-SA "
        "version of the Wikipedia text should be verified before redistribution."
    )


def add_reproducibility(document: Document) -> None:
    document.add_heading("6. Reproducibility", level=1)
    document.add_paragraph(
        "The pipeline is deterministic given the same input files and seed. "
        "Large raw/processed/cache/embedding artifacts are gitignored and are "
        "regenerated by the commands below."
    )
    add_code(document, [
        "cd bangla-rag",
        "python3 -m venv .venv",
        ".venv/bin/pip install -r requirements.txt",
        "",
        "# Phase 2: corpus + question sets",
        ".venv/bin/python experiments/build_corpus.py",
        ".venv/bin/python experiments/check_token_lengths.py",
        "",
        "# Phase 3: chunking, BM25, dense embeddings, evaluation",
        ".venv/bin/python experiments/build_chunks.py",
        ".venv/bin/python experiments/phase3_bm25.py",
        ".venv/bin/python experiments/phase3_embed_dense.py",
        ".venv/bin/python experiments/phase3_dense_hybrid_eval.py",
        "",
        "# Follow-up analysis and this report",
        ".venv/bin/python experiments/phase3_significance_error_analysis.py",
        ".venv/bin/python experiments/make_phase2_phase3_report.py",
        ".venv/bin/python -m pytest tests -q",
    ])
    document.add_paragraph(
        "The dense embedding step sets HF_HUB_OFFLINE=1 and reads the cached "
        "multilingual-e5-small model. No generation or judging API is used."
    )


def add_output_files(document: Document) -> None:
    document.add_heading("7. Output files", level=1)
    add_table(
        document,
        ["File", "Contents"],
        [
            ["data/processed/corpus.jsonl",
             "101,870 parent passages (MIRACL + TyDi)"],
            ["data/processed/corpus_chunks.jsonl",
             "104,045 retrieval units with parent_id / chunk_index / is_chunked"],
            ["data/processed/corpus_manifest.json",
             "Corpus composition, sources, licenses, seed"],
            ["data/processed/phase2_stats.json",
             "Corpus size and question-set counts"],
            ["results/phase3_chunking_stats.json",
             "Over-length counts, chunks per parent, chunk token stats"],
            ["results/phase3_chunk_token_stats.json",
             "Token-length distribution for all units and split chunks"],
            ["results/phase3_bm25_config.json",
             "BM25 tokenization, scoring, grid, best parameters, runtime"],
            ["results/phase3_bm25_metrics.csv",
             "BM25 held-out metrics (standalone baseline)"],
            ["results/phase3_bm25_grid.csv",
             "Full BM25 k1/b tuning grid on MIRACL train"],
            ["results/phase3_dense_metrics.csv",
             "Main strict parent-level metrics and 95% CIs"],
            ["results/phase3_tydi_answerstring.csv",
             "TyDi answer-string metrics and 95% CIs"],
            ["results/phase3_hybrid_tuning.csv",
             "Full RRF k/weight grid on MIRACL train"],
            ["results/phase3_ablation_truncated_vs_split.csv",
             "Truncated-vs-split dense R@10 ablation"],
            ["results/phase3_embed_checks.json",
             "Embedding shapes, non-finite checks, norms, determinism, RSS"],
            ["results/phase3_per_query.jsonl",
             "1,572 per-query records used for significance/error analysis"],
            ["results/phase3_significance.csv",
             "Paired bootstrap table (10,000 draws, 95% and Bonferroni CIs)"],
            ["results/phase3_error_analysis.md",
             "Full error tables and verbatim examples"],
            ["results/phase3_report.md",
             "Concise markdown report"],
            ["results/Phase2_Phase3_RAG_Report.docx",
             "This standalone Phase 2-3 Word report"],
            ["results/Bangla_RAG_Report_Phases_1-3.docx",
             "Combined report: existing Phase 1 report + Phase 2-3 sections"],
        ],
        font_size=7.5,
    )
    document.add_paragraph(
        "Gitignored (large/regenerable): data/raw/, data/processed/, "
        "data/cache/, data/embeddings/, logs/, and model/index caches."
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def load_data() -> dict:
    manifest = load_json(PROCESSED / "corpus_manifest.json")
    phase2_stats = load_json(PROCESSED / "phase2_stats.json")
    chunking = load_json(RESULTS / "phase3_chunking_stats.json")
    chunk_tokens = load_json(RESULTS / "phase3_chunk_token_stats.json")
    bm25_config = load_json(RESULTS / "phase3_bm25_config.json")
    summary = load_json(RESULTS / "phase3_dense_hybrid_summary.json")
    embed_checks = load_json(RESULTS / "phase3_embed_checks.json")
    embed_timing = load_json(RESULTS / "phase3_embedding_timing_chunks.json")
    metrics = load_csv(RESULTS / "phase3_dense_metrics.csv")
    tydi_answer = load_csv(RESULTS / "phase3_tydi_answerstring.csv")
    significance = load_csv(RESULTS / "phase3_significance.csv")
    ablation = load_csv(RESULTS / "phase3_ablation_truncated_vs_split.csv")
    hybrid_tuning = load_csv(RESULTS / "phase3_hybrid_tuning.csv")
    bm25_grid = load_csv(RESULTS / "phase3_bm25_grid.csv")
    wide = load_per_query()
    gold_tok_len = load_gold_token_lengths(wide)

    best_hybrid = hybrid_tuning[
        (hybrid_tuning["rrf_k"] == summary["hybrid"]["rrf_k"])
        & (hybrid_tuning["w_bm25"] == summary["hybrid"]["w_bm25"])
    ].iloc[0]
    bm25_best_train = bm25_grid[
        (bm25_grid["k1"] == bm25_config["best"]["k1"])
        & (bm25_grid["b"] == bm25_config["best"]["b"])
    ].iloc[0]

    return {
        "manifest": manifest,
        "phase2_stats": phase2_stats,
        "chunking": chunking,
        "chunk_tokens": chunk_tokens,
        "bm25_config": bm25_config,
        "summary": summary,
        "embed_checks": embed_checks,
        "embed_timing": embed_timing,
        "metrics": metrics,
        "tydi_answer": tydi_answer,
        "significance": significance,
        "ablation": ablation,
        "hybrid_tuning": hybrid_tuning,
        "bm25_grid": bm25_grid,
        "best_hybrid": best_hybrid,
        "bm25_best_train": bm25_best_train,
        "wide": wide,
        "gold_tok_len": gold_tok_len,
        "first_miracl": first_jsonl(PROCESSED / "miracl_dev_questions.jsonl"),
        "first_tydi": first_jsonl(PROCESSED / "tydi_validation_questions.jsonl"),
    }


def build_standalone(data: dict) -> None:
    doc = Document()
    setup_document(doc)
    add_title_page(doc)
    add_executive_summary(doc, data)
    add_datasets(doc, data)
    add_methods(doc, data)
    add_results(doc, data)
    add_interpretation(doc)
    add_limitations(doc)
    add_reproducibility(doc)
    add_output_files(doc)
    doc.save(OUT_STANDALONE)
    print(f"[ok] wrote {OUT_STANDALONE.relative_to(PROJECT_ROOT)}")


def build_combined(data: dict) -> None:
    if not PHASE1_DOCX.exists():
        print(f"[skip] combined report: {PHASE1_DOCX.name} not found")
        return
    doc = Document(PHASE1_DOCX)
    setup_document(doc)
    # Update the Phase 1 title for the combined document.
    if doc.paragraphs:
        doc.paragraphs[0].text = (
            "Bangla/English Token-Efficient RAG: Phases 1-3 Report"
        )
    doc.add_page_break()
    part = doc.add_heading("Part II: Phase 2 and Phase 3", level=0)
    part.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "Phase 1 (tokenizer fertility) is the preceding part. The following "
        "sections add the Phase 2 retrieval corpus/question sets and the Phase 3 "
        "chunking, BM25, dense, hybrid, significance, and error analysis."
    )
    add_executive_summary(doc, data)
    add_datasets(doc, data)
    add_methods(doc, data)
    add_results(doc, data)
    add_interpretation(doc)
    add_limitations(doc)
    add_reproducibility(doc)
    add_output_files(doc)
    doc.save(OUT_COMBINED)
    print(f"[ok] wrote {OUT_COMBINED.relative_to(PROJECT_ROOT)}")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = load_data()
    build_standalone(data)
    build_combined(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
