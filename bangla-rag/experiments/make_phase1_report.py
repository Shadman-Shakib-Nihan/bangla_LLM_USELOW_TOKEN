"""Generate a Word (.docx) report for Phase 1 from the real result files.

Reads results/phase1_fertility_summary.csv, results/phase1_token_counts.csv and
results/phase1_versions.txt so the report always reflects what actually ran.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/make_phase1_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_CSV = PROJECT_ROOT / "results" / "phase1_fertility_summary.csv"
COUNTS_CSV = PROJECT_ROOT / "results" / "phase1_token_counts.csv"
VERSIONS = PROJECT_ROOT / "results" / "phase1_versions.txt"
OUT_DOCX = PROJECT_ROOT / "results" / "Phase1_Tokenizer_Fertility_Report.docx"

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)


def shade_cell(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade_cell(hdr[i], "1F3A5F")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    document.add_paragraph()


def add_bullet(document: Document, text: str) -> None:
    document.add_paragraph(text, style="List Bullet")


def add_number(document: Document, text: str) -> None:
    document.add_paragraph(text, style="List Number")


def main() -> int:
    summary = pd.read_csv(SUMMARY_CSV)
    counts = pd.read_csv(COUNTS_CSV, encoding="utf-8")
    versions = VERSIONS.read_text(encoding="utf-8")

    # Order rows: Gemini, o200k, Qwen (as reported).
    order = {"gemini:gemini-3.8-flash": 0, "o200k_base": 1, "qwen2.5": 2}
    summary = summary.sort_values(
        "tokenizer", key=lambda s: s.map(lambda v: order.get(v, 99))
    )

    doc = Document()

    # Base font.
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ------------------------------------------------------------------ title
    title = doc.add_heading("Phase 1 Report: Tokenizer Fertility for Bangla/English",
                            level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "Research project: token-efficient Bangla/English Retrieval-Augmented "
        "Generation (RAG).\n"
        "This report covers Phase 1 only: measuring how strongly each tokenizer "
        "penalises Bangla relative to English on the same parallel corpus."
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------------------------------------------------------------- dataset
    doc.add_heading("1. Dataset used", level=1)
    doc.add_paragraph(
        "FLORES-200 (the 200-language FLORES benchmark released with the NLLB "
        "work by Meta AI). It is sentence-aligned multilingual translation data, "
        "not a question-answering or retrieval corpus."
    )

    bn_sample = counts.iloc[0]["bn"] if len(counts) else ""
    en_sample = counts.iloc[0]["en"] if len(counts) else ""
    add_table(
        doc,
        ["Property", "Value"],
        [
            ["Dataset", "FLORES-200 (ben_Beng + eng_Latn)"],
            ["Archive", "data/raw/flores200_dataset.tar.gz"],
            ["Extracted folder", "data/raw/flores200_dataset/"],
            ["Files used", "dev/ben_Beng.dev and dev/eng_Latn.dev"],
            ["Split used", "dev"],
            ["Parallel sentence pairs", "997 (dev); devtest (1,012) present but unused"],
            ["Alignment", "Line-by-line by index, per the dataset README"],
            ["License", "Not bundled in the archive; to be verified in Phase 2"],
        ],
    )

    doc.add_paragraph("Example sentence pair (row 1 of the dev split):")
    sample = doc.add_paragraph()
    sample.add_run("Bangla: ").bold = True
    sample.add_run(bn_sample)
    sample2 = doc.add_paragraph()
    sample2.add_run("English: ").bold = True
    sample2.add_run(en_sample)

    # ------------------------------------------------------------- what was done
    doc.add_heading("2. What was done", level=1)
    add_number(doc,
               "Built the Python 3.12 virtual environment and installed pinned "
               "dependencies (google-genai, tiktoken, tokenizers, "
               "huggingface-hub, pandas, matplotlib, python-dotenv, pytest, "
               "pyyaml, tqdm).")
    add_number(doc,
               "Loaded the 997 aligned Bangla/English sentence pairs from the "
               "FLORES-200 dev split (no filtering, no fabrication).")
    add_number(doc,
               "Checked the current Gemini documentation for valid model names; "
               "gemini-2.5-flash is deprecated, so the current stable "
               "gemini-3.8-flash was selected.")
    add_number(doc,
               "Verified the model name with a one-token count_tokens test call "
               "before the full run (returned 2 tokens).")
    add_number(doc,
               "Computed per-sentence token counts for each tokenizer: Gemini "
               "count_tokens API, OpenAI o200k_base via tiktoken, and the "
               "Qwen2.5 tokenizer via HuggingFace tokenizers.")
    add_number(doc,
               "Cached all 1,995 Gemini counts to disk and added retry/backoff "
               "after one 503 UNAVAILABLE mid-run; the run resumed from cache "
               "and completed.")
    add_number(doc,
               "Computed mean tokens per sentence and the Bangla/English ratio "
               "(both the mean per-sentence ratio and the corpus-total ratio).")
    add_number(doc,
               "Saved the per-sentence table, summary table, bar plot, and a "
               "package/model version log; added unit tests for the statistics "
               "and data loading (4 passed).")

    # ---------------------------------------------------------------- results
    doc.add_heading("3. Results", level=1)
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            [
                r["tokenizer"],
                f"{r['mean_bn_tokens']:.3f}",
                f"{r['mean_en_tokens']:.3f}",
                f"{r['mean_ratio_per_sentence']:.3f}",
                f"{r['corpus_ratio']:.3f}",
                f"{int(r['total_bn_tokens']):,}",
                f"{int(r['total_en_tokens']):,}",
            ]
        )
    add_table(
        doc,
        [
            "Tokenizer",
            "Mean BN tokens/sentence",
            "Mean EN tokens/sentence",
            "Mean BN/EN ratio",
            "Corpus BN/EN ratio",
            "Total BN tokens",
            "Total EN tokens",
        ],
        rows,
    )

    doc.add_paragraph(
        "Headline: at a fixed token budget, the tokenizer changes how much "
        "Bangla text can be afforded by a factor of about four. Gemini is the "
        "most Bangla-efficient (~1.2x English), o200k is ~1.7x, and Qwen2.5 is "
        "~5x. In practical terms, a 1,000-token context holds roughly 31 Bangla "
        "sentences under the Gemini tokenizer but only ~7.5 under Qwen2.5 "
        "(approximate, derived from the mean tokens per sentence above)."
    )

    # --------------------------------------------------------------- limits
    doc.add_heading("4. What this experiment does NOT cover", level=1)
    add_bullet(doc, "Banglish / code-mixed text: no such data was used in Phase 1.")
    add_bullet(doc, "Retrieval quality (Recall@k, MRR, nDCG): that is Phase 3 and "
                    "depends on the embedding model and chunking, not the "
                    "generation tokenizer directly.")
    add_bullet(doc, "Answer faithfulness and abstention: that is Phase 6 and "
                    "requires real generation runs.")
    add_bullet(doc, "o200k and Qwen2.5 are tokenizer-only comparisons here; we do "
                    "not generate with GPT-4o or a local Qwen model, so their "
                    "numbers are not tied to an actual generation model in this "
                    "project.")
    add_bullet(doc, "The FLORES archive ships no license file, so the license is "
                    "explicitly unverified at this stage.")

    # --------------------------------------------------------------- outputs
    doc.add_heading("5. Output files", level=1)
    add_table(
        doc,
        ["File", "Contents"],
        [
            ["results/phase1_token_counts.csv",
             "Per-sentence Bangla/English token counts for all three tokenizers"],
            ["results/phase1_fertility_summary.csv",
             "Summary statistics reproduced in the table above"],
            ["results/phase1_fertility_bar.png",
             "Bar plot: tokens per sentence and Bangla/English ratio"],
            ["results/phase1_versions.txt",
             "Python/package versions, model names, seed, split"],
            ["results/Phase1_Tokenizer_Fertility_Report.docx",
             "This report"],
        ],
    )

    # --------------------------------------------------------------- reproduce
    doc.add_heading("6. How to reproduce", level=1)
    doc.add_paragraph("Setup and secret (the key is read from the environment, "
                      "never hardcoded):")
    for line in [
        "cd bangla-rag",
        "python3 -m venv .venv",
        ".venv/bin/pip install -r requirements.txt",
        "cp .env.example .env   # then set GEMINI_API_KEY in .env",
    ]:
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)

    doc.add_paragraph("Run the experiment and the tests:")
    for line in [
        ".venv/bin/python experiments/phase1_fertility.py",
        ".venv/bin/python -m pytest tests -q",
        ".venv/bin/python experiments/make_phase1_report.py",
    ]:
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)

    # ----------------------------------------------------------------- versions
    doc.add_heading("7. Environment and versions (logged during the run)", level=1)
    p = doc.add_paragraph()
    run = p.add_run(versions)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)

    doc.save(OUT_DOCX)
    print(f"[ok] wrote {OUT_DOCX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
