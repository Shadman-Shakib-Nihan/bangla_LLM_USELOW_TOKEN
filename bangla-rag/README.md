# Bangla/English token-efficient RAG

Research codebase measuring how much a Bangla/English RAG pipeline can cut token
cost while preserving answer accuracy and faithfulness. The headline result is a
cost vs accuracy vs faithfulness curve across pipeline configurations.

## Environment

- macOS / Apple Silicon (M2 Pro, 8 GB RAM). Embeddings and rerankers run on
  `mps`/CPU only; generation and judging use the Gemini API.
- Python 3.12 (`.venv` inside `bangla-rag/`).

```bash
cd bangla-rag
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Secrets: generation and judging read `GEMINI_API_KEY` from the environment. The
key is never hardcoded or logged. Put it in `bangla-rag/.env` (gitignored):

```bash
cd bangla-rag
cp .env.example .env
# edit .env:  GEMINI_API_KEY=your_key_here
```

Or export it in the shell:

```bash
export GEMINI_API_KEY="..."
```

`.env`, `.venv/`, and large data/cache folders are gitignored.

## Layout

```
bangla-rag/
  data/raw/         FLORES-200, MIRACL, and TyDi QA downloads
  data/processed/   cleaned/derived data
  data/cache/       token counts, HF models, tiktoken vocab, embeddings
  src/              tokenizers.py, chunking.py, retrieval.py, compression.py,
                    generation.py, verification.py, evaluation.py, cache.py
  experiments/      one script per phase; configs in experiments/configs/
  results/          CSV/JSON tables, plots, markdown summaries, Word reports
  tests/            unit tests
```

## Phase 1: tokenizer fertility

Compares tokens-per-sentence and the Bangla/English ratio for:

- **Gemini** via `client.models.count_tokens` (requires `GEMINI_API_KEY`)
- **OpenAI o200k_base** via `tiktoken` (local)
- **Qwen2.5** tokenizer via HuggingFace `tokenizers` (local, `Qwen2.5-0.5B-Instruct`)

Data: FLORES-200 `dev` split (997 aligned Bangla/English sentences), already
downloaded to `data/raw/flores200_dataset/`.

```bash
cd bangla-rag
# full run (includes Gemini when the key is set)
.venv/bin/python experiments/phase1_fertility.py

# quick run without the API
.venv/bin/python experiments/phase1_fertility.py --limit 100 --skip-gemini
```

Config: `experiments/configs/phase1.yaml`.

Outputs:

| File | Contents |
| --- | --- |
| `results/phase1_token_counts.csv` | per-sentence Bangla/English token counts |
| `results/phase1_fertility_summary.csv` | mean tokens and Bangla/English ratios |
| `results/phase1_fertility_bar.png` | bar plot of tokens/sentence and ratio |
| `results/phase1_versions.txt` | package versions, model names, seed, split |
| `results/Phase1_Tokenizer_Fertility_Report.docx` | Word report generated from the result files |

Generate the Word report (reads the real CSVs/log, does not invent numbers):

```bash
.venv/bin/python experiments/make_phase1_report.py
```

Model name: `gemini-3.8-flash` (current stable model per
https://ai.google.dev/gemini-api/docs/models as of 2026-11-07). The brief
requires verifying the name with a one-token `count_tokens` call before any
billing-relevant run; that check runs automatically once `GEMINI_API_KEY` is set.

Tests:

```bash
.venv/bin/python -m pytest tests -q
```

## Phase 2: retrieval corpus and question sets

Downloads stay under `data/raw/` (gitignored; Apache-2.0 sources):

- MIRACL Bengali corpus (`miracl-corpus-v1.0-bn`, 297,265 passages).
- MIRACL Bengali topics/qrels (dev + train).
- TyDi QA GoldP Bengali (train + validation).

Build deterministically (seed 42):

```bash
.venv/bin/python experiments/build_corpus.py
```

Composition:

- 100,000 MIRACL passages: all 13,705 judged in dev+train qrels plus 86,295
  seeded random non-judged passages.
- 1,870 unique TyDi Bengali contexts, IDs prefixed `tydi_`.
- 101,870 parent passages total.

Question files: MIRACL train (tuning only), MIRACL dev (reporting only), TyDi
train (tuning only), TyDi validation (reporting only). Gold coverage is 100%.
`data/custom_questions.csv` is user input and is never rewritten.

Retrieval units are built in Phase 3; the 1,477 passages shorter than 20 chars are
kept and flagged with `is_short`.

## Phase 3: chunking, BM25, dense, and hybrid retrieval

Only the 1,501 passages that exceed e5's 512-token input limit are split.
Chunking is sentence-aware on the Bangla danda (।) and newlines, with one-sentence
overlap and a 400-e5-token ceiling. All other passages stay whole. Every unit
keeps `parent_id`; metrics count a parent once at its best-ranked chunk.

```bash
.venv/bin/python experiments/build_chunks.py
.venv/bin/python experiments/phase3_bm25.py
.venv/bin/python experiments/phase3_embed_dense.py
.venv/bin/python experiments/phase3_dense_hybrid_eval.py
.venv/bin/python experiments/phase3_significance_error_analysis.py
.venv/bin/python experiments/make_phase2_phase3_report.py
```

Chunking result: 3,676 chunks from the 1,501 over-length parents, net +2,175
units, 104,045 retrieval units. All new chunks are <=400 e5 tokens.

BM25: NFC -> lowercase -> maximal runs of Unicode `L*`/`M*`/`N*` (keeps Bangla
conjuncts intact), text only, `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`.
Tuned only on MIRACL train (`k1=0.9`, `b=0.35`). Dense retrieval uses
`intfloat/multilingual-e5-small` (384-d, fp16, mean pooling, L2-normalized,
FAISS `IndexFlatIP`). Hybrid uses weighted RRF with `k=5` and equal BM25/dense
weights, tuned on MIRACL train only.

Held-out strict parent-level results:

| Set | System | R@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|
| MIRACL dev | BM25 | 0.7202 | 0.5115 | 0.4551 |
| MIRACL dev | Dense e5-small | 0.9197 | 0.5232 | 0.5544 |
| MIRACL dev | Hybrid RRF | 0.9124 | 0.5914 | 0.5869 |
| TyDi validation | BM25 | 0.5575 | 0.3210 | 0.3729 |
| TyDi validation | Dense e5-small | 0.8938 | 0.6581 | 0.7139 |
| TyDi validation | Hybrid RRF | 0.8850 | 0.4937 | 0.5870 |

Dense and hybrid significantly beat BM25 on R@10 and nDCG@10 on both sets
(paired bootstrap, 10,000 draws, Bonferroni within each metric). On MIRACL dev,
hybrid is significantly better than dense on MRR/nDCG@10 but tied on R@10. On
TyDi validation, dense is significantly better than hybrid on MRR/nDCG@10 but
tied on R@10. Splitting over-length passages beats a single truncated parent by
0.08-0.15 R@10 on the over-length-gold slice. Dense has zero top-10 misses on
short-gold queries. No BGE-M3, compression, generation, or judging API was used.

Reports and analysis:

- `results/phase3_report.md` - concise markdown report.
- `results/phase3_error_analysis.md` - full error tables and verbatim examples.
- `results/Phase2_Phase3_RAG_Report.docx` - standalone Phase 2-3 Word report.
- `results/Bangla_RAG_Report_Phases_1-3.docx` - combined Phase 1-3 Word report.

Machine-readable results: `results/phase3_dense_metrics.csv`,
`results/phase3_tydi_answerstring.csv`, `results/phase3_significance.csv`,
`results/phase3_ablation_truncated_vs_split.csv`,
`results/phase3_hybrid_tuning.csv`, `results/phase3_per_query.jsonl`, and
`results/phase3_*.json`.

Large/derived data (`data/raw/`, `data/processed/`, `data/cache/`, embeddings,
logs) is gitignored and regenerated by the commands above.
