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
  data/raw/         FLORES-200 (downloaded), future retrieval corpora
  data/processed/   cleaned/derived data
  data/cache/       token counts, HF models, tiktoken vocab, embeddings
  src/              tokenizers.py, chunking.py, retrieval.py, compression.py,
                    generation.py, verification.py, evaluation.py, cache.py
  experiments/      one script per phase; configs in experiments/configs/
  results/          CSV tables and plots
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

## Phase 3: chunking, BM25, and retrieval metrics

Only the 1,501 passages that exceed e5's 512-token input limit are split.
Chunking is sentence-aware on the Bangla danda (।) and newlines, with one-sentence
overlap and a 400-e5-token ceiling. All other passages stay whole. Every unit
keeps `parent_id`; metrics count a parent once at its best-ranked chunk.

```bash
.venv/bin/python experiments/build_chunks.py
.venv/bin/python experiments/phase3_bm25.py
.venv/bin/python experiments/hand_verify_recall.py
```

Chunking result: 3,676 chunks from the 1,501 over-length parents, net +2,175
units, 104,045 retrieval units. All new chunks are <=400 e5 tokens.

BM25: NFC -> lowercase -> maximal runs of Unicode `L*`/`M*`/`N*` (keeps Bangla
conjuncts intact), text only, `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`.
Tuned only on MIRACL train (`k1=0.9`, `b=0.35`); held-out results:

| Set | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| MIRACL dev | 0.4136 | 0.5572 | 0.6229 | 0.7202 | 0.5115 | 0.4551 |
| TyDi validation | 0.1858 | 0.4071 | 0.4425 | 0.5575 | 0.3210 | 0.3729 |

Recall@k is the retrieval hit rate (any gold parent in the top k). Embedding
timing (`multilingual-e5-small`, MPS fp16, batch 16): 1,000 units in 4.90 s
(204 units/s); projected full run 8.5 min for the split index plus 7.4 s for the
1,501 truncated parents used in the dense ablation; ~77 MiB fp16 embeddings;
peak RSS 0.9 GiB. The full embedding run is gated on explicit approval.

Results: `results/phase3_*.json`, `results/phase3_bm25_metrics.csv`,
`results/phase3_bm25_grid.csv`, `results/phase3_recall_handcheck.txt`.

Large/derived data (`data/raw/`, `data/processed/`, `data/cache/`, embeddings)
is gitignored and regenerated by the commands above.
