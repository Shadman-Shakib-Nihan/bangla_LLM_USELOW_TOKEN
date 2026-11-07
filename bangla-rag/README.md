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
