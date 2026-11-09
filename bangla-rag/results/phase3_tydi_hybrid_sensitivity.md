# TyDi-train-tuned hybrid sensitivity (separate from the main hybrid)

RRF weights tuned on **TyDi train only** with rrf_k fixed at 5; evaluated on **TyDi validation**. The primary hybrid remains the MIRACL-train-tuned setting.

| setting | rrf_k | w_BM25 | w_dense | n | R@1 | R@3 | R@5 | R@10 | MRR@100 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TyDi-train-tuned hybrid sensitivity | 5 | 0.00 | 1.00 | 113 | 0.5133 | 0.7788 | 0.8319 | 0.8938 | 0.6580 | 0.7139 |
| Main hybrid (MIRACL-train-tuned, primary; reference only) | 5 | 0.50 | 0.50 | 113 | 0.2743 | 0.6372 | 0.7965 | 0.8850 | 0.4937 | 0.5870 |

Selected weights: **w_BM25=0.00, w_dense=1.00** (train nDCG@10=0.5830).

Bootstrap 95% CIs for the sensitivity row (10,000 question-level resamples, seed 42):

| metric | value | 95% CI |
|---|---:|---|
| recall@1 | 0.5133 | [0.4248, 0.6018] |
| recall@3 | 0.7788 | [0.6991, 0.8496] |
| recall@5 | 0.8319 | [0.7611, 0.8938] |
| recall@10 | 0.8938 | [0.8319, 0.9469] |
| mrr@100 | 0.6580 | [0.5880, 0.7262] |
| ndcg@10 | 0.7139 | [0.6498, 0.7740] |

No retrieval was rerun; rankings were reused from the frozen Phase 3 finalization artifact.
