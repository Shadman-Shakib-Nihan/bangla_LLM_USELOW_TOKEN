# Phase 3 report — BM25, dense, and hybrid retrieval

Scope: parent-passage retrieval over the shared 104,045 retrieval units in
`data/processed/corpus_chunks.jsonl` (101,870 parents). BM25 and dense use the
same units; dense uses `intfloat/multilingual-e5-small` (fp16, batch 16, mean
pooling, L2-normalized, FAISS `IndexFlatIP`). Hybrid is weighted RRF.

Tuning was done on MIRACL train only. Reporting sets are MIRACL dev (n=411) and
TyDi QA validation (n=113). Main-table CIs are 1,000-query bootstrap draws
(seed 42); paired significance tests use 10,000 draws (seed 42).

Word deliverables: `results/Phase2_Phase3_RAG_Report.docx` (standalone) and
`results/Bangla_RAG_Report_Phases_1-3.docx` (combined with Phase 1).

## Main results (strict gold passage IDs)

### MIRACL dev (n=411)

| system | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 | 0.4136 [0.3674, 0.4599] | 0.5572 [0.5085, 0.6010] | 0.6229 [0.5766, 0.6691] | 0.7202 [0.6764, 0.7591] | 0.5115 [0.4705, 0.5505] | 0.4551 [0.4188, 0.4875] |
| Dense (e5-small) | 0.2725 [0.2336, 0.3163] | 0.7445 [0.7032, 0.7883] | 0.8394 [0.8053, 0.8735] | 0.9197 [0.8929, 0.9440] | 0.5232 [0.4953, 0.5554] | 0.5544 [0.5283, 0.5790] |
| Hybrid (RRF) | 0.4331 [0.3869, 0.4793] | 0.6959 [0.6521, 0.7397] | 0.8005 [0.7639, 0.8370] | 0.9124 [0.8856, 0.9392] | 0.5914 [0.5564, 0.6253] | 0.5869 [0.5580, 0.6135] |

### TyDi validation (n=113)

| system | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 | 0.1858 [0.1150, 0.2655] | 0.4071 [0.3186, 0.4867] | 0.4425 [0.3451, 0.5223] | 0.5575 [0.4602, 0.6460] | 0.3210 [0.2532, 0.3876] | 0.3729 [0.3035, 0.4405] |
| Dense (e5-small) | 0.5133 [0.4248, 0.6018] | 0.7788 [0.6903, 0.8496] | 0.8319 [0.7611, 0.8938] | 0.8938 [0.8319, 0.9469] | 0.6581 [0.5881, 0.7257] | 0.7139 [0.6481, 0.7718] |
| Hybrid (RRF) | 0.2743 [0.1947, 0.3628] | 0.6372 [0.5487, 0.7168] | 0.7965 [0.7168, 0.8673] | 0.8850 [0.8230, 0.9381] | 0.4937 [0.4338, 0.5559] | 0.5870 [0.5318, 0.6414] |

MRR for BM25/dense is over the full ranking; hybrid MRR is over the fused
top-100 + top-100 candidate lists.

### TyDi validation — answer-string hit

A hit means any chunk of any top-k retrieved parent contains a gold answer
string after NFC + lowercase + whitespace normalization.

| system | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 | 0.4602 [0.3628, 0.5575] | 0.5929 [0.5042, 0.6816] | 0.6195 [0.5310, 0.7168] | 0.6991 [0.6106, 0.7790] | 0.5387 [0.4502, 0.6259] | 0.5719 [0.4857, 0.6551] |
| Dense (e5-small) | 0.7257 [0.6460, 0.8142] | 0.9027 [0.8407, 0.9558] | 0.9292 [0.8761, 0.9735] | 0.9469 [0.9027, 0.9823] | 0.8105 [0.7474, 0.8700] | 0.8431 [0.7888, 0.8935] |
| Hybrid (RRF) | 0.5487 [0.4513, 0.6460] | 0.7876 [0.7080, 0.8584] | 0.8938 [0.8319, 0.9469] | 0.9469 [0.9027, 0.9823] | 0.6882 [0.6161, 0.7534] | 0.7506 [0.6907, 0.8056] |

## RRF tuning (MIRACL train only)

Best: `rrf_k=5`, `w_BM25=0.50`, `w_dense=0.50`; train nDCG@10 = 0.6149.
Grid: k in {1,2,3,5,8,10,15,20,30,40,60,80,100}, weights 0.00–1.00 step 0.05.
The optimum is interior, not a grid boundary.

## Paired bootstrap: key contrasts

Differences are A − B; 95% percentile CI; Bonferroni CI is 98.33% for the
three-pair family. Both intervals exclude zero unless marked "no".

### MIRACL dev

| metric | A vs B | Δ | 95% CI | excl. 0? | Bonf. CI | excl. 0? |
|---|---|---|---|---|---|---|
| R@10 | BM25 vs Dense | -0.1995 | [-0.2433, -0.1557] | yes | [-0.2530, -0.1460] | yes |
| R@10 | BM25 vs Hybrid | -0.1922 | [-0.2311, -0.1533] | yes | [-0.2409, -0.1460] | yes |
| R@10 | Dense vs Hybrid | +0.0073 | [-0.0146, +0.0292] | no | [-0.0195, +0.0341] | no |
| MRR | BM25 vs Dense | -0.0117 | [-0.0540, +0.0308] | no | [-0.0638, +0.0404] | no |
| MRR | BM25 vs Hybrid | -0.0799 | [-0.1073, -0.0525] | yes | [-0.1130, -0.0468] | yes |
| MRR | Dense vs Hybrid | -0.0682 | [-0.0977, -0.0390] | yes | [-0.1047, -0.0322] | yes |
| nDCG@10 | BM25 vs Dense | -0.0992 | [-0.1342, -0.0635] | yes | [-0.1423, -0.0559] | yes |
| nDCG@10 | BM25 vs Hybrid | -0.1318 | [-0.1541, -0.1102] | yes | [-0.1590, -0.1051] | yes |
| nDCG@10 | Dense vs Hybrid | -0.0325 | [-0.0537, -0.0126] | yes | [-0.0584, -0.0076] | yes |

### TyDi validation

| metric | A vs B | Δ | 95% CI | excl. 0? | Bonf. CI | excl. 0? |
|---|---|---|---|---|---|---|
| R@10 | BM25 vs Dense | -0.3363 | [-0.4336, -0.2389] | yes | [-0.4513, -0.2212] | yes |
| R@10 | BM25 vs Hybrid | -0.3274 | [-0.4159, -0.2389] | yes | [-0.4336, -0.2212] | yes |
| R@10 | Dense vs Hybrid | +0.0088 | [-0.0265, +0.0531] | no | [-0.0354, +0.0619] | no |
| MRR | BM25 vs Dense | -0.3371 | [-0.4139, -0.2600] | yes | [-0.4315, -0.2438] | yes |
| MRR | BM25 vs Hybrid | -0.1728 | [-0.2199, -0.1275] | yes | [-0.2319, -0.1172] | yes |
| MRR | Dense vs Hybrid | +0.1643 | [+0.1058, +0.2257] | yes | [+0.0929, +0.2397] | yes |
| nDCG@10 | BM25 vs Dense | -0.3410 | [-0.4167, -0.2649] | yes | [-0.4344, -0.2476] | yes |
| nDCG@10 | BM25 vs Hybrid | -0.2141 | [-0.2624, -0.1664] | yes | [-0.2732, -0.1554] | yes |
| nDCG@10 | Dense vs Hybrid | +0.1269 | [+0.0801, +0.1750] | yes | [+0.0699, +0.1861] | yes |

Multiple-comparison note: within each (dataset, metric), the three pairwise
tests form one family; Bonferroni uses alpha = 0.05/3 = 0.0167. The tests reuse
the same queries and are correlated, so Bonferroni is conservative.

## Error analysis summary

Top-10 win/loss counts:

| dataset | BM25 hit / Dense miss | Dense hit / BM25 miss | Hybrid misses | Dense misses | BM25 misses |
|---|---|---|---|---|---|
| miracl_dev | 10 | 92 | 36 | 33 | 115 |
| tydi_validation | 2 | 40 | 13 | 12 | 50 |

Gold rank when the system misses top-10:

| dataset | system | n missed | 11-20 | 21-50 | 51-100 | 101-200 | 201-500 | 501-1000 | >1000 | absent |
|---|---|---|---|---|---|---|---|---|---|---|
| miracl_dev | BM25 | 115 | 31 | 30 | 14 | 14 | 11 | 5 | 6 | 4 |
| miracl_dev | Dense | 33 | 15 | 9 | 2 | 2 | 0 | 4 | 1 | 0 |
| miracl_dev | Hybrid | 36 | 18 | 9 | 4 | 3 | 0 | 0 | 0 | 2 |
| tydi_validation | BM25 | 50 | 6 | 9 | 7 | 4 | 6 | 5 | 11 | 2 |
| tydi_validation | Dense | 12 | 2 | 6 | 3 | 1 | 0 | 0 | 0 | 0 |
| tydi_validation | Hybrid | 13 | 3 | 6 | 3 | 1 | 0 | 0 | 0 | 0 |

Over-length gold (queries with at least one over-length gold parent; metrics
use the full gold set):

| dataset | subgroup | n | BM25 R@10 | Dense R@10 | Hybrid R@10 |
|---|---|---|---|---|---|
| miracl_dev | no | 374 | 0.7086 | 0.9171 | 0.9064 |
| miracl_dev | yes | 37 | 0.8378 | 0.9459 | 0.9730 |
| tydi_validation | no | 109 | 0.5596 | 0.8899 | 0.8899 |
| tydi_validation | yes | 4 | 0.5000 | 1.0000 | 0.7500 |

Short gold passages: the Phase 2 `is_short` flag (stripped characters <20)
matches no gold passage in either reporting set. Using e5-token thresholds,
the largest short-gold slice is <50 tokens (11 MIRACL dev queries, 1 TyDi
validation query), and dense has zero top-10 misses on it. Short gold is not a
failure mode in these reporting sets.

Concrete verbatim examples (up to five per category, including BM25-only wins,
dense-only wins, deep dense misses, hybrid misses, TyDi strict-ID misses that
are answer-string hits, short-gold misses, and over-length-gold misses) are in
`results/phase3_error_analysis.md`.

## Truncated-vs-split ablation (over-length gold)

| dataset | n | split R@10 (over gold) | truncated R@10 (over gold) | Δ |
|---|---|---|---|---|
| MIRACL train | 161 | 0.7516 [0.6893, 0.8199] | 0.6584 [0.5839, 0.7329] | -0.0932 |
| MIRACL dev | 37 | 0.8919 [0.7838, 0.9730] | 0.8108 [0.6757, 0.9189] | -0.0811 |
| TyDi train | 113 | 0.7699 [0.6903, 0.8496] | 0.6195 [0.5310, 0.7080] | -0.1504 |
| TyDi validation | 4 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.0000 |

Split chunks beat single truncated parents on the over-length-gold slice in
every non-degenerate set.

## Retrieval units under 20 tokens

- All units: 4,814 / 104,045 (4.63%); 966 <10; 2 <5.
- Split chunks only: 3 / 3,676 (0.08%); 3 <10; 0 <5.

## Caveats

- The 100k MIRACL slice is easier than the full 297k-passage corpus; absolute
  scores are not full-corpus estimates.
- TyDi has two scoring rules: strict gold ID and answer-string hit. The latter
  is more lenient.
- Validation sets are small (411 and 113), and subgroups are much smaller
  (TyDi validation over-length gold n=4).
- RRF hyperparameters were tuned on MIRACL train only.
- BM25/dense MRR uses full rankings; hybrid MRR uses the fused top-100+top-100
  candidate list.
- Bootstrap resamples queries, not passages.
- No BGE-M3, compression, or external API call was run.

## Artifacts

- Metrics: `results/phase3_dense_metrics.csv`
- TyDi answer-string: `results/phase3_tydi_answerstring.csv`
- RRF grid: `results/phase3_hybrid_tuning.csv`
- Ablation: `results/phase3_ablation_truncated_vs_split.csv`
- Chunk lengths: `results/phase3_chunk_token_stats.json`
- Embedding checks: `results/phase3_embed_checks.json`
- Per-query records: `results/phase3_per_query.jsonl`
- Significance: `results/phase3_significance.csv`
- Full error analysis + examples: `results/phase3_error_analysis.md`
