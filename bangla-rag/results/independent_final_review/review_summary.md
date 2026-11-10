# Independent Final Quality Review — Batch 1 + Batch 2

Review decisions were made from source passages, the frozen corpus and recorded retrieval evidence. Each present row has ACCEPT, REJECT, or SKIP with a reason and evidence IDs. Original `my_decision` and `my_notes` are preserved.

- Batch 1 source questions: 155
- Batch 2 source questions: 418 (418 present; planned 430, 12 Part B records absent)
- Total reviewed: 573
- ACCEPT: 491 (85.7%)
- REJECT: 22 (3.8%)
- SKIP: 60 (10.5%)
- Exact duplicate question groups: 0
- Exact cross-batch duplicate pairs: 0
- Near-duplicate/similar-template pairs reported: 3
- Final combined accepted dataset size: 491

## Batch and category results

- Batch 1/answerable: 125 total; ACCEPT 121 (96.8%), REJECT 4, SKIP 0
- Batch 1/naturally_unanswerable: 30 total; ACCEPT 22 (73.3%), REJECT 5, SKIP 3
- Batch 2/answerable: 200 total; ACCEPT 196 (98.0%), REJECT 2, SKIP 2
- Batch 2/naturally_unanswerable: 68 total; ACCEPT 17 (25.0%), REJECT 0, SKIP 51
- Batch 2/removed_evidence: 150 total; ACCEPT 135 (90.0%), REJECT 11, SKIP 4
- Overall answerable acceptance rate: 317/325 (97.5%)
- Overall naturally unanswerable acceptance rate: 39/98 (39.8%)
- Overall removed evidence acceptance rate: 135/150 (90.0%)

## Main rejection reasons

- A post-removal passage contains the answer in a sentence that also matches the question's subject and requested relation. (8)
- The wording is a malformed statement rather than a clear question, despite the passage supporting the intended fact. (1)
- The passage only calls RETScreen one of a few leading tools; it gives no numerical count to answer ‘how many’. (1)
- The question asks for a starting year, but the passage only says the index is issued every year. (1)
- The source dates the play's story/event to 429 BCE; it does not establish the play's publication year as asked. (1)
- Pabna Zilla School's founding year is stated in the corpus; the item is answerable. (1)
- The question has a false premise: Sinclair Lewis was an author, not a candidate who won Minnesota's 1905 gubernatorial election. (1)
- The premise that Charles Darwin formally founded evolutionary psychology is historically false; the passage does not support that event. (1)

## Main skip reasons

- The source audit flags possible hidden evidence; retrieved context requires a semantic answerability judgment. (48)
- A corpus passage is flagged as possible hidden evidence; its relevance needs a human semantic check. (3)
- Salma Akhtar is not uniquely identified and the retrieved medal passages concern other people or competitions; answerability is uncertain. (1)
- The question presupposes a special diet chart not established by the retrieved cricket passages. (1)
- The historical location of Newton's calculus work is interpretive and not established as a single directly evidenced city by these passages. (1)
- The Bengali wording is directionally ambiguous (which park lies west of the promenade versus which park the promenade lies west of); do not infer the intended relation. (1)
- The passage does not directly establish the recorded answer in a verifiable form. (1)
- The corpus passage calls 30 para ‘chapters’ while separately listing 114 suras; the question's use of chapter is ambiguous and should not be resolved by assumption. (1)

## Evidence and source issues

- Batch 2 was planned for 430 questions, but the source files contain 200 Part A, 68 Part B, and 150 Part C rows. Twelve planned Part B items are absent and were not fabricated.
- 11 Part C records were rejected because a post-removal BM25/dense candidate still directly contained the original answer, or because the question was a confirmed duplicate.
- 51 Part B rows remain SKIP because the source audit itself flags possible hidden evidence; these require semantic adjudication.
- Accepted rows with missing corpus evidence IDs: 0.
- The compact CSVs retain passage IDs and text; review decisions are represented by the accepted/rejected/skipped file partitions. Detailed CLI-only review columns were omitted at the user's request.

## Output files

All outputs are under `results/independent_final_review/`. The accepted file contains only ACCEPT rows. Rejected and skipped rows remain available for audit. The compact combined CSVs include passage ID, passage role, and passage text for each record where available; `removed_gold_reference` marks evidence excluded from Part C evaluation, while naturally unanswerable rows use a source hint or retrieved context. The original CSVs and corpus were not written.
