# Custom Bengali QA drafts v2 processing log

## Input inspection

- Input: `custom_drafts.csv`; UTF-8 with optional BOM accepted.
- Original rows: **180**; original columns: **14**.
- Duplicate IDs: **0**.
- Mappings: `gold_answer` is the answer; `passage_snippet` is the passage snippet; `max_shared_word_4gram` is the shared 4-gram count. `answerable` and `answer_is_substring` were read and preserved separately.
- `check_flags` is parsed as semicolon-delimited exact tokens. The observed `name_heavy` flag was present (1 row(s)).

### Column types, missing values, and sample values

| Column | Inferred CSV value type | Missing | Sample values |
|---|---|---:|---|
| `id` | string | 0 | cd_0001<br>cd_0002<br>cd_0003 |
| `question` | string | 0 | ব্লো-ব্জাং-য়ে-শেস-ব্স্তান-'দ্জিন-র্গ্যা-ম্ত্শোর বাবার নাম কী ছিল?<br>যুক্তরাষ্ট্রের ন্যাশনাল প্রেস ক্লাব কত সালে প্রতিষ্ঠিত হয়?<br>বাংলাদেশের দশ টাকার নোটে কোন মসজিদের প্রতিকৃতি মুদ্রিত আছে? |
| `question_type` | string | 0 | who<br>when<br>who |
| `gold_answer` | string | 30 | ত্শে-রিং-দোন-স্গ্রুব<br>১৯০৮<br>আতিয়া মসজিদ |
| `passage_snippet` | string | 0 | ব্লো-ব্জাং-য়ে-শেস-ব্স্তান-'দ্জিন-র্গ্যা-ম্ত্শো ১৯০১ খ্রিষ্টাব্দের পয়লা মার্চ তিব্বতের গুং-থাং অঞ্চ<br>১৯০৮ সালে প্রতিষ্ঠিত হবার পর থিওডোর রুজভেল্ট থেকে যুক্তরাষ্ট্রের প্রত্যেক রাষ্ট্রপতি এ সংগঠনটি পরিদর<br>বর্তমানে বাংলাদেশে প্রচলিত বাংলাদেশ ব্যাংক কর্তৃক মুদ্রিত১০ (দশ) টাকা মূল্যমানের নোটের একপার্শ্বে আত |
| `gold_passage_id` | string | 0 | 386361#1<br>95537#1<br>297394#7 |
| `alt_gold_ids` | string | 84 | 1098#34;1122#9;1197#8;1398#14;1516#2;1528#30;1528#33;2047#4;2047#6;2727#1;2730#36;3094#0;3426#4;3796<br>42558#16;297394#3;301610#3;543797#12<br>2850#21;401830#2;428289#2;710204#1;711413#1 |
| `answerable` | boolean-like string | 0 | true<br>true<br>true |
| `language_pair` | string | 0 | bn-bn<br>bn-bn<br>bn-bn |
| `answer_is_substring` | boolean-like string | 30 | true<br>true<br>true |
| `max_shared_word_4gram` | integer-like string | 0 | 4<br>0<br>0 |
| `check_flags` | string | 112 | shared_4gram=4;name_heavy<br>time_sensitive_undated<br>shared_4gram=1 |
| `my_decision` | string | 179 | reject |
| `my_notes` | string | 179 | Current banknote design; passage gives no dated fact. Replace this item. |

## Exclusions

- Answerable retained in `custom_drafts_v2.csv`: **125**.
- Unanswerable preserved in `custom_drafts_unanswerable.csv`: **30**.
- Genuinely excluded answerable records in `custom_drafts_excluded.csv`: **25**.
- Counts by applicable reason (a row may count under more than one reason):
  - `answer_appears_in_question`: 9
  - `answer_is_substring_false`: 12
  - `name_heavy`: 1
  - `normalized_answer_under_2_characters`: 2
  - `shared_4gram_at_least_5`: 4
- Excluded rows and all applicable reasons are in `custom_drafts_excluded.csv`. Original values remain present; `exclusion_reasons` was added.
- Unanswerable rows were routed before short-answer validation; their 14 original columns and values were copied unchanged.

## Answer-type validation

- Year/date questions with valid four-digit years: 18.
- Full-date answers: **6** (`valid_full_date`).
- Quantity questions with recognizable numeric answers: 17.
- Type mismatches: **3** (2 year/date; 1 quantity). They remain in the output for review.
- Non-date/non-quantity and unanswerable rows use `not_applicable`.
- Bengali and English digits are normalized to ASCII for validation. Common Bengali number words are accepted from a fixed documented list in the processing script.

## Alternative gold passages

- Corpus: `corpus.jsonl`, 101870 passage records; available and read successfully.
- Qualifying passage requires both normalized gold-answer presence and at least one rare question content term. A rare term has document frequency at most 509 (0.5% of corpus size, with a minimum threshold of 5). Question words, stopwords, numeric-only tokens, and short terms are excluded. Common Bengali suffixes are stripped for term matching.
- Existing `alt_gold_ids` changed for **70** retained rows. Results are deduplicated, sorted, and capped at 10 IDs.
- Rows with more than 10 qualifying passages: **11**; `alt_gold_total_count` retains the full number and `alt_gold_over_cap` marks overflow.
- Limitation: suffix handling and the fixed Bengali stopword/number-word lists are heuristic; rare/entity matching may miss inflected terms or may include uncommon non-entity content words.

## Question type

- Recomputed deterministically for retained answerable examples using question-form patterns: when/date, quantity, why/how, place, who, then what as the fallback.
- Labels changed among retained rows: **36**. `question_type_review.csv` records every source question, the assigned label, reason, and review status.
- Review statuses across all source questions: {'manual_review': 122, 'rule_assigned': 58}; assigned type counts: {'what': 61, 'when': 46, 'how_many': 25, 'why_how': 13, 'where': 24, 'who': 11}.
- The rules are not fully reliable. Ambiguous/uncertain cases and fallback assignments are marked `manual_review` for human checking.

## Record accounting

- Serialized outputs: 125 answerable + 30 unanswerable + 25 genuinely excluded = 180 records.
- Stable-ID accounting verified: 180 unique output IDs matched 180 unique source IDs, with no missing, extra, or duplicate records.
- Original values in unanswerable and excluded rows were compared to source. Retained values were compared except explicitly recomputed `question_type` and `alt_gold_ids`; `my_decision` and `my_notes` were checked unchanged.

## Preservation and outputs

- `my_decision` and `my_notes` were copied unchanged for every retained record.
- `answerable` and `answer_is_substring` remain separate original columns and were not conflated.
- Source file was read only. Outputs: `custom_drafts_v2.csv`, `custom_drafts_unanswerable.csv`, `custom_drafts_excluded.csv`, and `question_type_review.csv`.
- No answers, passage IDs, or evidence annotations were invented.

## Final validation

- Unanswerable recovery: **30/30** source records are in `custom_drafts_unanswerable.csv`, preserving every original column and value.
- Output accounting: **125 answerable + 30 unanswerable + 25 genuinely excluded = 180/180** source rows. All output IDs are unique across the three datasets; no rows are missing or duplicated.
- Exclusion-integrity tests and runtime checks passed for source-value preservation, complete exclusion reasons, multiple reasons, stable-ID accounting, no duplicate/lost rows, and unchanged `my_decision`/`my_notes`.
- `question_type_review.csv`: 180 rows; 58 `rule_assigned`, 122 `manual_review`. Classification remains deterministic but is not claimed fully reliable.
- Full test suite: `.venv/bin/python -m pytest tests -q` — **80 passed, 0 failed**.

## Follow-up validation

- Unanswerable recovery: **30/30** source records are present in `custom_drafts_unanswerable.csv`, preserving all 14 original columns and values.
- Final record accounting: **125 answerable + 30 unanswerable + 25 genuinely excluded = 180/180** source records; all IDs are unique across the three datasets.
- Exclusion integrity checks passed: excluded original values and all applicable reasons, multiple reasons, no missing/duplicate IDs, output partition accounting, and `my_decision`/`my_notes` preservation were covered by tests and runtime validation.
- Question-type review: **180** rows; **58** `rule_assigned`, **122** `manual_review`. These labels remain provisional and are not claimed to be fully reliable.
- Full test suite: `.venv/bin/python -m pytest tests -q` — **80 passed, 0 failed**.
