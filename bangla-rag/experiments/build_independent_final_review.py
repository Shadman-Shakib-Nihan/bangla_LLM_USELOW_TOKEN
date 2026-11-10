#!/usr/bin/env python3
"""Build an auditable, read-only independent review package for Batch 1/2.

Source CSVs and corpus artifacts are only read. Review decisions are generated
from direct passage checks, retrieved evidence, and explicit conservative
exceptions; uncertain cases are SKIP rather than inferred as ACCEPT.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "independent_final_review"
CORPUS = ROOT / "data/processed/corpus.jsonl"
FROZEN = ROOT / "data/custom_batch2/source_inventory.json"
PROTECTED = ROOT / "data/custom_batch2/protected_checksums.json"
PART_B_AUDIT = ROOT / "data/custom_batch2/batch2_full_approved/audits/part_b_audit.json"
PART_C_AUDIT = ROOT / "data/custom_batch2/batch2_full_approved/audits/part_c_audit.json"
FIELDS = ["cli_review_decision", "cli_review_reason", "cli_review_evidence", "cli_review_confidence", "duplicate_of"]
REVIEWER = "independent_cli_reviewer"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid/missing header in {path}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"Malformed extra fields: {path}:{line}")
            rows.append({k: (v or "") for k, v in row.items()})
        return list(reader.fieldnames), rows


def write_csv(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Review decisions remain in the separate accepted/rejected/skipped files
    # and summary; these CLI-only fields are intentionally omitted from CSVs.
    omitted = {"cli_review_decision", "cli_review_reason", "cli_review_evidence",
               "cli_review_confidence", "duplicate_of", "reviewed_by"}
    columns = [column for column in columns if column not in omitted]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        def clean_cell(value):
            if not isinstance(value, str):
                return value
            lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            return "\n".join(line.rstrip(" \t") for line in lines).rstrip()
        w.writerows({k: clean_cell(v) for k, v in row.items()} for row in rows)


def norm(s: str) -> str:
    # Python's ``\w`` drops Bengali vowel signs/combining marks. Preserve all
    # Unicode letters, marks, and numbers; discard only punctuation/spacing.
    return "".join(ch for ch in unicodedata.normalize("NFKC", s or "").casefold()
                   if unicodedata.category(ch)[0] in {"L", "M", "N"})


def direct_answer(answer: str, text: str) -> bool:
    a, t = norm(answer), norm(text)
    return bool(a and a in t)


def contains_all_answer_terms(answer: str, text: str) -> bool:
    terms = [norm(x) for x in re.split(r"[,،;؛]|\s+এবং\s+|\s+ও\s+", answer) if norm(x)]
    t = norm(text)
    return bool(terms) and all(x in t for x in terms)


def unique_ids(rows, key):
    vals = [r.get(key, "") for r in rows]
    if len(vals) != len(set(vals)):
        raise ValueError(f"Duplicate source record keys in {key}")


def load_corpus():
    out = {}
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            out[x["passage_id"]] = x
    return out


def evidence_text(ids, corpus):
    return "\n".join(corpus[i].get("text", "") for i in ids if i in corpus)


def decision(dec, reason, evidence, confidence):
    return {"cli_review_decision": dec, "cli_review_reason": reason,
            "cli_review_evidence": ";".join(dict.fromkeys(str(x) for x in evidence if x)),
            "cli_review_confidence": confidence, "duplicate_of": ""}


def review_answerable(row, text, pid, category):
    q, a = row.get("question", ""), row.get("gold_answer", "")
    flags = row.get("check_flags", row.get("filter_flags", ""))
    if not pid or not text:
        return decision("SKIP", "Gold passage ID is absent from the available corpus; support cannot be verified.", [pid], "LOW")
    if not a:
        return decision("REJECT", "Gold answer is empty for an answerable-category item.", [pid], "HIGH")
    if norm(a) and norm(a) in norm(q):
        return decision("REJECT", "The question states the gold answer, creating answer leakage.", [pid], "HIGH")
    supported = direct_answer(a, text) or contains_all_answer_terms(a, text)
    if not supported:
        return decision("SKIP", "The passage does not directly establish the recorded answer in a verifiable form.", [pid], "LOW")
    if "থার্মোপাইল যুদ্ধ" in q:
        return decision("REJECT", "The passage says the cited 479 BCE event occurred one year after Thermopylae; the recorded answer confuses that later event with the battle date.", [pid], "HIGH")
    return decision("ACCEPT", "The gold passage directly supports the answer and the question asks for that fact.", [pid], "HIGH")


def review_unanswerable(row, retrieved_ids, corpus, category, source_pid=""):
    q = row.get("question", "")
    candidate_ids = list(dict.fromkeys([source_pid] + list(retrieved_ids)))
    text = evidence_text(candidate_ids, corpus)
    # For rows with a gold answer (Batch 1 legacy unanswerables), use exact
    # answer support. For Batch 2, phrase overlap alone is never enough to reject.
    gold = row.get("gold_answer", "")
    if gold and direct_answer(gold, text):
        return decision("REJECT", "A retrieved corpus passage directly contains the recorded answer; this is answerable, not a valid negative.", candidate_ids, "HIGH")
    if not q.strip() or len(norm(q)) < 8:
        return decision("SKIP", "Question is too incomplete to judge answerability confidently.", candidate_ids, "LOW")
    if row.get("hidden_evidence_candidate", "") and row.get("hidden_evidence_candidate", "") != "manual_evidence_review":
        hidden = row["hidden_evidence_candidate"]
        if hidden in corpus:
            hidden_text = corpus[hidden].get("text", "")
            if direct_answer(row.get("gold_answer", ""), hidden_text) and row.get("gold_answer", ""):
                return decision("REJECT", "The flagged corpus passage directly contains the answer, so this is not unanswerable.", candidate_ids + [hidden], "HIGH")
            return decision("SKIP", "A corpus passage is flagged as possible hidden evidence; its relevance needs a human semantic check.", candidate_ids + [hidden], "LOW")
    if row.get("filter_flags", "") and "manual_evidence_review" in row.get("filter_flags", ""):
        return decision("SKIP", "The source audit flags possible hidden evidence; retrieved context requires a semantic answerability judgment.", candidate_ids, "LOW")
    if row.get("filter_flags", "") and "likely_answerable" in row.get("filter_flags", ""):
        return decision("SKIP", "The source audit flags this item as potentially answerable; the retrieved passages do not establish the answer confidently.", candidate_ids, "LOW")
    return decision("ACCEPT", "The question is coherent and neither the source hint nor the inspected BM25/dense top passages directly answers it; similar-topic passages alone do not establish an answer.", candidate_ids, "MEDIUM")


def review_removed(row, corpus, audit):
    pid = row.get("original_gold_passage_id", "")
    removed = [x for x in row.get("removed_evidence_ids", "").split(";") if x]
    q, answer = row.get("question", ""), row.get("original_gold_answer", "")
    ids = list(dict.fromkeys(removed + audit.get("bm25_top_after_removal", [])[:10] + audit.get("dense_top_after_removal", [])[:10]))
    if not audit:
        return decision("SKIP", "No matching post-removal BM25/dense audit record was found for this source question.", ids + [pid], "LOW")
    if pid not in corpus or pid not in removed:
        return decision("REJECT", "Original gold passage is missing or was not included in the removed-evidence IDs.", ids + [pid], "HIGH")
    if not answer or not direct_answer(answer, corpus[pid].get("text", "")):
        return decision("REJECT", "The original gold passage does not directly support the recorded original answer.", ids + [pid], "HIGH")
    rem = row.get("hidden_evidence_remaining", "").strip()
    if rem or audit.get("hidden_evidence_remaining"):
        return decision("REJECT", "Audit identifies hidden gold evidence remaining in the searchable corpus.", ids + [rem], "HIGH")
    remaining_ids = ids[len(removed):]
    from src.batch2_text import content_terms, question_terms
    qterms = question_terms(q)
    relation_words = {
        "প্রথম", "প্রকাশ", "পুরস্কার", "জাতীয়", "অধ্যায়", "রূকু", "সন্তান", "স্ত্রী", "স্ত্রীর",
        "আবিষ্কার", "সংকলন", "প্রতিষ্ঠাতা", "অবস্থিত", "রাজ্যে", "শ্রেণী", "ভাগ", "দল", "দলের",
        "খেল", "কর্নেল", "অধিনায়ক", "আখ্যা", "নদী", "উপাধি", "তারিখ", "সালে", "শহর",
    }
    relation_in_q = any(word in q for word in relation_words)
    direct_hits = []
    for evidence_id in remaining_ids:
        rec = corpus.get(evidence_id, {})
        full_text = (rec.get("title", "") + "।" + rec.get("text", ""))
        for sentence in re.split(r"[।!?\n]+", full_text):
            if not direct_answer(answer, sentence):
                continue
            overlap = qterms & content_terms(sentence)
            # Answer overlap alone is insufficient. Require question-specific
            # entities/concepts and a predicate cue in the same sentence.
            ordinal_ok = "প্রথম" not in q or "প্রথম" in sentence
            if len(overlap) >= 2 and (not relation_in_q or ordinal_ok):
                direct_hits.append(evidence_id)
                break
    if direct_hits:
        return decision("REJECT", "A post-removal passage contains the answer in a sentence that also matches the question's subject and requested relation.", direct_hits, "HIGH")
    if len(norm(q)) < 10:
        return decision("SKIP", "The post-removal question is too incomplete to verify its meaning confidently.", ids, "LOW")
    return decision("ACCEPT", "The original gold passage supports the answer, that evidence ID is removed, and inspected post-removal BM25/dense top passages do not directly support the answer.", ids, "MEDIUM")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    b1a = ROOT / "data/custom_drafts_v2.csv"
    b1b = ROOT / "data/custom_drafts_unanswerable.csv"
    b1r = ROOT / "data/custom_drafts_unanswerable_retrieval.csv"
    b2dir = ROOT / "data/custom_batch2/batch2_full_approved"
    b2a = b2dir / "batch2_answerable_review.csv"
    b2b = b2dir / "batch2_unanswerable_review.csv"
    b2c = b2dir / "batch2_removed_evidence_review.csv"
    sources = [b1a, b1b, b1r, b2a, b2b, b2c, CORPUS, FROZEN, PROTECTED, PART_B_AUDIT, PART_C_AUDIT]
    checksums_before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    _, arows = read_csv(b1a)
    _, brows = read_csv(b1b)
    _, rrows = read_csv(b1r)
    _, b2arows = read_csv(b2a)
    _, b2brows = read_csv(b2b)
    _, b2crows = read_csv(b2c)
    unique_ids(arows, "id"); unique_ids(brows, "id")
    ret = defaultdict(list)
    for r in rrows: ret[r["id"]].append(r)
    if any(len(ret[r["id"]]) != 5 for r in brows):
        raise ValueError("Batch 1 unanswerable retrieval mapping is not five per ID")
    b2baudit = json.loads(PART_B_AUDIT.read_text(encoding="utf-8"))
    b2caudit = json.loads(PART_C_AUDIT.read_text(encoding="utf-8"))
    ba_by_q = {x["question"]: x for x in b2baudit}
    # Part C source IDs map to the saved audit's original TyDi source questions.
    # Part C is explicitly sampled from the held-out portion of the TyDi train
    # pool after excluding tuning/frozen IDs (see run_batch2_full_approved.py).
    tydi_rows = [json.loads(l) for l in (ROOT / "data/processed/tydi_train_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    source_by_q = {norm(x.get("question", "")): x.get("question_id", x.get("qid", "")) for x in tydi_rows}
    ca_by_source = {x.get("source_question_id", ""): x for x in b2caudit}

    allrows = []
    batch1_rows = []
    for index, r in enumerate(arows, 1):
        pid = r.get("gold_passage_id", "")
        item = {**r, "source_file": b1a.name, "batch": "Batch 1", "category": "answerable", "source_record_id": r["id"]}
        item.update(review_answerable(r, corpus.get(pid, {}).get("text", r.get("passage_snippet", "")), pid, "answerable"))
        manual_b1a = {
            "cd_0015": ("REJECT", "The wording is a malformed statement rather than a clear question, despite the passage supporting the intended fact.", "HIGH"),
            "cd_0020": ("REJECT", "The passage only calls RETScreen one of a few leading tools; it gives no numerical count to answer ‘how many’.", "HIGH"),
            "cd_0044": ("REJECT", "The question asks for a starting year, but the passage only says the index is issued every year.", "HIGH"),
            "cd_0148": ("REJECT", "The source dates the play's story/event to 429 BCE; it does not establish the play's publication year as asked.", "HIGH"),
        }
        if r["id"] in manual_b1a:
            dec, reason, conf = manual_b1a[r["id"]]
            item.update(decision(dec, reason, [pid], conf))
        batch1_rows.append(item); allrows.append(item)
    for index, r in enumerate(brows, 1):
        passages = sorted(ret[r["id"]], key=lambda x: int(x["rank"]))
        ids = [x["passage_id"] for x in passages]
        # Add an independently computed lexical BM25 retrieval against the
        # frozen Phase 3 index, alongside the supplied dense top five.
        from src.retrieval import BM25Index, aggregate_parent_scores, build_parent_index, rank_parents, tokenize
        if index == 1:
            unit_rows = [json.loads(line) for line in (ROOT / "data/processed/corpus_chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
            unit_parents = [x["parent_id"] for x in unit_rows]
            pidx = build_parent_index(unit_parents)
            bm25_index = BM25Index.load(ROOT / "data/cache/bm25_index.pkl")
        scores = bm25_index.score(tokenize(r["question"]), k1=0.9, b=0.35)
        bm_ids = rank_parents(aggregate_parent_scores(scores, pidx), pidx, top_k=20)
        ids = list(dict.fromkeys(bm_ids + ids))
        # Preserve text of five retrieved passages in evidence field by ID;
        # a direct answer from any retrieved passage invalidates a negative.
        item = {**r, "source_file": b1b.name, "batch": "Batch 1", "category": "naturally_unanswerable", "source_record_id": r["id"]}
        item.update(review_unanswerable(r, ids, corpus, "naturally_unanswerable"))
        manual_b1 = {
            "cd_0152": ("SKIP", "Salma Akhtar is not uniquely identified and the retrieved medal passages concern other people or competitions; answerability is uncertain.", "LOW"),
            "cd_0153": ("SKIP", "The question presupposes a special diet chart not established by the retrieved cricket passages.", "LOW"),
            "cd_0154": ("SKIP", "The historical location of Newton's calculus work is interpretive and not established as a single directly evidenced city by these passages.", "LOW"),
            "cd_0160": ("REJECT", "Pabna Zilla School's founding year is stated in the corpus; the item is answerable.", "HIGH"),
            "cd_0170": ("REJECT", "The question has a false premise: Sinclair Lewis was an author, not a candidate who won Minnesota's 1905 gubernatorial election.", "HIGH"),
            "cd_0173": ("REJECT", "The premise that Charles Darwin formally founded evolutionary psychology is historically false; the passage does not support that event.", "HIGH"),
            "cd_0176": ("REJECT", "No scientist received a Nobel Prize for discovering streptococcal pharyngitis; the question has a false premise.", "HIGH"),
            "cd_0179": ("REJECT", "The source snippet directly states that the organization's headquarters is in New Paltan, Dhaka.", "HIGH"),
        }
        if r["id"] in manual_b1:
            dec, reason, conf = manual_b1[r["id"]]
            item.update(decision(dec, reason, ids, conf))
        batch1_rows.append(item); allrows.append(item)

    batch2_rows = []
    for index, r in enumerate(b2arows, 1):
        pid = r.get("gold_passage_id", "")
        item = {**r, "source_file": b2a.name, "batch": "Batch 2", "category": "answerable", "source_record_id": f"A:{pid}:{index}"}
        item.update(review_answerable(r, r.get("gold_passage_text", ""), pid, "answerable"))
        if r.get("question", "") == "চেশ প্রমেনাডে কোন পার্কের পশ্চিমে অবস্থিত?":
            item.update(decision("SKIP", "The Bengali wording is directionally ambiguous (which park lies west of the promenade versus which park the promenade lies west of); do not infer the intended relation.", [pid], "LOW"))
        batch2_rows.append(item); allrows.append(item)
    for index, r in enumerate(b2brows, 1):
        audit = ba_by_q.get(r["question"], {})
        ids = list(dict.fromkeys(audit.get("bm25_top", [])[:20] + audit.get("dense_top", [])[:20]))
        item = {**r, "source_file": b2b.name, "batch": "Batch 2", "category": "naturally_unanswerable", "source_record_id": f"B:{r.get('source_passage_hint','')}:{index}"}
        item.update(review_unanswerable(r, ids, corpus, "naturally_unanswerable", r.get("source_passage_hint", "")))
        batch2_rows.append(item); allrows.append(item)
    for index, r in enumerate(b2crows, 1):
        source_id = source_by_q.get(norm(r["question"]), "")
        audit = ca_by_source.get(source_id, {})
        pid = r.get("original_gold_passage_id", "")
        item = {**r, "source_file": b2c.name, "batch": "Batch 2", "category": "removed_evidence", "source_record_id": f"C:{pid}:{index}"}
        item.update(review_removed(r, corpus, audit))
        manual_c = {
            "কুরআনের প্রথম সূরার রূকুর সংখ্যা কত ?": ("ACCEPT", "The matching passages with 11 ruku concern other surahs; no retrieved sentence states the ruku count of the first surah.", "MEDIUM"),
            "হিন্দু মহাকাব্য রামায়ণ অনুসারে রামের মোট কত জন সন্তান ছিল ?": ("ACCEPT", "The post-removal passages discuss the Ramayana but do not state the number of Rama's children; the two-count match is not direct evidence.", "MEDIUM"),
            "গঠন ও প্রচলন নীতির ভিত্তি তে কম্পিউটারকে কয় ভাগে ভাগ করা হয় ?": ("ACCEPT", "The remaining computer-classification passages do not state the count for classification by construction and operating principle; unrelated occurrences of three do not answer it.", "MEDIUM"),
            "ভারতবর্ষের প্রথম সরকারি স্কুলের নাম কী ?": ("ACCEPT", "The retrieved Rangpur Zilla School passage calls it one of India's oldest schools, but does not establish that it was the first government school.", "MEDIUM"),
            "ইসলাম ধর্মের প্রধান গ্রন্থ কুরআন শরীফে মোট কতগুলো অধ্যায় রয়েছে ?": ("SKIP", "The corpus passage calls 30 para ‘chapters’ while separately listing 114 suras; the question's use of chapter is ambiguous and should not be resolved by assumption.", "LOW"),
            "কত সালে সর্বপ্রথম বাংলাদেশের অভিনেত্রী বিদ্যা সিনহা সাহা মীম জাতীয় চলচ্চিত্র পুরস্কার পেয়েছিলেন ?": ("SKIP", "The retrieved passage mentions a 2014 film release and Mim's National Film Award, but does not tie the award to 2014 in the same evidence; the date cannot be confirmed.", "LOW"),
            "এমা শার্লট ডিউয়ার ওয়াটসন অভিনীত প্রথম চলচ্চিত্রের নাম কী ?": ("SKIP", "The retrieved passage identifies Ballet Shoes as a film featuring Emma Watson but does not establish that it was her first film.", "LOW"),
            "হস্তিনাপুর বর্তমানে কোন রাজ্যে অবস্থিত ?": ("SKIP", "The retrieved passage places the Hastinapur wildlife sanctuary in Uttar Pradesh but does not establish the location of the town itself; entity scope is uncertain.", "LOW"),
            "রামায়ণ নামক মহাকাব্যটি কে সংকলন করেন ?": ("REJECT", "The recorded answer names R. K. Narayan, but the corpus identifies Valmiki as the Ramayana's author; Narayan's retelling is a different work.", "HIGH"),
        }
        if r["question"] in manual_c:
            dec, reason, conf = manual_c[r["question"]]
            item.update(decision(dec, reason, item["cli_review_evidence"].split(";"), conf))
        batch2_rows.append(item); allrows.append(item)

    exact, near = [], []
    # Confirmed near-duplicate pairs in the removed-evidence set. Keep the
    # more natural wording, reject the redundant variant, and link both IDs.
    duplicate_pairs = [
        ("পেরু জাতীয় ফুটবল দলের প্রথম অধিনায়ক কে ছিলেন ?", "পেরু জাতীয় ফুটবল দলটির প্রথম অধিনায়ক কে ছিলেন ?"),
        ("ভারতীয় জাতীয় পতাকার সাদা রংটি কিসের প্রতীক ?", "ভারতের জাতীয় পতাকার সাদা রংটি কীসের প্রতীক ?"),
    ]
    for left_q, right_q in duplicate_pairs:
        left = next((x for x in batch2_rows if x["category"] == "removed_evidence" and x["question"] == left_q), None)
        right = next((x for x in batch2_rows if x["category"] == "removed_evidence" and x["question"] == right_q), None)
        if left and right:
            left.update(decision("REJECT", f"Near-duplicate of {right['source_record_id']}; retain the clearer wording and reject this redundant item.", left.get("cli_review_evidence", "").split(";"), "HIGH"))
            left["duplicate_of"] = right["source_record_id"]
            near.append({"batch1_id": left["source_record_id"], "batch2_id": right["source_record_id"],
                         "similarity": f"{SequenceMatcher(None, norm(left_q), norm(right_q)).ratio():.3f}",
                         "batch1_question": left_q, "batch2_question": right_q, "kind": "near_confirmed",
                         "disposition": "reject first wording; retain clearer second wording"})
    turkey = next((x for x in batch2_rows if x["category"] == "removed_evidence" and "তুরস্কের বর্তমান (২০১৯)" in x["question"]), None)
    qatar = next((x for x in batch2_rows if x["category"] == "removed_evidence" and "কাতারের বর্তমান (২০১৯)" in x["question"]), None)
    if turkey and qatar:
        near.append({"batch1_id": turkey["source_record_id"], "batch2_id": qatar["source_record_id"],
                     "similarity": f"{SequenceMatcher(None, norm(turkey['question']), norm(qatar['question'])).ratio():.3f}",
                     "batch1_question": turkey["question"], "batch2_question": qatar["question"],
                     "kind": "similar_template_distinct_fact", "disposition": "retain both; country and population fact differ"})

    # Cross-batch exact duplicate guard. Near-duplicates are surfaced as
    # candidates and require the row-specific review verdict to be conservative.
    b1 = [x for x in allrows if x["batch"] == "Batch 1"]
    b2 = [x for x in allrows if x["batch"] == "Batch 2"]
    for x in b1:
        for y in b2:
            nx, ny = norm(x.get("question", "")), norm(y.get("question", ""))
            if not nx or not ny: continue
            score = SequenceMatcher(None, nx, ny).ratio()
            pair = {"batch1_id": x["source_record_id"], "batch2_id": y["source_record_id"], "similarity": f"{score:.3f}", "batch1_question": x["question"], "batch2_question": y["question"], "kind": "exact" if nx == ny else "near_candidate", "disposition": ""}
            if nx == ny: exact.append(pair)
            elif score >= .90: near.append(pair)
    for pair in exact:
        # Retain Batch 1 provenance on an exact cross-batch duplicate.
        for y in b2:
            if y["source_record_id"] == pair["batch2_id"]:
                y.update(decision("REJECT", f"Exact cross-batch duplicate of {pair['batch1_id']}; retain the Batch 1 record.", [], "HIGH"))
                y["duplicate_of"] = pair["batch1_id"]
                pair["disposition"] = "reject Batch 2 duplicate; retain Batch 1"
    duplicate_rows = exact + near
    # Save exactly all input columns plus provenance and review fields.
    write_csv(OUT / "batch1_final_review.csv", list(dict.fromkeys(list(arows[0].keys()) + list(brows[0].keys()) + ["source_file", "batch", "category", "source_record_id"] + FIELDS)), batch1_rows)
    write_csv(OUT / "batch2_final_review.csv", list(dict.fromkeys(list(b2arows[0].keys()) + list(b2brows[0].keys()) + list(b2crows[0].keys()) + ["source_file", "batch", "category", "source_record_id"] + FIELDS)), batch2_rows)
    accepted = [x for x in allrows if x["cli_review_decision"] == "ACCEPT"]
    rejected = [x for x in allrows if x["cli_review_decision"] == "REJECT"]
    skipped = [x for x in allrows if x["cli_review_decision"] == "SKIP"]
    # A compact shared schema avoids hundreds of category-specific blank
    # columns in the combined datasets. Full source metadata remains in the
    # separate batch review CSVs and the audit files.
    combined_cols = ["source_record_id", "source_file", "batch", "category",
                     "question_type", "question", "answer", "passage_id",
                     "passage_role", "passage_text", "alternate_evidence_ids",
                     "language_pair", "removed_evidence_ids"]
    def compact(item):
        category = item.get("category", "")
        if category == "answerable":
            passage_id = item.get("gold_passage_id", "")
            passage_role = "gold_evidence"
        elif category == "removed_evidence":
            passage_id = item.get("original_gold_passage_id", "")
            passage_role = "removed_gold_reference"
        else:
            passage_id = item.get("source_passage_hint", "")
            if not passage_id:
                passage_id = next(iter(filter(None, item.get("cli_review_evidence", "").split(";"))), "")
            passage_role = "source_hint" if item.get("source_passage_hint", "") else "retrieved_context"
        passage_text = corpus.get(passage_id, {}).get("text", "")
        if not passage_text:
            passage_text = item.get("gold_passage_text", "") or item.get("passage_snippet", "")
        return {
            "source_record_id": item.get("source_record_id", ""),
            "source_file": item.get("source_file", ""),
            "batch": item.get("batch", ""),
            "category": item.get("category", ""),
            "question_type": item.get("question_type", ""),
            "question": item.get("question", ""),
            "answer": item.get("gold_answer", "") or item.get("original_gold_answer", ""),
            "passage_id": passage_id,
            "passage_role": passage_role,
            "passage_text": passage_text,
            "alternate_evidence_ids": item.get("alt_gold_ids", ""),
            "language_pair": item.get("language_pair", ""),
            "removed_evidence_ids": item.get("removed_evidence_ids", ""),
        }
    write_csv(OUT / "combined_final_accepted.csv", combined_cols, [compact(x) for x in accepted])
    write_csv(OUT / "combined_rejected.csv", combined_cols, [compact(x) for x in rejected])
    write_csv(OUT / "combined_skipped.csv", combined_cols, [compact(x) for x in skipped])
    dupcols = ["record_id_1", "record_id_2", "batch_1", "batch_2", "similarity", "question_1", "question_2", "kind", "disposition"]
    duplicate_rows = [{
        "record_id_1": x["batch1_id"], "record_id_2": x["batch2_id"],
        "batch_1": "Batch 1" if not x["batch1_id"].startswith(("A:", "B:", "C:")) else "Batch 2",
        "batch_2": "Batch 1" if not x["batch2_id"].startswith(("A:", "B:", "C:")) else "Batch 2",
        "similarity": x["similarity"], "question_1": x["batch1_question"], "question_2": x["batch2_question"],
        "kind": x["kind"], "disposition": x["disposition"]
    } for x in duplicate_rows]
    write_csv(OUT / "duplicate_report.csv", dupcols, duplicate_rows)

    # Ensure no original record was lost and no source decisions were modified.
    checksums_after = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    if checksums_before != checksums_after:
        raise RuntimeError("A review input changed during package generation")
    counts_by = {}
    for batch in ("Batch 1", "Batch 2"):
        for cat in ("answerable", "naturally_unanswerable", "removed_evidence"):
            subset = [x for x in allrows if x["batch"] == batch and x["category"] == cat]
            if subset:
                counts_by[f"{batch}/{cat}"] = {"total": len(subset), **dict(Counter(x["cli_review_decision"] for x in subset))}
    # Source-to-output completeness and evidence integrity checks.
    expected_total = len(arows) + len(brows) + len(b2arows) + len(b2brows) + len(b2crows)
    if len(allrows) != expected_total or len({x["source_record_id"] for x in allrows}) != expected_total:
        raise RuntimeError("Source review rows are missing or source_record_id is not unique")
    missing_evidence = []
    for item in accepted:
        for evidence_id in filter(None, item["cli_review_evidence"].split(";")):
            if evidence_id not in corpus:
                missing_evidence.append((item["source_record_id"], evidence_id))
    if missing_evidence:
        raise RuntimeError(f"Accepted records cite missing corpus evidence: {missing_evidence[:5]}")
    if any(item["cli_review_decision"] not in {"ACCEPT", "REJECT", "SKIP"} for item in allrows):
        raise RuntimeError("One or more source rows lacks exactly one final decision")
    exact_text_groups = defaultdict(list)
    for item in allrows:
        exact_text_groups[norm(item["question"])].append(item["source_record_id"])
    exact_all = {k:v for k,v in exact_text_groups.items() if k and len(v) > 1}
    if exact_all:
        raise RuntimeError(f"Exact duplicate questions remain unadjudicated: {list(exact_all.values())[:5]}")
    audit = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "reviewed_by": REVIEWER,
             "source_counts": {"Batch 1": len(arows)+len(brows), "Batch 2": len(b2arows)+len(b2brows)+len(b2crows)},
             "total_reviewed": len(allrows), "decision_counts": dict(Counter(x["cli_review_decision"] for x in allrows)),
             "category_counts": counts_by, "exact_duplicate_question_groups": len(exact_all),
             "cross_batch_exact_duplicates": len(exact), "near_duplicate_candidates_total": len(near),
             "cross_batch_near_duplicate_candidates": sum(1 for x in near if not x["batch1_id"].startswith(("A:", "B:", "C:"))),
             "accepted_rows_with_missing_evidence": len(missing_evidence),
             "input_checksums_sha256": checksums_before, "frozen_review_ids": json.loads(FROZEN.read_text(encoding="utf-8"))["frozen_reviewed_items"]["ids"],
             "source_rows_preserved": len(allrows) == len(arows)+len(brows)+len(b2arows)+len(b2brows)+len(b2crows),
             "batch2_target_gap": {"planned": 430, "present": len(b2arows)+len(b2brows)+len(b2crows), "missing": 430-(len(b2arows)+len(b2brows)+len(b2crows)), "part_b_planned": 80, "part_b_present": len(b2brows)}}
    (OUT / "review_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    def pct(c, n): return f"{(c/n*100 if n else 0):.1f}%"
    summary = ["# Independent Final Quality Review — Batch 1 + Batch 2", "", "Review decisions were made from source passages, the frozen corpus and recorded retrieval evidence. Each present row has ACCEPT, REJECT, or SKIP with a reason and evidence IDs. Original `my_decision` and `my_notes` are preserved.", "", f"- Batch 1 source questions: {len(arows)+len(brows)}", f"- Batch 2 source questions: {len(b2arows)+len(b2brows)+len(b2crows)} (418 present; planned 430, 12 Part B records absent)", f"- Total reviewed: {len(allrows)}", f"- ACCEPT: {len(accepted)} ({pct(len(accepted),len(allrows))})", f"- REJECT: {len(rejected)} ({pct(len(rejected),len(allrows))})", f"- SKIP: {len(skipped)} ({pct(len(skipped),len(allrows))})", f"- Exact duplicate question groups: {len(exact_all)}", f"- Exact cross-batch duplicate pairs: {len(exact)}", f"- Near-duplicate/similar-template pairs reported: {len(near)}", f"- Final combined accepted dataset size: {len(accepted)}", "", "## Batch and category results", ""]
    for k,v in counts_by.items():
        summary.append(f"- {k}: {v['total']} total; ACCEPT {v.get('ACCEPT',0)} ({pct(v.get('ACCEPT',0),v['total'])}), REJECT {v.get('REJECT',0)}, SKIP {v.get('SKIP',0)}")
    for cat in ("answerable", "naturally_unanswerable", "removed_evidence"):
        subset=[x for x in allrows if x["category"]==cat]
        if subset:
            ac=sum(x["cli_review_decision"]=="ACCEPT" for x in subset)
            summary.append(f"- Overall {cat.replace('_',' ')} acceptance rate: {ac}/{len(subset)} ({pct(ac,len(subset))})")
    reject_reasons=Counter(x["cli_review_reason"] for x in rejected)
    skip_reasons=Counter(x["cli_review_reason"] for x in skipped)
    summary += ["", "## Main rejection reasons", ""]
    summary += [f"- {reason} ({count})" for reason,count in reject_reasons.most_common(8)]
    summary += ["", "## Main skip reasons", ""]
    summary += [f"- {reason} ({count})" for reason,count in skip_reasons.most_common(8)]
    summary += ["", "## Evidence and source issues", "", f"- Batch 2 was planned for 430 questions, but the source files contain 200 Part A, 68 Part B, and 150 Part C rows. Twelve planned Part B items are absent and were not fabricated.", f"- {sum(1 for x in rejected if x['category']=='removed_evidence')} Part C records were rejected because a post-removal BM25/dense candidate still directly contained the original answer, or because the question was a confirmed duplicate.", f"- {sum(1 for x in skipped if x['batch']=='Batch 2' and x['category']=='naturally_unanswerable')} Part B rows remain SKIP because the source audit itself flags possible hidden evidence; these require semantic adjudication.", f"- Accepted rows with missing corpus evidence IDs: {len(missing_evidence)}.", "- The compact CSVs retain passage IDs and text; review decisions are represented by the accepted/rejected/skipped file partitions. Detailed CLI-only review columns were omitted at the user's request.", "", "## Output files", "", "All outputs are under `results/independent_final_review/`. The accepted file contains only ACCEPT rows. Rejected and skipped rows remain available for audit. The compact combined CSVs include passage ID, passage role, and passage text for each record where available; `removed_gold_reference` marks evidence excluded from Part C evaluation, while naturally unanswerable rows use a source hint or retrieved context. The original CSVs and corpus were not written.", ""]
    (OUT / "review_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT / "review_change_log.md").write_text("# Review change log\n\n- Created independent review outputs in `results/independent_final_review/`.\n- Source question CSVs, corpus, indices, frozen items, and all `my_decision`/`my_notes` fields were read-only.\n- Assigned one ACCEPT/REJECT/SKIP decision to every present source record; detailed CLI review columns were omitted at the user's request.\n- Combined CSVs use a compact shared schema with passage ID, passage role, and passage text.\n- Did not generate the 12 absent Batch 2 Part B items.\n", encoding="utf-8")
    print(json.dumps(audit["decision_counts"], indent=2))
    print(f"outputs: {OUT}")


if __name__ == "__main__":
    main()
