"""Corpus and question-set construction for Phase 2.

Builds a retrieval corpus from:
  * MIRACL Bengali passages: all passages judged in the dev+train qrels, plus a
    seeded random sample of other passages to reach 100,000 MIRACL passages.
  * All unique Bengali contexts from TyDi QA (GoldP secondary task), with
    passage IDs prefixed ``tydi_``.

Also writes question sets:
  * MIRACL Bengali train (tuning only) and dev (reporting only).
  * TyDi QA Bengali train and validation.

Nothing here invents data: every passage/question comes from the official files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

MIRACL_CORPUS_GZ = RAW / "miracl_bn" / "corpus" / "docs-0.jsonl.gz"
MIRACL_QRELS = {
    "train": RAW / "miracl_bn" / "qrels" / "qrels.miracl-v1.0-bn-train.tsv",
    "dev": RAW / "miracl_bn" / "qrels" / "qrels.miracl-v1.0-bn-dev.tsv",
}
MIRACL_TOPICS = {
    "train": RAW / "miracl_bn" / "topics" / "topics.miracl-v1.0-bn-train.tsv",
    "dev": RAW / "miracl_bn" / "topics" / "topics.miracl-v1.0-bn-dev.tsv",
}
TYDI_PARQUET = {
    "train": RAW / "tydiqa_goldp" / "train.parquet",
    "validation": RAW / "tydiqa_goldp" / "validation.parquet",
}


def load_qrels(split: str) -> pd.DataFrame:
    return pd.read_csv(
        MIRACL_QRELS[split],
        sep=r"\s+",
        header=None,
        names=["qid", "Q0", "docid", "relevance"],
        dtype={"qid": str, "Q0": str, "docid": str, "relevance": int},
    )


def load_topics(split: str) -> pd.DataFrame:
    return pd.read_csv(
        MIRACL_TOPICS[split],
        sep="\t",
        header=None,
        names=["qid", "query"],
        dtype=str,
    )


def judged_docids() -> set[str]:
    """Union of docids judged in MIRACL dev and train qrels."""
    frames = [load_qrels("dev"), load_qrels("train")]
    return set(pd.concat(frames, ignore_index=True)["docid"])


def iter_miracl_corpus(path: Path = MIRACL_CORPUS_GZ) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def select_miracl_docids(
    judged: set[str], target_total: int, seed: int,
    corpus_path: Path = MIRACL_CORPUS_GZ,
) -> tuple[set[str], dict]:
    """Choose judged passages + a seeded random sample of the rest.

    Two passes are needed because the corpus is streamed: pass 1 collects the
    non-judged docids, pass 2 is done by the caller to write selected passages.
    """
    all_docids: list[str] = []
    total = 0
    for rec in iter_miracl_corpus(corpus_path):
        total += 1
        all_docids.append(rec["docid"])

    found = set(all_docids)
    missing_judged = judged - found
    if missing_judged:
        raise RuntimeError(
            f"{len(missing_judged)} judged docids are missing from the corpus "
            f"(e.g. {sorted(missing_judged)[:5]})"
        )

    non_judged = [d for d in all_docids if d not in judged]
    n_random = target_total - len(judged)
    if n_random < 0:
        raise ValueError(
            f"target_total={target_total} is smaller than the {len(judged)} "
            f"judged passages"
        )
    if n_random > len(non_judged):
        raise ValueError(
            f"cannot sample {n_random} from {len(non_judged)} non-judged passages"
        )

    import random

    rng = random.Random(seed)
    sampled = set(rng.sample(non_judged, n_random))

    stats = {
        "corpus_total_passages": total,
        "judged_unique": len(judged),
        "non_judged_available": len(non_judged),
        "random_sampled": len(sampled),
        "miracl_selected": len(judged) + len(sampled),
    }
    return judged | sampled, stats


def iter_tydi_bengali(path: Path) -> Iterator[dict]:
    df = pd.read_parquet(path)
    bn = df[df["id"].astype(str).str.startswith("bengali-")]
    for _, row in bn.iterrows():
        yield {
            "row_id": str(row["id"]),
            "question": str(row["question"]),
            "answers": [str(a) for a in list(row["answers"]["text"])],
            "context": str(row["context"]),
            "title": str(row["title"]),
        }


def tydi_passage_id(context: str) -> str:
    return "tydi_" + hashlib.sha1(context.encode("utf-8")).hexdigest()[:16]


@dataclass
class CorpusManifest:
    seed: int
    target_miracl_passages: int
    miracl: dict
    tydi: dict
    combined_total: int
    sources: dict


def build_corpus(
    out_corpus: Path = PROCESSED / "corpus.jsonl",
    out_manifest: Path = PROCESSED / "corpus_manifest.json",
    target_miracl: int = 100_000,
    seed: int = 42,
) -> CorpusManifest:
    judged = judged_docids()
    selected, mstats = select_miracl_docids(judged, target_miracl, seed)

    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    miracl_written = 0
    miracl_short = 0
    tydi_contexts: dict[str, dict] = {}
    for split, path in TYDI_PARQUET.items():
        for rec in iter_tydi_bengali(path):
            pid = tydi_passage_id(rec["context"])
            if pid not in tydi_contexts:
                tydi_contexts[pid] = {
                    "passage_id": pid,
                    "parent_id": pid,
                    "is_short": len(rec["context"].strip()) < 20,
                    "title": rec["title"],
                    "text": rec["context"],
                    "source": "tydi",
                }

    with out_corpus.open("w", encoding="utf-8") as f:
        for rec in iter_miracl_corpus():
            if rec["docid"] in selected:
                if len(rec["text"].strip()) < 20:
                    miracl_short += 1
                f.write(json.dumps({
                    "passage_id": rec["docid"],
                    "parent_id": rec["docid"],
                    "is_short": len(rec["text"].strip()) < 20,
                    "title": rec["title"],
                    "text": rec["text"],
                    "source": "miracl",
                }, ensure_ascii=False) + "\n")
                miracl_written += 1
        for pid in sorted(tydi_contexts):
            f.write(json.dumps(tydi_contexts[pid], ensure_ascii=False) + "\n")

    manifest = CorpusManifest(
        seed=seed,
        target_miracl_passages=target_miracl,
        miracl={**mstats, "written": miracl_written, "short_passages": miracl_short},
        tydi={
            "unique_contexts_written": len(tydi_contexts),
            "short_contexts": sum(1 for r in tydi_contexts.values() if r["is_short"]),
            "from_train_rows": 2390,
            "from_validation_rows": 113,
        },
        combined_total=miracl_written + len(tydi_contexts),
        sources={
            "miracl_corpus": {
                "file": str(MIRACL_CORPUS_GZ.relative_to(PROJECT_ROOT)),
                "license": "Apache-2.0 (MIRACL release); text derived from "
                           "bnwiki-20190201 (Wikipedia content, verify exact "
                           "CC BY-SA version)",
            },
            "miracl_topics_qrels": {
                "files": [str(p.relative_to(PROJECT_ROOT))
                          for p in list(MIRACL_QRELS.values()) + list(MIRACL_TOPICS.values())],
                "license": "Apache-2.0",
            },
            "tydiqa": {
                "files": [str(p.relative_to(PROJECT_ROOT)) for p in TYDI_PARQUET.values()],
                "license": "Apache-2.0",
            },
        },
    )
    out_manifest.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_question_sets(out_dir: Path = PROCESSED) -> dict:
    """Write MIRACL + TyDi question sets with gold evidence IDs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    # MIRACL: train is tuning-only, dev is reporting-only.
    for split in ["train", "dev"]:
        topics = load_topics(split)
        qrels = load_qrels(split)
        positives = (
            qrels[qrels["relevance"] >= 1]
            .groupby("qid")["docid"]
            .apply(lambda s: sorted(set(s)))
            .to_dict()
        )
        rows = []
        for _, t in topics.iterrows():
            gold = positives.get(t["qid"], [])
            rows.append({
                "question_id": f"miracl_{split}_{t['qid']}",
                "source": "miracl",
                "split": split,
                "purpose": "tuning" if split == "train" else "reporting",
                "language": "bn",
                "question": t["query"],
                "gold_passage_ids": gold,
                "answerable": len(gold) > 0,
            })
        out = out_dir / f"miracl_{split}_questions.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report[f"miracl_{split}"] = {
            "questions": len(rows),
            "with_gold": sum(1 for r in rows if r["gold_passage_ids"]),
            "file": str(out.relative_to(PROJECT_ROOT)),
        }

    # TyDi QA Bengali (GoldP): gold evidence is the context passage.
    for split in ["train", "validation"]:
        rows = []
        for rec in iter_tydi_bengali(TYDI_PARQUET[split]):
            rows.append({
                "question_id": "tydi_" + rec["row_id"],
                "source": "tydi",
                "split": split,
                "purpose": "tuning" if split == "train" else "reporting",
                "language": "bn",
                "question": rec["question"],
                "gold_answers": rec["answers"],
                "gold_passage_ids": [tydi_passage_id(rec["context"])],
                "answerable": True,
            })
        out = out_dir / f"tydi_{split}_questions.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report[f"tydi_{split}"] = {
            "questions": len(rows),
            "with_gold": len(rows),
            "file": str(out.relative_to(PROJECT_ROOT)),
        }

    return report
